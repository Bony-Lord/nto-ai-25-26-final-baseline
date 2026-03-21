"""Kaggle GPU solution for NTO AI "Потеряшки" final.

Pipeline:
1) load data
2) feature engineering (user/item + text)
3) candidate generation (ALS U2I + ALS I2I + user genre/author + global pop)
4) CatBoost ranking
5) submission.csv (top-20 per user)
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from implicit.als import AlternatingLeastSquares
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


SEED = 42
TOP_K = 20
CANDIDATES_PER_USER = 200
INCIDENT_DAYS = 31
HOLDOUT_DAYS = 31

TOP_GENRE_PER_USER = 10
TOP_AUTHOR_PER_USER = 10
EDS_PER_GENRE = 40
EDS_PER_AUTHOR = 30
GLOBAL_POP_CANDS = 100

ALS_FACTORS = 64
ALS_ITERS = 20
ALS_REG = 0.01
ALS_ALPHA = 1.0


@dataclass
class DataBundle:
    interactions: pd.DataFrame
    targets: pd.DataFrame
    editions: pd.DataFrame
    book_genres: pd.DataFrame
    users: pd.DataFrame


def _detect_data_dir() -> str:
    candidates = [
        "/kaggle/input/nto-ai",
        "/kaggle/input/datasets/artemnazemtsev/nto-ai",
        "./data",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "interactions.csv")):
            return path
    raise FileNotFoundError("Cannot find dataset directory with interactions.csv")


def load_data(data_dir: str) -> DataBundle:
    interactions = pd.read_csv(
        os.path.join(data_dir, "interactions.csv"),
        parse_dates=["event_ts"],
        dtype={"user_id": "int64", "edition_id": "int64", "event_type": "int8"},
    )
    targets = pd.read_csv(os.path.join(data_dir, "targets.csv"), dtype={"user_id": "int64"})
    editions = pd.read_csv(os.path.join(data_dir, "editions.csv"))
    book_genres = pd.read_csv(os.path.join(data_dir, "book_genres.csv"))

    users_path = os.path.join(data_dir, "users.csv")
    if os.path.exists(users_path):
        users = pd.read_csv(users_path)
    else:
        users = pd.DataFrame({"user_id": interactions["user_id"].unique()})

    return DataBundle(interactions, targets, editions, book_genres, users)


def preprocess_text(editions: pd.DataFrame) -> pd.DataFrame:
    stemmer = SnowballStemmer("russian")
    sw = set(stopwords.words("russian")) | set(stopwords.words("english"))
    re_html = re.compile(r"<[^>]+>")
    re_chars = re.compile(r"[^а-яёa-z\s]")

    def clean(series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str)
        s = s.map(lambda x: unicodedata.normalize("NFKC", x))
        s = s.str.replace(re_html, " ", regex=True).str.lower()
        s = s.str.replace(re_chars, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        return s

    def stem_text(text: str, max_tokens: int) -> str:
        tokens = []
        for tok in text.split():
            if len(tok) >= 3 and tok not in sw:
                tokens.append(stemmer.stem(tok))
            if len(tokens) >= max_tokens:
                break
        return " ".join(tokens)

    out = editions.copy()
    title = clean(out.get("title", pd.Series(dtype=str)))
    desc = clean(out.get("description", pd.Series(dtype=str)))
    out["title_clean"] = title.map(lambda x: stem_text(x, 30))
    out["desc_clean"] = desc.map(lambda x: stem_text(x, 120))
    return out


def build_profiles(interactions: pd.DataFrame, editions: pd.DataFrame, book_genres: pd.DataFrame):
    pos = interactions[interactions["event_type"].isin([1, 2])]
    e2b = editions[["edition_id", "book_id"]].drop_duplicates()
    e2a = editions[["edition_id", "author_id"]].drop_duplicates()

    ug = (
        pos.merge(e2b, on="edition_id", how="left")
        .merge(book_genres, on="book_id", how="left")
        .dropna(subset=["genre_id"])
        .groupby(["user_id", "genre_id"])
        .size()
        .rename("cnt")
        .reset_index()
    )
    ug["user_genre_affinity"] = ug["cnt"] / ug.groupby("user_id")["cnt"].transform("sum")

    ua = (
        pos.merge(e2a, on="edition_id", how="left")
        .dropna(subset=["author_id"])
        .groupby(["user_id", "author_id"])
        .size()
        .rename("cnt")
        .reset_index()
    )
    ua["user_author_affinity"] = ua["cnt"] / ua.groupby("user_id")["cnt"].transform("sum")
    return ug[["user_id", "genre_id", "user_genre_affinity"]], ua[["user_id", "author_id", "user_author_affinity"]]


def build_item_user_features(interactions, editions, users, book_genres, t_incident):
    pos = interactions[interactions["event_type"].isin([1, 2])]

    item = editions.copy()
    item = item.merge(pos.groupby("edition_id").size().rename("edition_pop"), on="edition_id", how="left")
    item = item.merge(pos[pos.event_type.eq(1)].groupby("edition_id").size().rename("wishlist_pop"), on="edition_id", how="left")
    item = item.merge(pos[pos.event_type.eq(2)].groupby("edition_id").size().rename("read_pop"), on="edition_id", how="left")
    item = item.merge(
        pos[pos.event_ts.ge(t_incident)].groupby("edition_id").size().rename("edition_recent_pop"),
        on="edition_id",
        how="left",
    )
    item = item.merge(pos.groupby("edition_id")["user_id"].nunique().rename("edition_unique_users"), on="edition_id", how="left")
    item = item.merge(book_genres.groupby("book_id")["genre_id"].count().rename("book_genre_count"), on="book_id", how="left")
    for c in ["edition_pop", "wishlist_pop", "read_pop", "edition_recent_pop", "edition_unique_users", "book_genre_count"]:
        item[c] = item[c].fillna(0).astype("float32")
    item["wishlist_ratio"] = item["wishlist_pop"] / (item["edition_pop"] + 1)
    item["read_ratio"] = item["read_pop"] / (item["edition_pop"] + 1)
    item["pop_log"] = np.log1p(item["edition_pop"])
    item["recent_pop_log"] = np.log1p(item["edition_recent_pop"])
    item["title_len"] = item["title_clean"].str.split().str.len().fillna(0).astype("float32")
    item["desc_len"] = item["desc_clean"].str.split().str.len().fillna(0).astype("float32")

    user = users[["user_id"]].drop_duplicates().copy()
    if "gender" in users.columns:
        user["gender"] = users["gender"].fillna(0)
    if "age" in users.columns:
        user["age"] = users["age"].fillna(0)
    user = user.merge(pos.groupby("user_id").size().rename("user_activity"), on="user_id", how="left")
    user = user.merge(pos[pos.event_type.eq(1)].groupby("user_id").size().rename("user_wishlist_count"), on="user_id", how="left")
    user = user.merge(pos[pos.event_type.eq(2)].groupby("user_id").size().rename("user_read_count"), on="user_id", how="left")
    user = user.merge(pos.groupby("user_id")["edition_id"].nunique().rename("user_uniq_editions"), on="user_id", how="left")
    for c in ["user_activity", "user_wishlist_count", "user_read_count", "user_uniq_editions", "gender", "age"]:
        if c in user.columns:
            user[c] = user[c].fillna(0).astype("float32")
    user["user_wish_ratio"] = user["user_wishlist_count"] / (user["user_activity"] + 1)
    user["user_read_ratio"] = user["user_read_count"] / (user["user_activity"] + 1)
    user["user_activity_log"] = np.log1p(user["user_activity"])
    return user, item


def train_als(interactions: pd.DataFrame):
    pos = interactions[interactions["event_type"].isin([1, 2])].copy()
    pos["base_w"] = pos["event_type"].map({1: 1.0, 2: 2.0}).astype("float32")
    max_date = pos["event_ts"].max()
    pos["days_old"] = (max_date - pos["event_ts"]).dt.days.clip(lower=0)
    pos["w"] = pos["base_w"] * np.exp(-np.log(2) * pos["days_old"] / 30)

    ue = LabelEncoder()
    ie = LabelEncoder()
    u_idx = ue.fit_transform(pos["user_id"])
    i_idx = ie.fit_transform(pos["edition_id"])
    mat = csr_matrix((pos["w"].to_numpy(), (u_idx, i_idx)), shape=(len(ue.classes_), len(ie.classes_)))

    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERS,
        regularization=ALS_REG,
        alpha=ALS_ALPHA,
        random_state=SEED,
        use_gpu=True,
    )
    model.fit(mat)
    u2idx = {u: i for i, u in enumerate(ue.classes_)}
    e2idx = {e: i for i, e in enumerate(ie.classes_)}
    return model, u2idx, e2idx, ue.classes_, ie.classes_


def generate_candidates(user_ids, interactions, item_features, book_genres, ug, ua, als_model, u2idx, e2idx, als_item_ids):
    pos = interactions[interactions["event_type"].isin([1, 2])]
    seen = set(zip(pos["user_id"], pos["edition_id"]))

    genre_index = (
        interactions[["edition_id"]]
        .merge(item_features[["edition_id", "book_id", "pop_log"]], on="edition_id", how="right")
        .merge(book_genres[["book_id", "genre_id"]], on="book_id", how="left")
        .sort_values(["genre_id", "pop_log"], ascending=[True, False])
        .groupby("genre_id")["edition_id"]
        .apply(lambda x: x.head(EDS_PER_GENRE).tolist())
        .to_dict()
    )
    author_index = (
        item_features.sort_values(["author_id", "pop_log"], ascending=[True, False])
        .groupby("author_id")["edition_id"]
        .apply(lambda x: x.head(EDS_PER_AUTHOR).tolist())
        .to_dict()
    )
    global_top = item_features.sort_values("pop_log", ascending=False)["edition_id"].head(GLOBAL_POP_CANDS).tolist()

    ug_map = ug.sort_values(["user_id", "user_genre_affinity"], ascending=[True, False]).groupby("user_id")["genre_id"].apply(list).to_dict()
    ua_map = ua.sort_values(["user_id", "user_author_affinity"], ascending=[True, False]).groupby("user_id")["author_id"].apply(list).to_dict()
    history = pos.sort_values("event_ts").groupby("user_id")["edition_id"].apply(lambda x: list(x)[-5:]).to_dict()

    rows = []
    item_factors = np.asarray(als_model.item_factors)
    for user_id in user_ids:
        cand = {}

        def add(eid, score, src):
            if (user_id, eid) in seen:
                return
            if eid not in cand:
                cand[eid] = {"score": 0.0, "src_genre": 0, "src_author": 0, "src_global": 0, "src_als_u2i": 0, "src_als_i2i": 0}
            cand[eid]["score"] += score
            cand[eid][src] = 1

        if user_id in u2idx:
            u_vec = np.asarray(als_model.user_factors)[u2idx[user_id]]
            scores = u_vec @ item_factors.T
            top_idx = np.argpartition(scores, -120)[-120:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            for r, idx in enumerate(top_idx):
                add(int(als_item_ids[idx]), 4.0 / (r + 1), "src_als_u2i")

        for h in history.get(user_id, []):
            if h not in e2idx:
                continue
            ids, sims = als_model.similar_items(e2idx[h], N=20)
            for r, (idx, sim) in enumerate(zip(ids, sims)):
                eid = int(als_item_ids[idx])
                if eid != h:
                    add(eid, float(sim) * 3.0 / (r + 1), "src_als_i2i")

        for g_rank, genre_id in enumerate(ug_map.get(user_id, [])[:TOP_GENRE_PER_USER]):
            for eid in genre_index.get(genre_id, []):
                add(int(eid), 1.5 / (g_rank + 1), "src_genre")
        for a_rank, author_id in enumerate(ua_map.get(user_id, [])[:TOP_AUTHOR_PER_USER]):
            for eid in author_index.get(author_id, []):
                add(int(eid), 1.5 / (a_rank + 1), "src_author")
        for r, eid in enumerate(global_top):
            add(int(eid), 0.5 / (r + 1), "src_global")

        top = sorted(cand.items(), key=lambda kv: kv[1]["score"], reverse=True)[:CANDIDATES_PER_USER]
        for eid, info in top:
            rows.append({"user_id": user_id, "edition_id": eid, "cand_score": info["score"], **{k: info[k] for k in info if k != "score"}})
    return pd.DataFrame(rows)


def main() -> None:
    data_dir = _detect_data_dir()
    data = load_data(data_dir)
    editions_local = preprocess_text(data.editions)

    t_max = data.interactions["event_ts"].max()
    t_incident = t_max - pd.Timedelta(days=INCIDENT_DAYS)
    t_train_end = t_incident - pd.Timedelta(days=HOLDOUT_DAYS)

    train_log = data.interactions[data.interactions["event_ts"] < t_train_end].copy()
    holdout_log = data.interactions[(data.interactions["event_ts"] >= t_train_end) & (data.interactions["event_ts"] < t_incident)].copy()

    user_features, item_features = build_item_user_features(train_log, editions_local, data.users, data.book_genres, t_incident)
    ug, ua = build_profiles(train_log, editions_local, data.book_genres)
    als_model, u2idx, e2idx, _, als_item_ids = train_als(train_log)

    train_users = holdout_log["user_id"].drop_duplicates().tolist()
    train_cands = generate_candidates(train_users, train_log, item_features, data.book_genres, ug, ua, als_model, u2idx, e2idx, als_item_ids)

    pos_pairs = set(zip(holdout_log[holdout_log["event_type"].isin([1, 2])]["user_id"], holdout_log[holdout_log["event_type"].isin([1, 2])]["edition_id"]))
    train_cands["target"] = train_cands.apply(lambda r: 1 if (r.user_id, r.edition_id) in pos_pairs else 0, axis=1).astype("int8")
    ds = train_cands.merge(user_features, on="user_id", how="left").merge(item_features, on="edition_id", how="left")
    ds = ds.merge(ug, on=["user_id", "genre_id"], how="left") if "genre_id" in ds.columns else ds
    ds = ds.merge(ua, on=["user_id", "author_id"], how="left") if "author_id" in ds.columns else ds
    ds["user_genre_affinity"] = ds.get("user_genre_affinity", 0).fillna(0)
    ds["user_author_affinity"] = ds.get("user_author_affinity", 0).fillna(0)

    feature_cols = [
        "cand_score", "src_genre", "src_author", "src_global", "src_als_u2i", "src_als_i2i",
        "edition_pop", "wishlist_pop", "read_pop", "edition_recent_pop", "edition_unique_users", "book_genre_count",
        "wishlist_ratio", "read_ratio", "pop_log", "recent_pop_log", "title_len", "desc_len",
        "gender", "age", "user_activity", "user_wishlist_count", "user_read_count", "user_uniq_editions",
        "user_wish_ratio", "user_read_ratio", "user_activity_log", "user_genre_affinity", "user_author_affinity",
    ]
    text_features = ["title_clean", "desc_clean"]

    train_pool = Pool(
        data=ds[feature_cols + text_features],
        label=ds["target"],
        group_id=ds["user_id"],
        text_features=text_features,
    )
    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=20",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        random_seed=SEED,
        task_type="GPU",
        verbose=200,
    )
    model.fit(train_pool)

    full_user_features, full_item_features = build_item_user_features(data.interactions, editions_local, data.users, data.book_genres, t_incident)
    full_ug, full_ua = build_profiles(data.interactions, editions_local, data.book_genres)
    full_als, full_u2idx, full_e2idx, _, full_item_ids = train_als(data.interactions)

    test_cands = generate_candidates(data.targets["user_id"].tolist(), data.interactions, full_item_features, data.book_genres, full_ug, full_ua, full_als, full_u2idx, full_e2idx, full_item_ids)
    test_ds = test_cands.merge(full_user_features, on="user_id", how="left").merge(full_item_features, on="edition_id", how="left")
    pred = model.predict(Pool(test_ds[feature_cols + text_features], text_features=text_features))
    test_ds["score"] = pred

    test_ds = test_ds.sort_values(["user_id", "score"], ascending=[True, False])
    test_ds["rank"] = test_ds.groupby("user_id").cumcount() + 1
    submission = test_ds[test_ds["rank"] <= TOP_K][["user_id", "edition_id", "rank"]]

    # fill missing users / ranks with global pop
    global_top = full_item_features.sort_values("pop_log", ascending=False)["edition_id"].tolist()
    out_rows = []
    by_user = {uid: g for uid, g in submission.groupby("user_id")}
    for uid in data.targets["user_id"]:
        used = set()
        if uid in by_user:
            for _, row in by_user[uid].iterrows():
                out_rows.append((uid, int(row["edition_id"]), int(row["rank"])))
                used.add(int(row["edition_id"]))
        rank = len(used) + 1
        for eid in global_top:
            if rank > TOP_K:
                break
            if eid not in used:
                out_rows.append((uid, int(eid), rank))
                used.add(int(eid))
                rank += 1

    submission_final = pd.DataFrame(out_rows, columns=["user_id", "edition_id", "rank"])
    submission_final.to_csv("submission.csv", index=False)
    print("Saved submission.csv", submission_final.shape)


if __name__ == "__main__":
    import nltk

    nltk.download("stopwords", quiet=True)
    main()
