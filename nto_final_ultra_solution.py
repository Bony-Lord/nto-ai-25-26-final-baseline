# NTO Final "Потеряшки" — Single-cell strong baseline+ pipeline (Kaggle-ready)
from __future__ import annotations

import os
import gc
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from catboost import CatBoostRanker, Pool

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- Config ----------
@dataclass
class CFG:
    seed: int = 42
    top_k: int = 20

    # folds
    fold_days: int = 30
    hide_rate: float = 0.20

    # weighting
    w_read: float = 2.2
    w_wish: float = 1.0
    tau_days: float = 45.0
    tau_recent_days: float = 10.0

    # retrieval budgets
    als_topk: int = 180
    cooc_topk: int = 80
    content_topk: int = 50
    author_topk: int = 60
    trend_topk: int = 80
    max_candidates: int = 420

    # ALS
    als_factors: int = 128
    als_iterations: int = 20
    als_reg: float = 0.06

    # text
    tfidf_max_features: int = 120000
    svd_components: int = 96

    # ranking
    max_neg_per_user: int = 220
    cb_iterations: int = 1800
    cb_lr: float = 0.04

cfg = CFG()

# ---------- Data path ----------
def find_data_root() -> Path:
    candidates = [
        Path('/kaggle/input/datasets/artemnazemtsev/nto-ai'),
        Path('/kaggle/input/nto-ai'),
        Path('/kaggle/input'),
        Path('.'),
    ]
    req = ['interactions.csv','targets.csv','editions.csv','users.csv','authors.csv','genres.csv','book_genres.csv']
    for root in candidates:
        if root.is_dir() and all((root / f).exists() for f in req):
            return root
    for root in [Path('/kaggle/input'), Path('.')]:
        if not root.exists():
            continue
        for p in root.rglob('interactions.csv'):
            d = p.parent
            if all((d / f).exists() for f in req):
                return d
    raise FileNotFoundError('Data folder with required csv not found. Example: /kaggle/input/datasets/artemnazemtsev/nto-ai')

DATA_ROOT = find_data_root()
print('DATA_ROOT =', DATA_ROOT)

# ---------- Load ----------
interactions = pd.read_csv(DATA_ROOT / 'interactions.csv', low_memory=False)
interactions['event_ts'] = pd.to_datetime(interactions['event_ts'])
interactions = interactions[interactions['event_type'].isin([1,2])].copy()
interactions['user_id'] = interactions['user_id'].astype('int64')
interactions['edition_id'] = interactions['edition_id'].astype('int64')
interactions['event_type'] = interactions['event_type'].astype('int8')
if 'rating' in interactions:
    interactions['rating'] = pd.to_numeric(interactions['rating'], errors='coerce').astype('float32')
else:
    interactions['rating'] = np.nan

editions = pd.read_csv(DATA_ROOT / 'editions.csv', low_memory=False)
authors = pd.read_csv(DATA_ROOT / 'authors.csv', low_memory=False)
users = pd.read_csv(DATA_ROOT / 'users.csv', low_memory=False)
targets = pd.read_csv(DATA_ROOT / 'targets.csv')
book_genres = pd.read_csv(DATA_ROOT / 'book_genres.csv', low_memory=False)

for c in ['edition_id','book_id','author_id','language_id','publisher_id']:
    editions[c] = pd.to_numeric(editions[c], errors='coerce').fillna(-1).astype('int64')
editions['publication_year'] = pd.to_numeric(editions.get('publication_year'), errors='coerce').fillna(0).astype('float32')
editions['age_restriction'] = pd.to_numeric(editions.get('age_restriction'), errors='coerce').fillna(-1).astype('int16')
editions['title'] = editions.get('title','').fillna('').astype(str)
editions['description'] = editions.get('description','').fillna('').astype(str)
editions['text_data'] = (editions['title'] + ' ' + editions['description']).str.slice(0, 2500)

users['user_id'] = pd.to_numeric(users['user_id'], errors='coerce').astype('int64')
users['gender'] = pd.to_numeric(users.get('gender'), errors='coerce').fillna(-1).astype('int16')
users['age'] = pd.to_numeric(users.get('age'), errors='coerce').fillna(-1).astype('float32')
users['age_bucket'] = pd.cut(users['age'], bins=[-2,17,24,34,44,54,130], labels=[0,1,2,3,4,5]).astype('float').fillna(-1).astype('int16')

targets['user_id'] = targets['user_id'].astype('int64')

# ---------- Utility ----------
def add_weights(df: pd.DataFrame, ref_ts: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    age_days = ((ref_ts - out['event_ts']).dt.total_seconds() / 86400.0).clip(lower=0)
    base = np.where(out['event_type'].eq(2), cfg.w_read, cfg.w_wish).astype('float32')
    rating_adj = np.where(out['rating'].notna(), 1.0 + 0.06 * (out['rating'].fillna(0).to_numpy(np.float32)-3.0), 1.0).astype('float32')
    rating_adj = np.clip(rating_adj, 0.7, 1.3)
    out['w_full'] = base * (0.25 + 0.75 * np.exp(-age_days / cfg.tau_days)) * rating_adj
    out['w_recent'] = base * np.exp(-age_days / cfg.tau_recent_days) * rating_adj
    out['is_read'] = (out['event_type'] == 2).astype('int8')
    return out

def ndcg_at_k(y_true: list[int], y_pred: list[int], k=20) -> float:
    true_set = set(y_true)
    dcg = 0.0
    for i, item in enumerate(y_pred[:k], start=1):
        if item in true_set:
            dcg += 1.0 / math.log2(i + 1)
    m = min(len(true_set), k)
    if m == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg

def eval_ndcg(df_scored: pd.DataFrame, hidden_pairs: pd.DataFrame, k=20) -> float:
    truth = hidden_pairs.groupby('user_id')['edition_id'].apply(list).to_dict()
    pred = df_scored.sort_values(['user_id','score'], ascending=[True,False]).groupby('user_id')['edition_id'].apply(list).to_dict()
    users_eval = list(truth.keys())
    if not users_eval:
        return 0.0
    return float(np.mean([ndcg_at_k(truth[u], pred.get(u, []), k=k) for u in users_eval]))

def build_fold(inter: pd.DataFrame):
    t_max = inter['event_ts'].max().normalize() + pd.Timedelta(days=1)
    w_end = t_max
    w_start = w_end - pd.Timedelta(days=cfg.fold_days)

    history_before = inter[inter['event_ts'] < w_start][['user_id','edition_id']].drop_duplicates()
    window = inter[(inter['event_ts'] >= w_start) & (inter['event_ts'] < w_end)].copy()

    eligible = window.merge(history_before.assign(seen_before=1), on=['user_id','edition_id'], how='left')
    eligible = eligible[eligible['seen_before'].isna()].drop(columns='seen_before')
    pairs = eligible.groupby(['user_id','edition_id'], as_index=False).agg(last_ts=('event_ts','max'), cnt=('event_ts','size'), read=('event_type', lambda x: int((x==2).any())))

    rng = np.random.default_rng(cfg.seed + 100)
    keep = []
    for u, g in pairs.groupby('user_id'):
        n = len(g)
        if n == 0:
            continue
        p = (1.0 + 0.35 * g['read'].astype(float) + 0.25 * np.log1p(g['cnt'])).to_numpy(np.float64)
        p = p / p.sum()
        n_hide = max(1 if n >= 4 else 0, int(round(n * cfg.hide_rate)))
        n_hide = min(n_hide, n)
        if n_hide > 0:
            idx = rng.choice(g.index.to_numpy(), size=n_hide, replace=False, p=p)
            keep.extend(idx.tolist())
    hidden = pairs.loc[keep, ['user_id','edition_id']].drop_duplicates()

    observed = inter[inter['event_ts'] < w_end].copy()
    observed = observed.merge(hidden.assign(h=1), on=['user_id','edition_id'], how='left')
    observed = observed[observed['h'].isna()].drop(columns='h')
    users_fold = np.sort(hidden['user_id'].unique())
    return observed.reset_index(drop=True), hidden.reset_index(drop=True), users_fold, w_end

# ---------- Retrieval models ----------
class ALSRetriever:
    def __init__(self):
        self.user_to_idx = {}
        self.idx_to_item = None
        self.seen = None
        self.model = None
        self.user_f = None
        self.item_f = None

    def fit(self, pair_df: pd.DataFrame):
        users_ = np.sort(pair_df['user_id'].unique())
        items_ = np.sort(pair_df['edition_id'].unique())
        self.user_to_idx = {int(u): i for i, u in enumerate(users_)}
        item_to_idx = {int(i): j for j, i in enumerate(items_)}
        self.idx_to_item = items_

        rows = pair_df['user_id'].map(self.user_to_idx).to_numpy()
        cols = pair_df['edition_id'].map(item_to_idx).to_numpy()
        vals = pair_df['w_full'].to_numpy(np.float32)
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users_), len(items_)), dtype=np.float32)
        mat.sum_duplicates()
        self.seen = mat.tocsr()

        try:
            import implicit
            self.model = implicit.als.AlternatingLeastSquares(
                factors=cfg.als_factors,
                regularization=cfg.als_reg,
                iterations=cfg.als_iterations,
                random_state=cfg.seed,
                use_gpu=Path('/proc/driver/nvidia/version').exists(),
            )
            self.model.fit(mat.T.tocsr(), show_progress=True)
            self.user_f = self.model.user_factors.astype(np.float32)
            self.item_f = self.model.item_factors.astype(np.float32)
        except Exception:
            svd = TruncatedSVD(n_components=min(64, mat.shape[1]-1), random_state=cfg.seed)
            self.user_f = svd.fit_transform(mat).astype(np.float32)
            self.item_f = (svd.components_.T).astype(np.float32)
        return self

    def recommend(self, user_ids: np.ndarray, n=200) -> pd.DataFrame:
        if self.user_f is None or self.item_f is None:
            return pd.DataFrame(columns=['user_id','edition_id','als_rank','als_score'])
        rows = []
        for u in user_ids:
            if int(u) not in self.user_to_idx:
                continue
            ui = self.user_to_idx[int(u)]
            scores = self.user_f[ui] @ self.item_f.T
            seen_idx = self.seen[ui].indices
            scores[seen_idx] = -1e18
            k = min(n, len(scores))
            top = np.argpartition(-scores, k-1)[:k]
            order = np.argsort(-scores[top])
            top = top[order]
            rows.append(pd.DataFrame({'user_id':int(u), 'edition_id':self.idx_to_item[top].astype('int64'), 'als_rank':np.arange(1, len(top)+1), 'als_score':scores[top].astype('float32')}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['user_id','edition_id','als_rank','als_score'])

class CoocRetriever:
    def __init__(self):
        self.item_to_idx = {}
        self.idx_to_item = None
        self.item_user = None

    def fit(self, pair_df: pd.DataFrame):
        # read-centric co-visitation
        read_df = pair_df[pair_df['is_read'] > 0][['user_id','edition_id','w_recent']].copy()
        if read_df.empty:
            read_df = pair_df[['user_id','edition_id','w_recent']].copy()
        users_ = np.sort(read_df['user_id'].unique())
        items_ = np.sort(read_df['edition_id'].unique())
        u2i = {int(u): i for i, u in enumerate(users_)}
        self.item_to_idx = {int(it): j for j, it in enumerate(items_)}
        self.idx_to_item = items_
        rows = read_df['user_id'].map(u2i).to_numpy()
        cols = read_df['edition_id'].map(self.item_to_idx).to_numpy()
        vals = np.ones(len(read_df), dtype=np.float32)
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users_), len(items_)), dtype=np.float32)
        self.item_user = mat.T.tocsr()
        return self

    def neighbors_for_seed(self, seed_item: int, topk=100):
        if self.item_user is None or seed_item not in self.item_to_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        j = self.item_to_idx[int(seed_item)]
        col = self.item_user[j]  # (1, n_users)
        sim = (self.item_user @ col.T).toarray().ravel().astype(np.float32)
        sim[j] = 0.0
        if sim.sum() <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(topk, len(sim))
        idx = np.argpartition(-sim, k-1)[:k]
        idx = idx[np.argsort(-sim[idx])]
        return self.idx_to_item[idx].astype(np.int64), sim[idx]

class ContentRetriever:
    def __init__(self):
        self.item_to_idx = {}
        self.idx_to_item = None
        self.nn = None

    def fit(self, editions_df: pd.DataFrame):
        self.idx_to_item = editions_df['edition_id'].to_numpy(dtype=np.int64)
        self.item_to_idx = {int(x): i for i, x in enumerate(self.idx_to_item)}
        tfidf = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=(1,2), min_df=2, max_df=0.9)
        X = tfidf.fit_transform(editions_df['text_data'].fillna(''))
        n_comp = min(cfg.svd_components, max(8, X.shape[1]-1))
        svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed)
        Z = svd.fit_transform(X).astype(np.float32)
        Z = normalize(Z, norm='l2', axis=1)
        self.nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(Z)
        self.Z = Z
        return self

    def neighbors_for_seed(self, seed_item: int, topk=50):
        if self.nn is None or seed_item not in self.item_to_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        i = self.item_to_idx[int(seed_item)]
        k = min(topk + 1, len(self.idx_to_item))
        d, ind = self.nn.kneighbors(self.Z[i:i+1], n_neighbors=k)
        items = self.idx_to_item[ind[0]].astype(np.int64)
        sim = (1.0 - d[0]).astype(np.float32)
        mask = items != int(seed_item)
        return items[mask][:topk], sim[mask][:topk]

# ---------- Build context ----------
def build_context(observed: pd.DataFrame, ref_ts: pd.Timestamp):
    obs = add_weights(observed, ref_ts)
    pair = (obs.groupby(['user_id','edition_id'], as_index=False)
            .agg(events=('event_ts','size'),
                 is_read=('is_read','max'),
                 w_full=('w_full','sum'),
                 w_recent=('w_recent','sum'),
                 last_ts=('event_ts','max')))
    pair['last_days'] = ((ref_ts - pair['last_ts']).dt.total_seconds() / 86400.0).astype('float32')

    # user/item stats
    user_feat = pair.groupby('user_id', as_index=False).agg(
        user_seen=('edition_id','nunique'),
        user_events=('events','sum'),
        user_w_recent=('w_recent','sum'),
        user_last_days=('last_days','min'),
    )
    item_feat = pair.groupby('edition_id', as_index=False).agg(
        item_users=('user_id','nunique'),
        item_events=('events','sum'),
        item_w_recent=('w_recent','sum'),
    )

    recent7 = obs[obs['event_ts'] >= ref_ts - pd.Timedelta(days=7)].groupby('edition_id', as_index=False).agg(item_w7=('w_full','sum'))
    recent14 = obs[obs['event_ts'] >= ref_ts - pd.Timedelta(days=14)].groupby('edition_id', as_index=False).agg(item_w14=('w_full','sum'))
    recent30 = obs[obs['event_ts'] >= ref_ts - pd.Timedelta(days=30)].groupby('edition_id', as_index=False).agg(item_w30=('w_full','sum'))
    item_feat = item_feat.merge(recent7, on='edition_id', how='left').merge(recent14, on='edition_id', how='left').merge(recent30, on='edition_id', how='left')
    for c in ['item_w7','item_w14','item_w30']:
        item_feat[c] = item_feat[c].fillna(0).astype('float32')
    item_feat['item_momentum'] = ((item_feat['item_w7'] + 1.0) / (item_feat['item_w30'] + 1.0)).astype('float32')

    user_feat = user_feat.merge(users[['user_id','gender','age','age_bucket']], on='user_id', how='left')
    item_feat = item_feat.merge(editions[['edition_id','author_id','book_id','publication_year','age_restriction','language_id','publisher_id']], on='edition_id', how='left')

    # author affinity
    pair_meta = pair.merge(editions[['edition_id','author_id','book_id']], on='edition_id', how='left')
    ua = pair_meta.groupby(['user_id','author_id'], as_index=False).agg(ua_w_recent=('w_recent','sum'), ua_cnt=('edition_id','nunique'))
    ua['ua_rank'] = ua.groupby('user_id')['ua_w_recent'].rank(method='dense', ascending=False)

    author_item_pop = pair_meta.groupby(['author_id','edition_id'], as_index=False).agg(author_item_score=('w_recent','sum'))
    author_item_pop['author_item_rank'] = author_item_pop.groupby('author_id')['author_item_score'].rank(method='dense', ascending=False)

    trend = item_feat[['edition_id','item_w7','item_w14','item_momentum']].copy()
    trend['trend_score'] = trend['item_w7'] + 0.7 * trend['item_w14'] + 2.0 * trend['item_momentum']
    trend = trend.sort_values('trend_score', ascending=False).head(300)

    # seed items
    seeds = pair.sort_values(['user_id','w_recent','last_days'], ascending=[True,False,True]).groupby('user_id').head(8).copy()

    # models
    als = ALSRetriever().fit(pair)
    cooc = CoocRetriever().fit(pair)
    content = ContentRetriever().fit(editions[['edition_id','text_data']])

    seen = pair[['user_id','edition_id']].drop_duplicates()

    return {
        'obs': obs,
        'pair': pair,
        'user_feat': user_feat,
        'item_feat': item_feat,
        'ua': ua,
        'author_item_pop': author_item_pop,
        'trend': trend,
        'seeds': seeds,
        'als': als,
        'cooc': cooc,
        'content': content,
        'seen': seen,
    }

# ---------- Retrieval ----------
def retrieve_candidates(ctx: dict[str,Any], user_ids: np.ndarray) -> pd.DataFrame:
    src = []

    als_df = ctx['als'].recommend(user_ids, n=cfg.als_topk)
    if not als_df.empty:
        als_df['pre'] = 2.8 / (als_df['als_rank'] + 1)
        src.append(als_df[['user_id','edition_id','pre']])

    # author affinity retrieval
    ua_top = ctx['ua'][ctx['ua']['ua_rank'] <= 8].copy()
    ua_top = ua_top[ua_top['user_id'].isin(user_ids)]
    author_pop = ctx['author_item_pop'][ctx['author_item_pop']['author_item_rank'] <= 25].copy()
    au = ua_top.merge(author_pop, on='author_id', how='inner')
    if not au.empty:
        au['pre'] = (au['ua_w_recent'] * au['author_item_score']).astype('float32')
        au = au.sort_values(['user_id','pre'], ascending=[True,False]).groupby('user_id').head(cfg.author_topk)
        src.append(au[['user_id','edition_id','pre']])

    # trend retrieval
    trend = ctx['trend'][['edition_id','trend_score']].copy()
    if not trend.empty:
        uu = pd.DataFrame({'user_id': user_ids})
        uu['_k'] = 1
        trend['_k'] = 1
        tr = uu.merge(trend, on='_k', how='inner').drop(columns='_k')
        tr = tr.sort_values(['user_id','trend_score'], ascending=[True,False]).groupby('user_id').head(cfg.trend_topk)
        tr['pre'] = (0.6 * tr['trend_score']).astype('float32')
        src.append(tr[['user_id','edition_id','pre']])

    # seed-based cooc + content
    seeds = ctx['seeds'][ctx['seeds']['user_id'].isin(user_ids)]
    rows = []
    for u, g in seeds.groupby('user_id'):
        acc = defaultdict(float)
        seed_items = g['edition_id'].tolist()[:8]
        for rank_seed, s in enumerate(seed_items, start=1):
            w_seed = 1.0 / rank_seed
            items_c, sim_c = ctx['cooc'].neighbors_for_seed(int(s), topk=cfg.cooc_topk)
            for it, sim in zip(items_c, sim_c):
                acc[int(it)] += float(1.6 * w_seed * sim)
            items_t, sim_t = ctx['content'].neighbors_for_seed(int(s), topk=cfg.content_topk)
            for it, sim in zip(items_t, sim_t):
                acc[int(it)] += float(0.9 * w_seed * sim)
        if acc:
            arr = sorted(acc.items(), key=lambda x: -x[1])[: (cfg.cooc_topk + cfg.content_topk)]
            rows.extend([{'user_id': int(u), 'edition_id': int(it), 'pre': float(sc)} for it, sc in arr])
    if rows:
        src.append(pd.DataFrame(rows))

    if not src:
        return pd.DataFrame(columns=['user_id','edition_id','pre'])

    pool = pd.concat(src, ignore_index=True)
    pool = pool.groupby(['user_id','edition_id'], as_index=False).agg(pre=('pre','sum'))

    pool = pool.merge(ctx['seen'].assign(seen=1), on=['user_id','edition_id'], how='left')
    pool = pool[pool['seen'].isna()].drop(columns='seen')

    pool = pool.sort_values(['user_id','pre'], ascending=[True,False]).groupby('user_id').head(cfg.max_candidates)
    return pool.reset_index(drop=True)

# ---------- Features ----------
def build_features(ctx: dict[str,Any], candidates: pd.DataFrame, hidden: pd.DataFrame | None = None) -> pd.DataFrame:
    df = candidates.copy()
    df = df.merge(ctx['item_feat'], on='edition_id', how='left')
    df = df.merge(ctx['user_feat'], on='user_id', how='left')

    df = df.merge(editions[['edition_id','author_id','book_id','publication_year','age_restriction','language_id','publisher_id']], on='edition_id', how='left')
    df = df.merge(ctx['ua'][['user_id','author_id','ua_w_recent','ua_cnt']], on=['user_id','author_id'], how='left')

    # cross features
    df['f_user_item_ratio'] = (df['item_w_recent'].fillna(0) / np.maximum(df['user_w_recent'].fillna(0), 1e-6)).astype('float32')
    df['f_user_author_bias'] = (df['ua_w_recent'].fillna(0) / np.maximum(df['user_w_recent'].fillna(0), 1e-6)).astype('float32')
    df['f_age_restr_gap'] = (df['age'].fillna(-1) - df['age_restriction'].fillna(-1)).astype('float32')
    df['f_item_fresh'] = (2026 - df['publication_year'].fillna(0)).clip(0, 250).astype('float32')
    df['f_user_cold'] = (df['user_seen'].fillna(0) <= 2).astype('int8')
    df['f_item_tail'] = (df['item_users'].fillna(0) <= 2).astype('int8')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c in ('user_id','edition_id'):
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')

    cat_cols = ['author_id','book_id','language_id','publisher_id','gender','age_bucket']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype('int64').astype(str)

    if hidden is not None:
        df = df.merge(hidden.assign(label=1), on=['user_id','edition_id'], how='left')
        df['label'] = df['label'].fillna(0).astype('int8')

    return df

# ---------- Hard negatives ----------
def hard_negative_sample(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df['label'] == 1].copy()
    neg = df[df['label'] == 0].copy()
    if neg.empty:
        return pos
    neg = neg.sort_values(['user_id','pre'], ascending=[True,False])
    hard = neg.groupby('user_id').head(cfg.max_neg_per_user // 2)
    rnd_parts = []
    rng = np.random.default_rng(cfg.seed + 77)
    rest = neg.merge(hard[['user_id','edition_id']].assign(h=1), on=['user_id','edition_id'], how='left')
    rest = rest[rest['h'].isna()].drop(columns='h')
    n_rnd = cfg.max_neg_per_user - cfg.max_neg_per_user // 2
    for u, g in rest.groupby('user_id'):
        if len(g) <= n_rnd:
            rnd_parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=n_rnd, replace=False)
            rnd_parts.append(g.loc[idx])
    out = [pos, hard] + rnd_parts
    return pd.concat(out, ignore_index=True).sort_values(['user_id','pre'], ascending=[True,False]).reset_index(drop=True)

# ---------- Train/valid ----------
observed, hidden, fold_users, ref_ts = build_fold(interactions)
ctx = build_context(observed, ref_ts)
cand = retrieve_candidates(ctx, fold_users)

# inject missed positives to avoid training leakage from retrieval miss
miss = hidden.merge(cand[['user_id','edition_id']], on=['user_id','edition_id'], how='left', indicator=True)
miss = miss[miss['_merge']=='left_only'][['user_id','edition_id']]
if len(miss):
    miss['pre'] = 0.0
    cand = pd.concat([cand, miss], ignore_index=True).drop_duplicates(['user_id','edition_id'])

train_df = build_features(ctx, cand, hidden=hidden)
train_df = hard_negative_sample(train_df)

feature_cols = [c for c in train_df.columns if c not in ['label','user_id','edition_id']]
cat_cols = [c for c in ['author_id','book_id','language_id','publisher_id','gender','age_bucket'] if c in feature_cols]

# filter users with at least one positive
gcnt = train_df.groupby('user_id')['edition_id'].size()
gpos = train_df.groupby('user_id')['label'].sum()
ok_users = set(gcnt[gcnt >= 8].index) & set(gpos[gpos > 0].index)
train_df = train_df[train_df['user_id'].isin(ok_users)].copy()

pool = Pool(
    data=train_df[feature_cols],
    label=train_df['label'],
    group_id=train_df['user_id'],
    cat_features=cat_cols,
    feature_names=feature_cols,
)

task_type = 'GPU' if Path('/proc/driver/nvidia/version').exists() else 'CPU'
model = CatBoostRanker(
    loss_function='YetiRankPairwise' if task_type == 'GPU' else 'YetiRank',
    eval_metric='NDCG:top=20',
    task_type=task_type,
    random_seed=cfg.seed,
    iterations=cfg.cb_iterations,
    learning_rate=cfg.cb_lr,
    depth=8 if task_type == 'GPU' else 10,
    l2_leaf_reg=8.0,
    subsample=0.85,
    bootstrap_type='Bernoulli',
    verbose=200,
)
model.fit(pool)

# sanity metric on pseudo fold
train_scored = train_df[['user_id','edition_id']].copy()
train_scored['score'] = model.predict(Pool(train_df[feature_cols], cat_features=cat_cols, feature_names=feature_cols)).astype(np.float32)
cv_ndcg = eval_ndcg(train_scored, hidden, k=20)
print(f'Pseudo-fold NDCG@20: {cv_ndcg:.6f}')

# ---------- Final fit on full data ----------
full_ref = interactions['event_ts'].max().normalize() + pd.Timedelta(days=1)
full_ctx = build_context(interactions, full_ref)
target_user_ids = np.sort(targets['user_id'].unique())

test_cand = retrieve_candidates(full_ctx, target_user_ids)
test_df = build_features(full_ctx, test_cand, hidden=None)

for c in feature_cols:
    if c not in test_df.columns:
        test_df[c] = 'UNK' if c in cat_cols else 0.0
for c in cat_cols:
    test_df[c] = test_df[c].astype(str)

pred = model.predict(Pool(test_df[feature_cols], cat_features=cat_cols, feature_names=feature_cols)).astype(np.float32)
scored = test_df[['user_id','edition_id']].copy()
scored['score'] = pred

# fallback
seen_map = interactions.groupby('user_id')['edition_id'].apply(set).to_dict()
global_pop = full_ctx['item_feat'].sort_values(['item_w_recent','item_events'], ascending=False)['edition_id'].astype(int).tolist()

sub_rows = []
grouped = {int(u): g.sort_values('score', ascending=False)['edition_id'].astype(int).tolist() for u, g in scored.groupby('user_id')}
for u in target_user_ids:
    used = set()
    recs = []
    seen = seen_map.get(int(u), set())

    for it in grouped.get(int(u), []):
        if it in used or it in seen:
            continue
        used.add(it)
        recs.append(it)
        if len(recs) == cfg.top_k:
            break

    if len(recs) < cfg.top_k:
        for it in global_pop:
            if it in used or it in seen:
                continue
            used.add(it)
            recs.append(it)
            if len(recs) == cfg.top_k:
                break

    if len(recs) != cfg.top_k:
        raise RuntimeError(f'Unable to fill top-{cfg.top_k} for user {u}')

    for r, it in enumerate(recs, start=1):
        sub_rows.append({'user_id': int(u), 'edition_id': int(it), 'rank': int(r)})

submission = pd.DataFrame(sub_rows)
submission.to_csv('submission.csv', index=False)
print('saved submission.csv', submission.shape)
print(submission.head(20))
