"""User language and publisher affinity generator."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.platform.core.dataset import Dataset


class UserLanguagePublisherGenerator:
    """Project user language and publisher affinities to candidate editions."""

    name = "user_language_publisher"

    def __init__(
        self,
        language_weight: float = 1.0,
        publisher_weight: float = 0.8,
        smoothing: float = 1.0,
        show_progress: bool = False,
    ) -> None:
        self.language_weight = language_weight
        self.publisher_weight = publisher_weight
        self.smoothing = smoothing
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        del seed
        pop = features[features["feature_type"] == "edition_popularity_all"][["edition_id", "value"]].copy()
        pop_map = dict(zip(pop["edition_id"].astype(int), pop["value"].astype(float)))

        lang_profile = features[features["feature_type"] == "user_language_profile"][
            ["user_id", "genre_id", "value"]
        ].rename(columns={"genre_id": "language_id", "value": "lang_weight"})
        pub_profile = features[features["feature_type"] == "user_publisher_profile"][
            ["user_id", "genre_id", "value"]
        ].rename(columns={"genre_id": "publisher_id", "value": "pub_weight"})
        if lang_profile.empty and pub_profile.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        catalog = dataset.catalog_df[["edition_id", "language_id", "publisher_id"]].copy()
        catalog["pop"] = catalog["edition_id"].map(pop_map).fillna(0.0)

        rows: list[dict[str, int | float | str]] = []
        lang_by_user = {
            int(uid): group[["language_id", "lang_weight"]].copy()
            for uid, group in lang_profile.groupby("user_id")
        }
        pub_by_user = {
            int(uid): group[["publisher_id", "pub_weight"]].copy()
            for uid, group in pub_profile.groupby("user_id")
        }

        for user_id in tqdm(
            user_ids.tolist(),
            total=len(user_ids),
            desc=f"{self.name}_users",
            leave=False,
            dynamic_ncols=True,
            disable=not (self.show_progress and sys.stdout.isatty()),
            file=sys.stdout,
        ):
            user_id_int = int(user_id)
            score_frame = catalog[["edition_id", "language_id", "publisher_id", "pop"]].copy()
            score_frame["lang_weight"] = 0.0
            score_frame["pub_weight"] = 0.0

            if user_id_int in lang_by_user:
                score_frame = score_frame.merge(
                    lang_by_user[user_id_int], on="language_id", how="left", suffixes=("", "_new")
                )
                score_frame["lang_weight"] = score_frame["lang_weight_new"].fillna(0.0)
                score_frame = score_frame.drop(columns=["lang_weight_new"])

            if user_id_int in pub_by_user:
                score_frame = score_frame.merge(
                    pub_by_user[user_id_int], on="publisher_id", how="left", suffixes=("", "_new")
                )
                score_frame["pub_weight"] = score_frame["pub_weight_new"].fillna(0.0)
                score_frame = score_frame.drop(columns=["pub_weight_new"])

            score_frame["score"] = (
                self.language_weight * score_frame["lang_weight"]
                + self.publisher_weight * score_frame["pub_weight"]
            ) * (score_frame["pop"] + self.smoothing)

            top = score_frame.sort_values(["score", "edition_id"], ascending=[False, True]).head(k)
            for row in top.itertuples(index=False):
                if float(row.score) <= 0.0:
                    continue
                rows.append(
                    {
                        "user_id": user_id_int,
                        "edition_id": int(row.edition_id),
                        "score": float(row.score),
                        "source": self.name,
                    }
                )

        return pd.DataFrame(rows, columns=["user_id", "edition_id", "score", "source"])
