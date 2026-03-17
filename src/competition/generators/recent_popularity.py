"""Recent popularity generator for incident-adjacent recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class RecentPopularityGenerator:
    """Recommend globally recent items to capture incident-time trends."""

    name = "recent_popularity"

    def __init__(self, decay_days: float = 14.0, show_progress: bool = False) -> None:
        self.decay_days = decay_days
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        del seed, features
        interactions = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])].copy()
        if interactions.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        max_ts = interactions["event_ts"].max()
        age_days = (max_ts - interactions["event_ts"]).dt.total_seconds() / 86400.0
        interactions["w"] = np.exp(-age_days / max(self.decay_days, 1e-6))
        weighted = (
            interactions.groupby("edition_id", as_index=False)["w"]
            .sum()
            .sort_values(["w", "edition_id"], ascending=[False, True])
            .head(k)
        )
        if weighted.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        payload = []
        for uid in user_ids.tolist():
            for row in weighted.itertuples(index=False):
                payload.append(
                    {
                        "user_id": int(uid),
                        "edition_id": int(row.edition_id),
                        "score": float(row.w),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(payload, columns=["user_id", "edition_id", "score", "source"])
