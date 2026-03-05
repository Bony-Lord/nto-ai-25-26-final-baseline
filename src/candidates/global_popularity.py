"""Global popularity candidate generator."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.dataset import Dataset


class GlobalPopularityGenerator:
    """Recommend globally popular editions to each user."""

    name = "global_popularity"

    def __init__(self, show_progress: bool = False) -> None:
        self.show_progress = show_progress

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        popularity = features[
            features["feature_type"] == "edition_popularity_all"
        ][["edition_id", "value"]].copy()
        popularity = popularity.sort_values(["value", "edition_id"], ascending=[False, True]).head(k)

        if popularity.empty:
            return pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])

        rows: list[dict[str, float | int | str]] = []
        for user_id in tqdm(
            user_ids.tolist(),
            total=len(user_ids),
            desc=f"{self.name}_users",
            leave=False,
            dynamic_ncols=True,
            disable=not (self.show_progress and sys.stdout.isatty()),
            file=sys.stdout,
        ):
            for _, row in popularity.iterrows():
                rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(row["edition_id"]),
                        "score": float(row["value"]),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows)

