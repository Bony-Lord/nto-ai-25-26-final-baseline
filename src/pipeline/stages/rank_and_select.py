"""Rank and select stage implementation."""

from __future__ import annotations

import pandas as pd

from src.core.dataset import Dataset
from src.pipeline.stage_helpers import StageHelpers


class RankAndSelectStage:
    """Rank candidates and persist final top-k predictions."""

    name = "rank_and_select"

    def __init__(self, helpers: StageHelpers) -> None:
        self.helpers = helpers

    def run(self) -> dict[str, Any]:
        dataset = Dataset.load(self.helpers.paths.data_dir)
        interactions, seen_positive = self.helpers.read_data_cache()
        dataset = Dataset(
            interactions_df=interactions,
            targets_df=dataset.targets_df,
            catalog_df=dataset.catalog_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=seen_positive,
        )
        candidates = pd.read_parquet(self.helpers.paths.candidates_path)
        predictions = self.helpers.rank_predictions(dataset, candidates)
        self.helpers.write_dataframe(predictions, self.helpers.paths.predictions_path)
        return {
            "rows": int(len(predictions)),
            "users": int(predictions["user_id"].nunique() if not predictions.empty else 0),
        }

