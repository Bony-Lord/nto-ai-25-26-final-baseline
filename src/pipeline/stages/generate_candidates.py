"""Generate candidates stage implementation."""

from __future__ import annotations

import pandas as pd

from src.core.dataset import Dataset
from src.pipeline.stage_helpers import StageHelpers


class GenerateCandidatesStage:
    """Run candidate generators and save candidate artifact."""

    name = "generate_candidates"

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
        features = pd.read_parquet(self.helpers.paths.features_path)
        user_ids = dataset.targets_df["user_id"].drop_duplicates().astype("int64")
        candidates = self.helpers.run_generators(dataset, features, user_ids)
        self.helpers.write_dataframe(candidates, self.helpers.paths.candidates_path)
        return {
            "rows": int(len(candidates)),
            "users": int(candidates["user_id"].nunique() if not candidates.empty else 0),
            "sources": int(candidates["source"].nunique() if not candidates.empty else 0),
        }

