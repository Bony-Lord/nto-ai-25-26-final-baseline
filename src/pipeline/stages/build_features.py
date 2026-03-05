"""Build features stage implementation."""

from __future__ import annotations

from src.core.dataset import Dataset
from src.pipeline.stage_helpers import StageHelpers


class BuildFeaturesStage:
    """Compute baseline feature artifacts."""

    name = "build_features"

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
        features = self.helpers.build_features_frame(dataset)
        self.helpers.write_dataframe(features, self.helpers.paths.features_path)
        return {
            "rows": int(len(features)),
            "users": int(dataset.targets_df["user_id"].nunique()),
        }

