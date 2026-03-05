"""Prepare data stage implementation."""

from __future__ import annotations

from src.core.dataset import Dataset
from src.pipeline.stage_helpers import StageHelpers


class PrepareDataStage:
    """Normalize inputs and build data cache artifact."""

    name = "prepare_data"

    def __init__(self, helpers: StageHelpers) -> None:
        self.helpers = helpers

    def run(self) -> dict[str, Any]:
        dataset = Dataset.load(self.helpers.paths.data_dir)
        data_cache = self.helpers.pack_data_cache(dataset)
        self.helpers.write_dataframe(data_cache, self.helpers.paths.data_cache_path)
        return {
            "rows": int(len(data_cache)),
            "users": int(dataset.interactions_df["user_id"].nunique()),
            "positives": int(len(dataset.seen_positive_df)),
        }

