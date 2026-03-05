"""Build features stage implementation."""

from __future__ import annotations

from typing import Any

from src.core.artifacts import atomic_write_dataframe
from src.participants.features import build_features_frame
from src.pipeline.models import PipelineContext
from src.pipeline.runtime import load_runtime_dataset


class BuildFeaturesStage:
    """Compute baseline feature artifacts."""

    name = "build_features"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        dataset = load_runtime_dataset(self.context.paths)
        features = build_features_frame(
            dataset=dataset,
            recent_days=int(self.context.config["pipeline"]["recent_days"]),
        )
        atomic_write_dataframe(features, self.context.paths.features_path)
        return {
            "rows": int(len(features)),
            "users": int(dataset.targets_df["user_id"].nunique()),
        }

