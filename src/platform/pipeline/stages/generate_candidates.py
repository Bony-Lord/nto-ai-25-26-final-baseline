"""Candidate-generation stage using competition generator registry."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from src.competition.generators import run_generators
from src.platform.core.artifacts import atomic_write_dataframe
from src.platform.pipeline.models import PipelineContext
from src.platform.pipeline.runtime import load_runtime_dataset


class GenerateCandidatesStage:
    """Run configured generators and persist candidate artifact."""

    name = "generate_candidates"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        """Generate candidate rows for all target users.

        Returns:
            Dictionary with row/user/source counts for metadata reporting.
        """
        dataset = load_runtime_dataset(self.context.paths)
        features = pd.read_parquet(self.context.paths.features_path)
        user_ids = dataset.targets_df["user_id"].drop_duplicates().astype("int64")
        logs_cfg = self.context.config.get("logs", {})
        tqdm_enabled = bool(logs_cfg.get("tqdm_enabled", True)) and sys.stdout.isatty()
        candidates = run_generators(
            dataset=dataset,
            features=features,
            user_ids=user_ids,
            generators_cfg=list(self.context.config["candidates"]["generators"]),
            per_generator_k=int(self.context.config["candidates"]["per_generator_k"]),
            seed=int(self.context.config["pipeline"]["seed"]),
            tqdm_enabled=tqdm_enabled,
        )
        atomic_write_dataframe(candidates, self.context.paths.candidates_path)
        return {
            "rows": int(len(candidates)),
            "users": int(candidates["user_id"].nunique() if not candidates.empty else 0),
            "sources": int(candidates["source"].nunique() if not candidates.empty else 0),
        }

