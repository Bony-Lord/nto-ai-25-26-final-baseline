"""Compatibility helper facade for migrated participant/core modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.artifacts import atomic_write_dataframe
from src.core.dataset import Dataset
from src.participants.features import build_features_frame
from src.participants.generators import run_generators, validate_candidate_contract
from src.participants.ranking import rank_predictions
from src.participants.validation import validate_submission
from src.pipeline.models import PipelineContext, PipelinePaths
from src.pipeline.runtime import load_runtime_dataset, pack_data_cache


class StageHelpers:
    """Thin compatibility layer around participant/core entrypoints."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    @property
    def config(self) -> dict[str, Any]:
        return self.context.config

    @property
    def paths(self) -> PipelinePaths:
        return self.context.paths

    def is_tqdm_enabled(self) -> bool:
        """Return whether tqdm progress bars should be enabled."""
        logs_cfg = self.config.get("logs", {})
        enabled_cfg = bool(logs_cfg.get("tqdm_enabled", True))
        return enabled_cfg

    def pack_data_cache(self, dataset: Dataset) -> pd.DataFrame:
        """Pack interactions and seen positives into one cache file."""
        return pack_data_cache(dataset)

    def read_data_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read packed data cache and split into interactions/seen positives."""
        dataset = load_runtime_dataset(self.paths)
        return dataset.interactions_df, dataset.seen_positive_df

    def build_features_frame(self, dataset: Dataset) -> pd.DataFrame:
        """Build all default feature frames into one long table."""
        return build_features_frame(
            dataset=dataset,
            recent_days=int(self.config["pipeline"]["recent_days"]),
        )

    def validate_candidate_contract(
        self, frame: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Validate candidate generator output schema and types."""
        return validate_candidate_contract(frame, source_name)

    def run_generators(
        self, dataset: Dataset, features: pd.DataFrame, user_ids: pd.Series
    ) -> pd.DataFrame:
        """Execute configured generators and return concatenated candidates."""
        return run_generators(
            dataset=dataset,
            features=features,
            user_ids=user_ids,
            generators_cfg=self.config["candidates"]["generators"],
            per_generator_k=int(self.config["candidates"]["per_generator_k"]),
            seed=int(self.config["pipeline"]["seed"]),
            tqdm_enabled=self.is_tqdm_enabled(),
        )

    def rank_predictions(self, dataset: Dataset, candidates: pd.DataFrame) -> pd.DataFrame:
        """Run configured ranker and return top-k predictions."""
        return rank_predictions(
            dataset=dataset,
            candidates=candidates,
            source_weights=self.config.get("ranking", {}).get("source_weights", {}),
            k=int(self.config["pipeline"]["k"]),
        )

    def validate_submission(self, submission: pd.DataFrame) -> None:
        """Validate submission against targets and pipeline k."""
        validate_submission(
            submission=submission,
            data_dir=self.paths.data_dir,
            k=int(self.config["pipeline"]["k"]),
        )

    def write_dataframe(self, frame: pd.DataFrame, output_path: Path) -> None:
        """Persist dataframe atomically to the target path."""
        atomic_write_dataframe(frame, output_path)

