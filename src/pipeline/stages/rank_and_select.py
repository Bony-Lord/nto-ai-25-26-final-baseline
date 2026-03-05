"""Rank and select stage implementation."""

from __future__ import annotations

import pandas as pd
from typing import Any

from src.core.artifacts import atomic_write_dataframe
from src.participants.ranking import rank_predictions
from src.pipeline.models import PipelineContext
from src.pipeline.runtime import load_runtime_dataset


class RankAndSelectStage:
    """Rank candidates and persist final top-k predictions."""

    name = "rank_and_select"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        dataset = load_runtime_dataset(self.context.paths)
        candidates = pd.read_parquet(self.context.paths.candidates_path)
        predictions = rank_predictions(
            dataset=dataset,
            candidates=candidates,
            source_weights=self.context.config.get("ranking", {}).get("source_weights", {}),
            k=int(self.context.config["pipeline"]["k"]),
        )
        atomic_write_dataframe(predictions, self.context.paths.predictions_path)
        return {
            "rows": int(len(predictions)),
            "users": int(predictions["user_id"].nunique() if not predictions.empty else 0),
        }

