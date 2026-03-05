"""Participant-facing ranking entrypoints."""

from __future__ import annotations

import pandas as pd

from src.core.dataset import Dataset
from src.ranking.simple_blend import SimpleBlendRanker


def rank_predictions(
    dataset: Dataset,
    candidates: pd.DataFrame,
    source_weights: dict[str, float],
    k: int,
) -> pd.DataFrame:
    """Run configured ranker and return top-k predictions."""
    ranker = SimpleBlendRanker(
        source_weights={key: float(value) for key, value in source_weights.items()}
    )
    return ranker.rank(
        dataset=dataset,
        candidates=candidates,
        k=int(k),
    )
