"""Metric helpers for NDCG@20 evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricsSummary:
    """NDCG summary statistics."""

    mean_ndcg: float
    quantiles: dict[str, float]
    per_user: pd.DataFrame


def ndcg_at_k(predicted: list[int], relevant: set[int], k: int) -> float:
    """Compute binary NDCG@k for one user."""
    dcg = 0.0
    for rank, edition_id in enumerate(predicted[:k], start=1):
        rel = 1.0 if edition_id in relevant else 0.0
        dcg += rel / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_ndcg(per_user_df: pd.DataFrame) -> MetricsSummary:
    """Summarize per-user NDCG scores with quantiles."""
    if per_user_df.empty:
        return MetricsSummary(mean_ndcg=0.0, quantiles={"q25": 0.0, "q50": 0.0, "q75": 0.0}, per_user=per_user_df)
    scores = per_user_df["ndcg@20"]
    quantiles = {
        "q25": float(scores.quantile(0.25)),
        "q50": float(scores.quantile(0.50)),
        "q75": float(scores.quantile(0.75)),
    }
    return MetricsSummary(
        mean_ndcg=float(scores.mean()),
        quantiles=quantiles,
        per_user=per_user_df,
    )

