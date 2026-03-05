"""Ranker contract."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.core.dataset import Dataset


class Ranker(Protocol):
    """Common ranker interface."""

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Return dataframe with user_id, edition_id, rank."""

