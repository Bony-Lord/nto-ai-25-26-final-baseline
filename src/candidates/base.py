"""Candidate generator contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.core.dataset import Dataset


class CandidateGenerator(Protocol):
    """Common interface for all candidate generators."""

    name: str

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        """Generate candidates with columns: user_id, edition_id, score, source."""


@dataclass(frozen=True)
class GeneratorConfig:
    """Generator configuration loaded from YAML."""

    name: str
    params: dict[str, float]

