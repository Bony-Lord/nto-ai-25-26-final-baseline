"""Schema utilities for strict dataframe checks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    """Raise ValueError if required columns are missing."""
    required = list(columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(sorted(missing))}"
        )

