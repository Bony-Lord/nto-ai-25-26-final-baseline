"""Low-level IO loaders for CSV and parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with pandas using default parsing."""
    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file with pandas."""
    return pd.read_parquet(path)

