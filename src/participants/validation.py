"""Participant-facing validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.validate import validate_submission_df


def validate_submission(submission: pd.DataFrame, data_dir: Path, k: int) -> None:
    """Validate submission against targets and pipeline k."""
    target_users = set(pd.read_csv(data_dir / "targets.csv")["user_id"].astype("int64").tolist())
    validate_submission_df(
        submission_df=submission,
        target_users=target_users,
        k=int(k),
    )
