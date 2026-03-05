"""Submission stage implementation."""

from __future__ import annotations

import pandas as pd
from typing import Any

from src.core.artifacts import atomic_write_dataframe
from src.participants.validation import validate_submission
from src.pipeline.models import PipelineContext


class MakeSubmissionStage:
    """Build final submission file and validate format."""

    name = "make_submission"

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def run(self) -> dict[str, Any]:
        predictions = pd.read_parquet(self.context.paths.predictions_path).copy()
        submission = predictions[["user_id", "edition_id", "rank"]].sort_values(
            ["user_id", "rank", "edition_id"]
        )
        submission["user_id"] = submission["user_id"].astype("int64")
        submission["edition_id"] = submission["edition_id"].astype("int64")
        submission["rank"] = submission["rank"].astype("int32")
        validate_submission(
            submission=submission,
            data_dir=self.context.paths.data_dir,
            k=int(self.context.config["pipeline"]["k"]),
        )
        atomic_write_dataframe(submission, self.context.paths.submission_path)
        return {
            "rows": int(len(submission)),
            "users": int(submission["user_id"].nunique() if not submission.empty else 0),
        }

