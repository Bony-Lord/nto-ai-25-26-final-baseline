"""Submission stage implementation."""

from __future__ import annotations

import pandas as pd

from src.pipeline.stage_helpers import StageHelpers


class MakeSubmissionStage:
    """Build final submission file and validate format."""

    name = "make_submission"

    def __init__(self, helpers: StageHelpers) -> None:
        self.helpers = helpers

    def run(self) -> dict[str, Any]:
        predictions = pd.read_parquet(self.helpers.paths.predictions_path).copy()
        submission = predictions[["user_id", "edition_id", "rank"]].sort_values(
            ["user_id", "rank", "edition_id"]
        )
        submission["user_id"] = submission["user_id"].astype("int64")
        submission["edition_id"] = submission["edition_id"].astype("int64")
        submission["rank"] = submission["rank"].astype("int32")
        self.helpers.validate_submission(submission)
        self.helpers.write_dataframe(submission, self.helpers.paths.submission_path)
        return {
            "rows": int(len(submission)),
            "users": int(submission["user_id"].nunique() if not submission.empty else 0),
        }

