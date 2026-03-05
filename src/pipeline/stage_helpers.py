"""Shared helper logic for pipeline stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.candidates import build_generator
from src.core.artifacts import atomic_write_dataframe
from src.core.dataset import Dataset
from src.core.validate import validate_submission_df
from src.pipeline.models import PipelineContext, PipelinePaths
from src.ranking.simple_blend import SimpleBlendRanker


class StageHelpers:
    """Reusable operations shared across multiple stages."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    @property
    def config(self) -> dict[str, Any]:
        return self.context.config

    @property
    def paths(self) -> PipelinePaths:
        return self.context.paths

    def is_tqdm_enabled(self) -> bool:
        """Return whether tqdm progress bars should be enabled."""
        logs_cfg = self.config.get("logs", {})
        enabled_cfg = bool(logs_cfg.get("tqdm_enabled", True))
        return enabled_cfg and sys.stdout.isatty()

    def pack_data_cache(self, dataset: Dataset) -> pd.DataFrame:
        """Pack interactions and seen positives into one cache file."""
        interactions = dataset.interactions_df.copy()
        interactions["_record_type"] = "interaction"
        seen = dataset.seen_positive_df.copy()
        seen["_record_type"] = "seen_positive"
        seen["event_type"] = pd.NA
        seen["rating"] = pd.NA
        seen["event_ts"] = pd.NaT
        combined = pd.concat([interactions, seen], ignore_index=True, sort=False)
        return combined[
            ["_record_type", "user_id", "edition_id", "event_type", "rating", "event_ts"]
        ]

    def read_data_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read packed data cache and split into interactions/seen positives."""
        cache = pd.read_parquet(self.paths.data_cache_path)
        interactions = cache[cache["_record_type"] == "interaction"][
            ["user_id", "edition_id", "event_type", "rating", "event_ts"]
        ].copy()
        interactions["user_id"] = interactions["user_id"].astype("int64")
        interactions["edition_id"] = interactions["edition_id"].astype("int64")
        interactions["event_type"] = interactions["event_type"].astype("int32")
        interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
        seen_positive = cache[cache["_record_type"] == "seen_positive"][
            ["user_id", "edition_id"]
        ].copy()
        seen_positive["user_id"] = seen_positive["user_id"].astype("int64")
        seen_positive["edition_id"] = seen_positive["edition_id"].astype("int64")
        return interactions, seen_positive

    def build_features_frame(self, dataset: Dataset) -> pd.DataFrame:
        """Build all default feature frames into one long table."""
        recent_days = int(self.config["pipeline"]["recent_days"])
        positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]

        popularity_all = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "value"})
        )
        popularity_all["feature_type"] = "edition_popularity_all"
        popularity_all["user_id"] = pd.NA
        popularity_all["genre_id"] = pd.NA
        popularity_all["author_id"] = pd.NA

        max_ts = positives["event_ts"].max()
        cutoff = max_ts - pd.Timedelta(days=recent_days)
        recent = positives[positives["event_ts"] >= cutoff]
        popularity_recent = (
            recent.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "value"})
        )
        popularity_recent["feature_type"] = "edition_popularity_recent"
        popularity_recent["user_id"] = pd.NA
        popularity_recent["genre_id"] = pd.NA
        popularity_recent["author_id"] = pd.NA

        user_genres = positives[["user_id", "edition_id"]].merge(
            dataset.catalog_df[["edition_id", "book_id"]],
            on="edition_id",
            how="inner",
        )
        user_genres = user_genres.merge(dataset.book_genres_df, on="book_id", how="inner")
        user_genre_profile = (
            user_genres.groupby(["user_id", "genre_id"], as_index=False)["edition_id"]
            .count()
            .rename(columns={"edition_id": "value"})
        )
        user_genre_profile["value"] = user_genre_profile["value"] / user_genre_profile.groupby(
            "user_id"
        )["value"].transform("sum")
        user_genre_profile["feature_type"] = "user_genre_profile"
        user_genre_profile["edition_id"] = pd.NA
        user_genre_profile["author_id"] = pd.NA

        user_authors = positives[["user_id", "edition_id"]].merge(
            dataset.catalog_df[["edition_id", "author_id"]],
            on="edition_id",
            how="inner",
        )
        user_author_profile = (
            user_authors.groupby(["user_id", "author_id"], as_index=False)["edition_id"]
            .count()
            .rename(columns={"edition_id": "value"})
        )
        user_author_profile["value"] = user_author_profile["value"] / user_author_profile.groupby(
            "user_id"
        )["value"].transform("sum")
        user_author_profile["feature_type"] = "user_author_profile"
        user_author_profile["edition_id"] = pd.NA
        user_author_profile["genre_id"] = pd.NA

        return pd.concat(
            [
                popularity_all[
                    ["feature_type", "user_id", "edition_id", "genre_id", "author_id", "value"]
                ],
                popularity_recent[
                    ["feature_type", "user_id", "edition_id", "genre_id", "author_id", "value"]
                ],
                user_genre_profile[
                    ["feature_type", "user_id", "edition_id", "genre_id", "author_id", "value"]
                ],
                user_author_profile[
                    ["feature_type", "user_id", "edition_id", "genre_id", "author_id", "value"]
                ],
            ],
            ignore_index=True,
        )

    def validate_candidate_contract(
        self, frame: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Validate candidate generator output schema and types."""
        required = {"user_id", "edition_id", "score", "source"}
        if not required.issubset(frame.columns):
            missing = required - set(frame.columns)
            raise ValueError(
                f"generator '{source_name}' returned invalid schema, missing={sorted(missing)}"
            )
        frame = frame[["user_id", "edition_id", "score", "source"]].copy()
        frame["user_id"] = frame["user_id"].astype("int64")
        frame["edition_id"] = frame["edition_id"].astype("int64")
        frame["score"] = frame["score"].astype(float)
        frame["source"] = frame["source"].astype(str)
        if (frame["source"] != source_name).any():
            raise ValueError(f"generator '{source_name}' returned rows with invalid source")
        return frame

    def run_generators(
        self, dataset: Dataset, features: pd.DataFrame, user_ids: pd.Series
    ) -> pd.DataFrame:
        """Execute configured generators and return concatenated candidates."""
        per_generator_k = int(self.config["candidates"]["per_generator_k"])
        seed = int(self.config["pipeline"]["seed"])
        frames: list[pd.DataFrame] = []
        generators_cfg = self.config["candidates"]["generators"]
        for generator_cfg in tqdm(
            generators_cfg,
            total=len(generators_cfg),
            desc="generate_candidates",
            leave=False,
            dynamic_ncols=True,
            disable=not self.is_tqdm_enabled(),
            file=sys.stdout,
        ):
            generator = build_generator(
                generator_cfg["name"],
                generator_cfg.get("params", {}),
                tqdm_enabled=self.is_tqdm_enabled(),
            )
            generated = generator.generate(
                dataset=dataset,
                user_ids=user_ids.astype("int64").to_numpy(),
                features=features,
                k=per_generator_k,
                seed=seed,
            )
            frames.append(self.validate_candidate_contract(generated, generator.name))

        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])
        )

    def rank_predictions(self, dataset: Dataset, candidates: pd.DataFrame) -> pd.DataFrame:
        """Run configured ranker and return top-k predictions."""
        ranker = SimpleBlendRanker(
            source_weights={
                key: float(value)
                for key, value in self.config.get("ranking", {}).get("source_weights", {}).items()
            }
        )
        return ranker.rank(
            dataset=dataset,
            candidates=candidates,
            k=int(self.config["pipeline"]["k"]),
        )

    def validate_submission(self, submission: pd.DataFrame) -> None:
        """Validate submission against targets and pipeline k."""
        target_users = set(
            pd.read_csv(self.paths.data_dir / "targets.csv")["user_id"].astype("int64").tolist()
        )
        validate_submission_df(
            submission_df=submission,
            target_users=target_users,
            k=int(self.config["pipeline"]["k"]),
        )

    def write_dataframe(self, frame: pd.DataFrame, output_path: Path) -> None:
        """Persist dataframe atomically to the target path."""
        atomic_write_dataframe(frame, output_path)

