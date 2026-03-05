"""Pipeline orchestration for ADR-001 baseline."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.candidates import build_generator
from src.core.artifacts import ArtifactsManager, atomic_write_dataframe
from src.core.dataset import Dataset
from src.core.metrics import ndcg_at_k, summarize_ndcg
from src.core.progress import StageProgressTracker, format_seconds
from src.core.validate import validate_submission_df
from src.io.hashing import compute_inputs_fingerprint
from src.ranking.simple_blend import SimpleBlendRanker
from src.utils.time import utc_now_iso

STAGES = [
    "prepare_data",
    "build_features",
    "generate_candidates",
    "rank_and_select",
    "make_submission",
]

DEPENDENCIES: dict[str, list[str]] = {
    "prepare_data": [],
    "build_features": ["prepare_data"],
    "generate_candidates": ["build_features"],
    "rank_and_select": ["generate_candidates"],
    "make_submission": ["rank_and_select"],
}

STAGE_SCHEMA_VERSIONS: dict[str, int] = {
    "prepare_data": 2,
    "build_features": 1,
    "generate_candidates": 2,
    "rank_and_select": 2,
    "make_submission": 2,
}


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved project paths."""

    data_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    data_cache_path: Path
    features_path: Path
    candidates_path: Path
    predictions_path: Path
    submission_path: Path


class PipelineRunner:
    """Run pipeline stages with fingerprint-based skipping."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.paths = self._resolve_paths(config)
        self.artifacts = ArtifactsManager(self.paths.artifacts_dir)

    @staticmethod
    def _resolve_paths(config: dict[str, Any]) -> PipelinePaths:
        paths_cfg = config["paths"]
        artifacts_dir = Path(paths_cfg["artifacts_dir"]).resolve()
        data_dir = Path(paths_cfg["data_dir"]).resolve()
        logs_dir = Path(config.get("logs", {}).get("dir", "./logs")).resolve()
        return PipelinePaths(
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            data_cache_path=artifacts_dir / "data_cache.parquet",
            features_path=artifacts_dir / "features.parquet",
            candidates_path=artifacts_dir / "candidates.parquet",
            predictions_path=artifacts_dir / "predictions.parquet",
            submission_path=artifacts_dir / "submission.csv",
        )

    def run(self, stage: str | None = None) -> None:
        """Run all stages or up to selected stage with dependencies."""
        if stage is not None and stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Allowed: {', '.join(STAGES)}")

        run_meta = {
            "started_at": utc_now_iso(),
            "config": self.config,
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "artifacts_dir": str(self.paths.artifacts_dir),
            },
            "inputs": self._collect_input_metadata(),
        }
        self.artifacts.write_run_meta(run_meta)

        stages_to_run = self._resolve_stage_chain(stage)
        total_stages = len(stages_to_run)
        historical_durations = self.artifacts.get_step_durations(stages_to_run)
        tracker = StageProgressTracker(
            total_stages=total_stages,
            historical_durations=historical_durations,
        )
        run_started_at = time.perf_counter()
        self.logger.info("Pipeline start: %s stages", total_stages)
        for stage_index, stage_name in enumerate(stages_to_run, start=1):
            remaining_after = stages_to_run[stage_index:]
            eta_before = tracker.estimate_remaining_seconds(stage_index, remaining_after)
            self.logger.info(
                "Stage %s/%s %s started (remaining_stages=%s, eta~%s)",
                stage_index,
                total_stages,
                stage_name,
                len(remaining_after),
                format_seconds(eta_before),
            )
            stage_duration, was_skipped = self._run_stage(
                stage_name=stage_name,
                stage_index=stage_index,
                stage_total=total_stages,
            )
            if not was_skipped:
                tracker.register_completed_stage(stage_duration)
                eta_after = tracker.estimate_remaining_seconds(stage_index, remaining_after)
                self.logger.info(
                    "Stage %s/%s %s done in %.2fs, remaining=%s, remaining~%s",
                    stage_index,
                    total_stages,
                    stage_name,
                    stage_duration,
                    len(remaining_after),
                    format_seconds(eta_after),
                )
        self.logger.info(
            "Pipeline done in %s", format_seconds(time.perf_counter() - run_started_at)
        )

    def _resolve_stage_chain(self, stage: str | None) -> list[str]:
        if stage is None:
            return STAGES
        chain: list[str] = []

        def collect(name: str) -> None:
            for dep in DEPENDENCIES[name]:
                collect(dep)
            if name not in chain:
                chain.append(name)

        collect(stage)
        return chain

    def _run_stage(
        self, stage_name: str, stage_index: int, stage_total: int
    ) -> tuple[float, bool]:
        inputs = self._stage_inputs(stage_name)
        fingerprint = compute_inputs_fingerprint(
            inputs=inputs,
            config_snapshot=self._stage_config_snapshot(stage_name),
        )
        output_path = self._stage_output(stage_name)
        if not self.artifacts.should_run(stage_name, fingerprint, output_path):
            self.logger.info(
                "Skip stage %s/%s %s (cache hit)",
                stage_index,
                stage_total,
                stage_name,
            )
            return 0.0, True

        stage_started_at = time.perf_counter()
        self.logger.info("Run stage=%s", stage_name)
        self.artifacts.mark_started(stage_name, fingerprint)
        if stage_name == "prepare_data":
            stats = self._step_prepare_data()
        elif stage_name == "build_features":
            stats = self._step_build_features()
        elif stage_name == "generate_candidates":
            stats = self._step_generate_candidates()
        elif stage_name == "rank_and_select":
            stats = self._step_rank_and_select()
        elif stage_name == "make_submission":
            stats = self._step_make_submission()
        else:
            raise RuntimeError(f"Unsupported stage: {stage_name}")
        duration_sec = time.perf_counter() - stage_started_at
        self.artifacts.mark_done(
            step_name=stage_name,
            fingerprint=fingerprint,
            payload=stats,
            duration_sec=duration_sec,
        )
        self.logger.info("Done stage=%s stats=%s", stage_name, stats)
        return duration_sec, False

    def _stage_output(self, stage_name: str) -> Path:
        mapping = {
            "prepare_data": self.paths.data_cache_path,
            "build_features": self.paths.features_path,
            "generate_candidates": self.paths.candidates_path,
            "rank_and_select": self.paths.predictions_path,
            "make_submission": self.paths.submission_path,
        }
        return mapping[stage_name]

    def _stage_inputs(self, stage_name: str) -> list[Path]:
        data_files = [
            self.paths.data_dir / "interactions.csv",
            self.paths.data_dir / "targets.csv",
            self.paths.data_dir / "editions.csv",
            self.paths.data_dir / "book_genres.csv",
            self.paths.data_dir / "genres.csv",
            self.paths.data_dir / "users.csv",
        ]
        if stage_name == "prepare_data":
            return data_files
        if stage_name == "build_features":
            return [
                self.paths.data_cache_path,
                self.paths.data_dir / "editions.csv",
                self.paths.data_dir / "book_genres.csv",
            ]
        if stage_name == "generate_candidates":
            return [self.paths.features_path, self.paths.data_dir / "targets.csv"]
        if stage_name == "rank_and_select":
            return [self.paths.candidates_path, self.paths.data_cache_path]
        if stage_name == "make_submission":
            return [self.paths.predictions_path, self.paths.data_dir / "targets.csv"]
        raise RuntimeError(f"Unknown stage: {stage_name}")

    def _stage_config_snapshot(self, stage_name: str) -> dict[str, Any]:
        pipeline_cfg = self.config.get("pipeline", {})
        candidates_cfg = self.config.get("candidates", {})
        ranking_cfg = self.config.get("ranking", {})
        if stage_name in {"prepare_data", "build_features"}:
            return {
                "pipeline": pipeline_cfg,
                "schema_version": STAGE_SCHEMA_VERSIONS[stage_name],
            }
        if stage_name == "generate_candidates":
            return {
                "pipeline": pipeline_cfg,
                "candidates": candidates_cfg,
                "schema_version": STAGE_SCHEMA_VERSIONS[stage_name],
            }
        if stage_name in {"rank_and_select", "make_submission"}:
            return {
                "pipeline": pipeline_cfg,
                "ranking": ranking_cfg,
                "schema_version": STAGE_SCHEMA_VERSIONS[stage_name],
            }
        return self.config

    def _collect_input_metadata(self) -> list[dict[str, Any]]:
        files = [
            self.paths.data_dir / "interactions.csv",
            self.paths.data_dir / "targets.csv",
            self.paths.data_dir / "editions.csv",
            self.paths.data_dir / "authors.csv",
            self.paths.data_dir / "genres.csv",
            self.paths.data_dir / "book_genres.csv",
            self.paths.data_dir / "users.csv",
        ]
        metadata: list[dict[str, Any]] = []
        for path in files:
            if not path.exists():
                metadata.append({"path": str(path), "missing": True})
                continue
            stat = path.stat()
            metadata.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return metadata

    def _is_tqdm_enabled(self) -> bool:
        logs_cfg = self.config.get("logs", {})
        enabled_cfg = bool(logs_cfg.get("tqdm_enabled", True))
        return enabled_cfg and sys.stdout.isatty()

    def _pack_data_cache(self, dataset: Dataset) -> pd.DataFrame:
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

    def _read_data_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def _step_prepare_data(self) -> dict[str, Any]:
        dataset = Dataset.load(self.paths.data_dir)
        data_cache = self._pack_data_cache(dataset)
        atomic_write_dataframe(data_cache, self.paths.data_cache_path)
        return {
            "rows": int(len(data_cache)),
            "users": int(dataset.interactions_df["user_id"].nunique()),
            "positives": int(len(dataset.seen_positive_df)),
        }

    def _build_features_frame(self, dataset: Dataset) -> pd.DataFrame:
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
        user_genre_profile["value"] = user_genre_profile["value"] / user_genre_profile.groupby("user_id")[
            "value"
        ].transform("sum")
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
        user_author_profile["value"] = user_author_profile["value"] / user_author_profile.groupby("user_id")[
            "value"
        ].transform("sum")
        user_author_profile["feature_type"] = "user_author_profile"
        user_author_profile["edition_id"] = pd.NA
        user_author_profile["genre_id"] = pd.NA

        frame = pd.concat(
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
        return frame

    def _step_build_features(self) -> dict[str, Any]:
        dataset = Dataset.load(self.paths.data_dir)
        interactions, seen_positive = self._read_data_cache()
        dataset = Dataset(
            interactions_df=interactions,
            targets_df=dataset.targets_df,
            catalog_df=dataset.catalog_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=seen_positive,
        )
        features = self._build_features_frame(dataset)
        atomic_write_dataframe(features, self.paths.features_path)
        return {
            "rows": int(len(features)),
            "users": int(dataset.targets_df["user_id"].nunique()),
        }

    def _validate_candidate_contract(self, frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
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

    def _step_generate_candidates(self) -> dict[str, Any]:
        dataset = Dataset.load(self.paths.data_dir)
        interactions, seen_positive = self._read_data_cache()
        dataset = Dataset(
            interactions_df=interactions,
            targets_df=dataset.targets_df,
            catalog_df=dataset.catalog_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=seen_positive,
        )
        features = pd.read_parquet(self.paths.features_path)
        user_ids = dataset.targets_df["user_id"].astype("int64").to_numpy()
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
            disable=not self._is_tqdm_enabled(),
            file=sys.stdout,
        ):
            generator = build_generator(
                generator_cfg["name"],
                generator_cfg.get("params", {}),
                tqdm_enabled=self._is_tqdm_enabled(),
            )
            generated = generator.generate(
                dataset=dataset,
                user_ids=user_ids,
                features=features,
                k=per_generator_k,
                seed=seed,
            )
            checked = self._validate_candidate_contract(generated, generator.name)
            frames.append(checked)

        candidates = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])
        )
        atomic_write_dataframe(candidates, self.paths.candidates_path)
        return {
            "rows": int(len(candidates)),
            "users": int(candidates["user_id"].nunique()) if not candidates.empty else 0,
        }

    def _step_rank_and_select(self) -> dict[str, Any]:
        dataset = Dataset.load(self.paths.data_dir)
        interactions, seen_positive = self._read_data_cache()
        dataset = Dataset(
            interactions_df=interactions,
            targets_df=dataset.targets_df,
            catalog_df=dataset.catalog_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=seen_positive,
        )
        candidates = pd.read_parquet(self.paths.candidates_path)
        ranker = SimpleBlendRanker(
            source_weights={
                key: float(value)
                for key, value in self.config.get("ranking", {}).get("source_weights", {}).items()
            }
        )
        predictions = ranker.rank(dataset=dataset, candidates=candidates, k=int(self.config["pipeline"]["k"]))
        atomic_write_dataframe(predictions, self.paths.predictions_path)
        return {
            "rows": int(len(predictions)),
            "users": int(predictions["user_id"].nunique()) if not predictions.empty else 0,
        }

    def _step_make_submission(self) -> dict[str, Any]:
        predictions = pd.read_parquet(self.paths.predictions_path)
        submission = predictions[["user_id", "edition_id", "rank"]].copy()
        target_users = set(
            pd.read_csv(self.paths.data_dir / "targets.csv")["user_id"].astype("int64").tolist()
        )
        validate_submission_df(
            submission_df=submission,
            target_users=target_users,
            k=int(self.config["pipeline"]["k"]),
        )
        atomic_write_dataframe(submission, self.paths.submission_path)
        return {
            "rows": int(len(submission)),
            "users": int(submission["user_id"].nunique()),
        }

    def run_local_validation(self) -> dict[str, Any]:
        """Run pseudo-incident validation in clean period and report NDCG@20."""
        dataset = Dataset.load(self.paths.data_dir)
        interactions = dataset.interactions_df.copy()
        positives = interactions[interactions["event_type"].isin([1, 2])]
        if positives.empty:
            raise ValueError("No positive interactions available for local validation")

        min_ts = positives["event_ts"].min()
        clean_end = min_ts + pd.Timedelta(days=150)
        clean_df = positives[positives["event_ts"] < clean_end].copy()
        pseudo_days = int(self.config.get("validation", {}).get("pseudo_incident_days", 14))
        incident_end = clean_df["event_ts"].max()
        incident_start = incident_end - pd.Timedelta(days=pseudo_days)
        pseudo_window_df = clean_df[clean_df["event_ts"] >= incident_start].copy()
        pseudo_pairs = pseudo_window_df[["user_id", "edition_id"]].drop_duplicates()

        rng = np.random.default_rng(int(self.config["pipeline"]["seed"]))
        if pseudo_pairs.empty:
            raise ValueError("Pseudo-incident window does not contain positive pairs")
        mask_count = max(1, int(len(pseudo_pairs) * 0.2))
        mask_indexes = rng.choice(len(pseudo_pairs), size=mask_count, replace=False)
        masked_pairs = pseudo_pairs.iloc[mask_indexes].copy()
        masked_pairs["_masked"] = 1

        observed = clean_df.merge(masked_pairs, on=["user_id", "edition_id"], how="left")
        observed = observed[observed["_masked"].isna()].drop(columns=["_masked"])

        targets = (
            pseudo_window_df[["user_id"]]
            .drop_duplicates()
            .rename(columns={"user_id": "user_id"})
            .astype({"user_id": "int64"})
        )
        validation_dataset = Dataset(
            interactions_df=observed,
            targets_df=targets,
            catalog_df=dataset.catalog_df,
            book_genres_df=dataset.book_genres_df,
            genres_df=dataset.genres_df,
            users_df=dataset.users_df,
            seen_positive_df=observed[["user_id", "edition_id"]].drop_duplicates(),
        )

        features = self._build_features_frame(validation_dataset)
        user_ids = validation_dataset.targets_df["user_id"].to_numpy()
        per_generator_k = int(self.config["candidates"]["per_generator_k"])
        frames: list[pd.DataFrame] = []
        generators_cfg = self.config["candidates"]["generators"]
        for generator_cfg in tqdm(
            generators_cfg,
            total=len(generators_cfg),
            desc="validation_generators",
            leave=False,
            dynamic_ncols=True,
            disable=not self._is_tqdm_enabled(),
            file=sys.stdout,
        ):
            generator = build_generator(
                generator_cfg["name"],
                generator_cfg.get("params", {}),
                tqdm_enabled=self._is_tqdm_enabled(),
            )
            frame = generator.generate(
                dataset=validation_dataset,
                user_ids=user_ids,
                features=features,
                k=per_generator_k,
                seed=int(self.config["pipeline"]["seed"]),
            )
            frames.append(self._validate_candidate_contract(frame, generator.name))
        candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["user_id", "edition_id", "score", "source"]
        )

        ranker = SimpleBlendRanker(
            source_weights={
                key: float(value)
                for key, value in self.config.get("ranking", {}).get("source_weights", {}).items()
            }
        )
        predictions = ranker.rank(
            dataset=validation_dataset,
            candidates=candidates,
            k=int(self.config["pipeline"]["k"]),
        )

        relevant_by_user: dict[int, set[int]] = {}
        for row in masked_pairs.to_dict(orient="records"):
            relevant_by_user.setdefault(int(row["user_id"]), set()).add(int(row["edition_id"]))

        per_user_rows: list[dict[str, float | int]] = []
        target_user_ids = targets["user_id"].tolist()
        for user_id in tqdm(
            target_user_ids,
            total=len(target_user_ids),
            desc="validation_users",
            leave=False,
            dynamic_ncols=True,
            disable=not self._is_tqdm_enabled(),
            file=sys.stdout,
        ):
            user_pred = predictions[predictions["user_id"] == int(user_id)].sort_values("rank")
            predicted = user_pred["edition_id"].astype("int64").tolist()
            relevant = relevant_by_user.get(int(user_id), set())
            ndcg = ndcg_at_k(predicted=predicted, relevant=relevant, k=int(self.config["pipeline"]["k"]))
            per_user_rows.append({"user_id": int(user_id), "ndcg@20": ndcg})
        per_user_df = pd.DataFrame(per_user_rows)
        summary = summarize_ndcg(per_user_df)
        result = {
            "mean_ndcg@20": summary.mean_ndcg,
            "quantiles": summary.quantiles,
            "users": int(len(per_user_df)),
        }
        self.logger.info("Local validation result: %s", result)
        return result

