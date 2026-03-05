"""Pipeline orchestrator with stage registry and cache routing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.core.artifacts import ArtifactsManager
from src.core.progress import StageProgressTracker, format_seconds
from src.io.hashing import compute_inputs_fingerprint
from src.pipeline.models import (
    DEPENDENCIES,
    STAGES,
    STAGE_SCHEMA_VERSIONS,
    PipelineContext,
    PipelinePaths,
)
from src.pipeline.stage_helpers import StageHelpers
from src.pipeline.stages import (
    BuildFeaturesStage,
    GenerateCandidatesStage,
    MakeSubmissionStage,
    PrepareDataStage,
    RankAndSelectStage,
)
from src.pipeline.stages.base import PipelineStage
from src.pipeline.workflows.local_validation import PseudoIncidentValidationWorkflow
from src.utils.time import utc_now_iso


class PipelineRunner:
    """Run pipeline stages with fingerprint-based skipping."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.paths = self._resolve_paths(config)
        self.artifacts = ArtifactsManager(self.paths.artifacts_dir)
        self.context = PipelineContext(
            config=self.config,
            paths=self.paths,
            logger=self.logger,
            artifacts=self.artifacts,
        )
        self.helpers = StageHelpers(self.context)
        self.validation = PseudoIncidentValidationWorkflow(self.context, self.helpers)
        self.stage_registry: dict[str, PipelineStage] = {
            "prepare_data": PrepareDataStage(self.helpers),
            "build_features": BuildFeaturesStage(self.helpers),
            "generate_candidates": GenerateCandidatesStage(self.helpers),
            "rank_and_select": RankAndSelectStage(self.helpers),
            "make_submission": MakeSubmissionStage(self.helpers),
        }

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

    def run_local_validation(self) -> dict[str, Any]:
        """Run local pseudo-incident validation workflow."""
        return self.validation.run()

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
        stats = self.stage_registry[stage_name].run()
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
            self.paths.data_dir / "book_genres.csv",
            self.paths.data_dir / "genres.csv",
            self.paths.data_dir / "users.csv",
        ]
        metadata: list[dict[str, Any]] = []
        for path in files:
            if path.exists():
                stat = path.stat()
                metadata.append(
                    {
                        "path": str(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return metadata

