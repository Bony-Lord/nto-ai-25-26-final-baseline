"""CLI for running baseline pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.core.logging import configure_logging
from src.pipeline import STAGES, PipelineRunner


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML config from disk with recursive imports support."""
    resolved_path = config_path.resolve()
    visited = visited or set()
    if resolved_path in visited:
        raise ValueError(f"Cyclic config import detected: {resolved_path}")
    visited.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    imports = config.pop("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        raise ValueError("Config key 'imports' must be a list")

    merged: dict[str, Any] = {}
    for import_path in imports:
        if not isinstance(import_path, str):
            raise ValueError("Each import path must be a string")
        child_path = (resolved_path.parent / import_path).resolve()
        child_cfg = load_config(child_path, visited=visited)
        merged = _deep_merge(merged, child_cfg)
    merged = _deep_merge(merged, config)
    return merged


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run whole pipeline or selected stage")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config",
    )
    run_parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default=None,
        help="Run target stage and dependencies only",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Run local pseudo-incident validation"
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        logs_dir = Path(config.get("logs", {}).get("dir", "./logs")).resolve()
        logger, log_path = configure_logging(logs_dir)
        logger.info("Log file: %s", log_path)
        runner = PipelineRunner(config=config, logger=logger)

        if args.command == "run":
            runner.run(stage=args.stage)
        elif args.command == "validate":
            result = runner.run_local_validation()
            print(json.dumps(result, ensure_ascii=False))
        else:
            raise RuntimeError(f"Unsupported command {args.command}")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()

