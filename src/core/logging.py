"""Logging setup for CLI and pipeline runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.time import utc_now_iso


class TqdmCompatibleStreamHandler(logging.StreamHandler):
    """Console handler compatible with active tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover
            super().emit(record)


def configure_logging(logs_dir: Path) -> tuple[logging.Logger, Path]:
    """Configure stdout + file logger for current run."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_now_iso().replace(":", "-")
    log_path = logs_dir / f"run_{ts}.log"

    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = TqdmCompatibleStreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logging initialized")
    return logger, log_path

