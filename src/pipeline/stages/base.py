"""Contracts for pipeline stage implementations."""

from __future__ import annotations

from typing import Any, Protocol


class PipelineStage(Protocol):
    """Common execution interface for all pipeline stages."""

    name: str

    def run(self) -> dict[str, Any]:
        """Execute stage logic and return execution stats."""

