"""Progress and ETA helpers for pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field


def format_seconds(seconds: float) -> str:
    """Format duration in seconds to human-readable mm:ss or hh:mm:ss."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class StageProgressTracker:
    """Track completed stage durations and estimate remaining runtime."""

    total_stages: int
    historical_durations: dict[str, float] = field(default_factory=dict)
    completed_durations: list[float] = field(default_factory=list)

    def estimate_remaining_seconds(
        self, current_index: int, remaining_stage_names: list[str]
    ) -> float:
        """Estimate seconds for stages after current index."""
        if not remaining_stage_names:
            return 0.0

        if self.completed_durations:
            avg = sum(self.completed_durations) / len(self.completed_durations)
            return avg * len(remaining_stage_names)

        known = [
            self.historical_durations[name]
            for name in remaining_stage_names
            if name in self.historical_durations
        ]
        if known:
            return float(sum(known))

        # No signal yet: unknown ETA.
        return 0.0

    def register_completed_stage(self, duration_sec: float) -> None:
        """Store measured duration of one finished stage."""
        self.completed_durations.append(max(0.0, float(duration_sec)))

