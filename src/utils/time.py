"""Time-related helpers for pipeline steps."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()

