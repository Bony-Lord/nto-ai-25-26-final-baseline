"""Candidate generators registry."""

from __future__ import annotations

from src.candidates.global_popularity import GlobalPopularityGenerator
from src.candidates.user_author import UserAuthorGenerator
from src.candidates.user_genre_popularity import UserGenrePopularityGenerator


def build_generator(name: str, params: dict[str, float], tqdm_enabled: bool = False):
    """Instantiate generator by config name."""
    if name == "global_popularity":
        return GlobalPopularityGenerator(show_progress=tqdm_enabled)
    if name == "user_genre_popularity":
        return UserGenrePopularityGenerator(
            genre_smoothing=float(params.get("genre_smoothing", 1.0)),
            show_progress=tqdm_enabled,
        )
    if name == "user_author":
        return UserAuthorGenerator(
            author_smoothing=float(params.get("author_smoothing", 1.0)),
            show_progress=tqdm_enabled,
        )
    raise ValueError(f"Unknown generator name: {name}")

