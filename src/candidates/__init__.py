"""Candidate generators registry."""

from __future__ import annotations

from collections.abc import Callable

from src.candidates.global_popularity import GlobalPopularityGenerator
from src.candidates.user_author import UserAuthorGenerator
from src.candidates.user_genre_popularity import UserGenrePopularityGenerator

GeneratorFactory = Callable[[dict[str, float], bool], object]


def _build_global_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    return GlobalPopularityGenerator(show_progress=tqdm_enabled)


def _build_user_genre_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserGenrePopularityGenerator(
        genre_smoothing=float(params.get("genre_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )


def _build_user_author(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserAuthorGenerator(
        author_smoothing=float(params.get("author_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )


GENERATOR_REGISTRY: dict[str, GeneratorFactory] = {
    "global_popularity": _build_global_popularity,
    "user_genre_popularity": _build_user_genre_popularity,
    "user_author": _build_user_author,
}


def build_generator(name: str, params: dict[str, float], tqdm_enabled: bool = False) -> object:
    """Instantiate generator by config name."""
    try:
        factory = GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(f"Unknown generator name: {name}. Available: {available}") from exc
    return factory(params, tqdm_enabled)


__all__ = ["GENERATOR_REGISTRY", "build_generator"]

