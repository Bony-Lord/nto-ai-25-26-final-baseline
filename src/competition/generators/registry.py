"""Registry for participant-configurable generator factories."""

from __future__ import annotations

from collections.abc import Callable

from src.competition.generators.global_popularity import GlobalPopularityGenerator
from src.competition.generators.recent_popularity import RecentPopularityGenerator
from src.competition.generators.user_author import UserAuthorGenerator
from src.competition.generators.user_genre import UserGenrePopularityGenerator
from src.competition.generators.user_language_publisher import UserLanguagePublisherGenerator

GeneratorFactory = Callable[[dict[str, float], bool], object]


def _build_global_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    del params
    return GlobalPopularityGenerator(show_progress=tqdm_enabled)


def _build_user_genre(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserGenrePopularityGenerator(
        genre_smoothing=float(params.get("genre_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )




def _build_recent_popularity(params: dict[str, float], tqdm_enabled: bool) -> object:
    return RecentPopularityGenerator(
        decay_days=float(params.get("decay_days", 14.0)),
        show_progress=tqdm_enabled,
    )


def _build_user_language_publisher(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserLanguagePublisherGenerator(
        language_weight=float(params.get("language_weight", 1.0)),
        publisher_weight=float(params.get("publisher_weight", 0.8)),
        smoothing=float(params.get("smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )

def _build_user_author(params: dict[str, float], tqdm_enabled: bool) -> object:
    return UserAuthorGenerator(
        author_smoothing=float(params.get("author_smoothing", 1.0)),
        show_progress=tqdm_enabled,
    )


GENERATOR_REGISTRY: dict[str, GeneratorFactory] = {
    "global_popularity": _build_global_popularity,
    "user_genre": _build_user_genre,
    "user_author": _build_user_author,
    "recent_popularity": _build_recent_popularity,
    "user_language_publisher": _build_user_language_publisher,
}


def build_generator(name: str, params: dict[str, float], tqdm_enabled: bool = False) -> object:
    """Instantiate a configured generator factory by name.

    Args:
        name: Generator identifier from YAML config.
        params: Generator parameter mapping from YAML config.
        tqdm_enabled: Whether generator may display progress bars.

    Returns:
        Concrete generator instance implementing `.generate(...)`.

    Raises:
        ValueError: If no registered generator matches `name`.
    """
    try:
        factory = GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(f"Unknown generator name: {name}. Available: {available}") from exc
    return factory(params, tqdm_enabled)

