"""Generator execution and contract checks for participant surface."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.core.dataset import Dataset
from src.participants.generators.registry import GENERATOR_REGISTRY


def build_generator(name: str, params: dict[str, float], tqdm_enabled: bool = False) -> object:
    """Instantiate a generator from registry by name."""
    try:
        factory = GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(f"Unknown generator name: {name}. Available: {available}") from exc
    return factory(params, tqdm_enabled)


def validate_candidate_contract(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Validate candidate generator output schema and types."""
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


def run_generators(
    dataset: Dataset,
    features: pd.DataFrame,
    user_ids: pd.Series,
    generators_cfg: list[dict[str, Any]],
    per_generator_k: int,
    seed: int,
    tqdm_enabled: bool,
) -> pd.DataFrame:
    """Execute configured generators and return concatenated candidates."""
    frames: list[pd.DataFrame] = []
    for generator_cfg in tqdm(
        generators_cfg,
        total=len(generators_cfg),
        desc="generate_candidates",
        leave=False,
        dynamic_ncols=True,
        disable=not tqdm_enabled,
        file=sys.stdout,
    ):
        generator = build_generator(
            generator_cfg["name"],
            generator_cfg.get("params", {}),
            tqdm_enabled=tqdm_enabled,
        )
        generated = generator.generate(
            dataset=dataset,
            user_ids=user_ids.astype("int64").to_numpy(),
            features=features,
            k=int(per_generator_k),
            seed=int(seed),
        )
        frames.append(validate_candidate_contract(generated, generator.name))

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["user_id", "edition_id", "score", "source"])
    )
