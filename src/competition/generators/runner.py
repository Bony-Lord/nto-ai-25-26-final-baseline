"""Generator runner and contract checks for participant solution."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.competition.generators.registry import build_generator
from src.platform.core.dataset import Dataset


def validate_candidate_contract(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize and validate generator output schema.

    Args:
        frame: Raw DataFrame emitted by a generator.
        source_name: Expected source value for all rows.

    Returns:
        Normalized DataFrame with stable column order and dtypes.

    Raises:
        ValueError: If required columns are missing or source labels mismatch.
    """
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
    """Execute configured generators and aggregate all candidate rows.

    Args:
        dataset: Runtime dataset passed to each generator.
        features: Feature matrix computed for the current run.
        user_ids: Distinct target users for candidate generation.
        generators_cfg: Ordered generator config list from YAML.
        per_generator_k: Max rows per user for each generator.
        seed: Global deterministic seed for generators.
        tqdm_enabled: Whether runner should display progress bars.

    Returns:
        Concatenated candidate DataFrame across all configured generators.
    """
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

