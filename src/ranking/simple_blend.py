"""Simple max-based blending ranker."""

from __future__ import annotations

import pandas as pd

from src.core.dataset import Dataset


class SimpleBlendRanker:
    """Blend candidate sources with weighted max score and select top-k."""

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        self.source_weights = source_weights or {}

    def _apply_weights(self, candidates: pd.DataFrame) -> pd.DataFrame:
        weighted = candidates.copy()
        weighted["weight"] = weighted["source"].map(self.source_weights).fillna(1.0)
        weighted["final_score"] = weighted["score"] * weighted["weight"]
        return weighted

    def rank(self, dataset: Dataset, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """Apply filtering, deduplication, blending, and fallback to popularity."""
        if candidates.empty:
            return self._fallback_only(dataset, k)

        seen = dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates()
        filtered = candidates.merge(
            seen.assign(_seen=1),
            on=["user_id", "edition_id"],
            how="left",
        )
        filtered = filtered[filtered["_seen"].isna()].drop(columns=["_seen"])
        if filtered.empty:
            return self._fallback_only(dataset, k)

        filtered = self._apply_weights(filtered)
        blended = (
            filtered.groupby(["user_id", "edition_id"], as_index=False)["final_score"]
            .max()
            .sort_values(["user_id", "final_score", "edition_id"], ascending=[True, False, True])
        )

        selected = blended.groupby("user_id", group_keys=False).head(k).copy()
        selected["rank"] = selected.groupby("user_id").cumcount() + 1
        selected = selected[["user_id", "edition_id", "rank", "final_score"]]

        completed = self._apply_fallback(selected, dataset, k)
        return completed.sort_values(["user_id", "rank"]).reset_index(drop=True)

    def _fallback_only(self, dataset: Dataset, k: int) -> pd.DataFrame:
        rows: list[dict[str, int | float]] = []
        positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        ranked_editions = popularity["edition_id"].tolist()
        seen_pairs = set(
            tuple(x)
            for x in dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates().to_numpy()
        )
        for user_id in dataset.targets_df["user_id"].tolist():
            rank = 1
            for edition_id in ranked_editions:
                if (int(user_id), int(edition_id)) in seen_pairs:
                    continue
                rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(edition_id),
                        "rank": rank,
                        "final_score": 0.0,
                    }
                )
                rank += 1
                if rank > k:
                    break
        return pd.DataFrame(rows)

    def _apply_fallback(
        self,
        selected: pd.DataFrame,
        dataset: Dataset,
        k: int,
    ) -> pd.DataFrame:
        positives = dataset.interactions_df[dataset.interactions_df["event_type"].isin([1, 2])]
        popularity = (
            positives.groupby("edition_id", as_index=False)["user_id"]
            .nunique()
            .rename(columns={"user_id": "pop"})
            .sort_values(["pop", "edition_id"], ascending=[False, True])
        )
        popular_editions = popularity["edition_id"].tolist()
        seen_pairs = set(
            tuple(x)
            for x in dataset.seen_positive_df[["user_id", "edition_id"]].drop_duplicates().to_numpy()
        )
        chosen_pairs = set(tuple(x) for x in selected[["user_id", "edition_id"]].to_numpy())
        missing_rows: list[dict[str, int | float]] = []
        by_user_counts = selected.groupby("user_id").size().to_dict()

        for user_id in dataset.targets_df["user_id"].tolist():
            count = int(by_user_counts.get(int(user_id), 0))
            rank = count + 1
            if count >= k:
                continue
            for edition_id in popular_editions:
                pair = (int(user_id), int(edition_id))
                if pair in chosen_pairs or pair in seen_pairs:
                    continue
                missing_rows.append(
                    {
                        "user_id": int(user_id),
                        "edition_id": int(edition_id),
                        "rank": rank,
                        "final_score": 0.0,
                    }
                )
                chosen_pairs.add(pair)
                rank += 1
                if rank > k:
                    break
        if missing_rows:
            selected = pd.concat([selected, pd.DataFrame(missing_rows)], ignore_index=True)
        return selected

