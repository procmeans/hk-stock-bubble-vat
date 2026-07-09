"""Model registry and common score table helpers."""

from __future__ import annotations

import pandas as pd

from alpha101 import alphas, compose


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def score_alpha101_equal_weight(panel: dict, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score with the existing full Alpha101 equal-weight composite."""
    factors = alphas.compute_all(panel)
    return compose.composite(factors, mask=mask)


def score_alpha101_single_101(panel: dict, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score with WQ Alpha101 factor #101 only."""
    score = alphas.alpha_101(panel)
    if mask is not None:
        score = score.where(mask.reindex_like(score).fillna(False))
    return score


MODEL_REGISTRY = {
    "alpha101_equal_weight": score_alpha101_equal_weight,
    "alpha101_single_101": score_alpha101_single_101,
}


def scores_to_long_frame(
    score: pd.DataFrame,
    model: str,
    names: dict | None = None,
    eligible: bool = True,
    reason: str = "",
) -> pd.DataFrame:
    """Convert a date x code score matrix to a stable long-form score table."""
    rows = []
    names = names or {}
    for date, row in score.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        for rank, (code, value) in enumerate(ranked.items(), start=1):
            rows.append({
                "date": pd.Timestamp(date).date().isoformat(),
                "code": str(code).zfill(6),
                "name": names.get(str(code).zfill(6), ""),
                "model": model,
                "score": value,
                "rank": rank,
                "eligible": bool(eligible),
                "reason": reason,
            })
    return pd.DataFrame(
        rows,
        columns=["date", "code", "name", "model", "score", "rank", "eligible", "reason"],
    )
