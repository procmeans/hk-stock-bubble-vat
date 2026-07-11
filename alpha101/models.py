"""Model registry and common score table helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def run_scores(
    panel: dict,
    model_names: list[str],
    names: dict | None = None,
    output: Path | None = None,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run multiple registered models and write one long-form score table."""
    frames = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model: {model_name}")
        score = MODEL_REGISTRY[model_name](panel, mask=mask)
        frames.append(scores_to_long_frame(score, model_name, names=names))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def _load_names(universe_path: Path) -> dict:
    from alpha101.ths_today import load_code_pool

    return load_code_pool(universe_path).set_index("code")["name"].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--cache", type=Path, required=True)
    score_parser.add_argument("--universe", type=Path, required=True)
    score_parser.add_argument(
        "--models",
        default="alpha101_equal_weight,alpha101_single_101",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/model_scores/latest_scores.csv"),
    )
    args = parser.parse_args()

    if args.cmd == "score":
        from alpha101.ths_history import load_panel

        model_names = [item.strip() for item in args.models.split(",") if item.strip()]
        panel = load_panel(args.cache, args.universe)
        result = run_scores(
            panel,
            model_names,
            names=_load_names(args.universe),
            output=args.output,
        )
        print(f"wrote {len(result)} rows to {args.output}")
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
