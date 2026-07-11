"""Convert model scores into target-position trade signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SIGNAL_COLUMNS = [
    "date",
    "model",
    "code",
    "name",
    "action",
    "current_weight",
    "target_weight",
    "score",
    "rank",
    "reason",
]


def generate_signals(
    scores: pd.DataFrame,
    current_positions: dict | None = None,
    buy_top_n: int = 20,
    sell_below_rank: int = 60,
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Generate buy/hold/sell signals from one date/model score table."""
    current_positions = current_positions or {}
    score_table = scores.copy()
    score_table = score_table[score_table.get("eligible", True).astype(bool)]
    score_table = score_table.sort_values(["rank", "score"], ascending=[True, False])

    selected_codes = set(
        score_table[score_table["rank"] <= buy_top_n]["code"].astype(str).tolist()
    )
    buffer_codes = set(
        score_table[score_table["rank"] <= sell_below_rank]["code"].astype(str).tolist()
    )
    hold_codes = {
        code for code in current_positions
        if code in buffer_codes and code not in selected_codes
    }
    target_codes = sorted(selected_codes | hold_codes)
    target_weight = min(max_weight, 1.0 / len(target_codes)) if target_codes else 0.0

    score_by_code = score_table.set_index("code")
    all_codes = list(dict.fromkeys(score_table["code"].astype(str).tolist() + list(current_positions)))
    rows = []
    for code in all_codes:
        has_score = code in score_by_code.index
        score_row = score_by_code.loc[code] if has_score else None
        current_weight = float(current_positions.get(code, 0.0))
        if code in selected_codes:
            action = "hold" if current_weight > 0 else "buy"
            next_weight = target_weight
            reason = "top_rank"
        elif code in hold_codes:
            action = "hold"
            next_weight = target_weight
            reason = "inside_sell_buffer"
        elif current_weight > 0:
            action = "sell"
            next_weight = 0.0
            reason = "rank_below_sell_buffer"
        else:
            continue

        rows.append({
            "date": score_row["date"] if has_score else "",
            "model": score_row["model"] if has_score else "",
            "code": code,
            "name": score_row["name"] if has_score else "",
            "action": action,
            "current_weight": current_weight,
            "target_weight": next_weight,
            "score": score_row["score"] if has_score else pd.NA,
            "rank": score_row["rank"] if has_score else pd.NA,
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    signal_parser = subparsers.add_parser("signal")
    signal_parser.add_argument("--scores", type=Path, required=True)
    signal_parser.add_argument("--model", required=True)
    signal_parser.add_argument("--date")
    signal_parser.add_argument("--output", type=Path, default=Path("output/signals/latest.csv"))
    signal_parser.add_argument("--buy-top-n", type=int, default=20)
    signal_parser.add_argument("--sell-below-rank", type=int, default=60)
    signal_parser.add_argument("--max-weight", type=float, default=0.05)
    args = parser.parse_args()

    if args.cmd == "signal":
        scores = pd.read_csv(args.scores, dtype={"code": str})
        scores = scores[scores["model"] == args.model]
        signal_date = args.date or scores["date"].max()
        scores = scores[scores["date"] == signal_date]
        result = generate_signals(
            scores,
            buy_top_n=args.buy_top_n,
            sell_below_rank=args.sell_below_rank,
            max_weight=args.max_weight,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"wrote {len(result)} rows to {args.output}")
        print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
