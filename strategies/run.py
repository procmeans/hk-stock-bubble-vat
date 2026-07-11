"""CLI:单策略回测或全部策略对比。"""
import argparse
from pathlib import Path

import pandas as pd

from strategies import backtest, data, metrics
from strategies import (
    alpha101_composite, elastic_net, equal_weight, icir_weight, lasso,
    ma_cross, market_neutral, mean_reversion, ml, momentum, pairs,
)

REGISTRY = {
    "alpha101_composite": alpha101_composite.signal,
    "equal_weight": equal_weight.signal,
    "ma_cross": ma_cross.signal,
    "mean_reversion": mean_reversion.signal,
    "momentum": momentum.signal,
    "market_neutral": market_neutral.signal,
    "pairs": pairs.signal,
    "ml": ml.signal,
    "elastic_net": elastic_net.signal,
    "icir_weight": icir_weight.signal,
    "lasso": lasso.signal,
}
OUTPUT_DIR = Path("output/strategies")


def run_one(name, panel, market, cost_bps=20.0):
    result = backtest.run(REGISTRY[name](panel), panel, cost_bps=cost_bps)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / f"{market}_{name}_equity.csv")
    stats = metrics.summary(result)
    if name == "market_neutral" and market == "a":
        stats["note"] = "A股做空为纸面模拟"
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "hk", "a"], required=True)
    parser.add_argument("--strategy", choices=sorted(REGISTRY))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    args = parser.parse_args()
    if not args.all and not args.strategy:
        parser.error("需要 --strategy 或 --all")

    panel = data.load_panel(args.market, top=args.top)
    rows = []
    for name in sorted(REGISTRY) if args.all else [args.strategy]:
        stats = run_one(name, panel, args.market, cost_bps=args.cost_bps)
        rows.append({"strategy": name, **stats})
        print(f"{name}: 完成", flush=True)
    table = pd.DataFrame(rows).set_index("strategy")
    print(table.round(4).to_string())
    if args.all:
        table.to_csv(OUTPUT_DIR / f"{args.market}_compare.csv")


if __name__ == "__main__":
    main()
