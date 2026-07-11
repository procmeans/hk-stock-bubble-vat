"""防过拟合的参数搜索:训练段选参,留出段一次性验货。

信号均为 point-in-time,故全面板算权重、按日期切段评估;
敏感性表只含训练段夏普,留出段只评估最优组合一次。
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from strategies import backtest, data, metrics
from strategies.run import OUTPUT_DIR, REGISTRY

GRIDS = {
    "ma_cross": {"fast": [10, 20, 50], "slow": [60, 120, 200]},
    "mean_reversion": {"window": [10, 20, 40], "entry": [-1.5, -2.0, -2.5]},
    "momentum": {"top_n": [10, 20, 40], "lookback": [126, 252]},
    "market_neutral": {"top_n": [10, 20, 40]},
    "pairs": {"n_pairs": [3, 5, 10], "window": [10, 20, 40]},
    "ml": {"top_n": [10, 20], "horizon": [10, 21]},
    "elastic_net": {"alpha": [0.001, 0.005, 0.02], "top_n": [20, 30]},
    "icir_weight": {"top_n": [20, 30], "horizon": [5, 10]},
    "lasso": {"alpha": [0.001, 0.005, 0.02], "top_n": [10, 20]},
}
CONSTRAINTS = {"ma_cross": lambda p: p["fast"] < p["slow"]}


def combos(name):
    grid = GRIDS[name]
    keys = sorted(grid)
    items = [dict(zip(keys, values))
             for values in itertools.product(*(grid[key] for key in keys))]
    check = CONSTRAINTS.get(name)
    return [p for p in items if check is None or check(p)]


def split_dates(index, ratio=0.6):
    cut = int(len(index) * ratio)
    return index[:cut], index[cut:]


def grid_search(name, panel, ratio=0.6, cost_bps=20.0):
    train_idx, holdout_idx = split_dates(panel["close"].index, ratio)
    scored = []
    for params in combos(name):
        weights = REGISTRY[name](panel, **params)
        net = backtest.run(weights, panel, cost_bps=cost_bps)["net"]
        sharpe = metrics.daily_sharpe(net.loc[train_idx])
        scored.append((params, sharpe))
    valid = [(p, s) for p, s in scored if not np.isnan(s)]
    if not valid:
        raise RuntimeError(f"{name}: 所有参数组合在训练段均无有效收益")
    valid.sort(key=lambda item: item[1], reverse=True)
    best, train_sharpe = valid[0]

    # 留出段只评估最优组合(避免二次挑选污染)
    weights = REGISTRY[name](panel, **best)
    net = backtest.run(weights, panel, cost_bps=cost_bps)["net"]
    holdout_sharpe = metrics.daily_sharpe(net.loc[holdout_idx])
    overfit = bool(np.isnan(holdout_sharpe) or holdout_sharpe < 0
                   or holdout_sharpe < 0.5 * train_sharpe)
    table = pd.DataFrame([{**p, "train_sharpe": s} for p, s in scored]) \
        .sort_values("train_sharpe", ascending=False)
    return {"table": table, "best": best, "train_sharpe": float(train_sharpe),
            "holdout_sharpe": float(holdout_sharpe), "overfit_flag": overfit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "hk", "a"], required=True)
    parser.add_argument("--strategy", choices=sorted(GRIDS), required=True)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    args = parser.parse_args()

    panel = data.load_panel(args.market, top=args.top)
    result = grid_search(args.strategy, panel, ratio=args.ratio,
                         cost_bps=args.cost_bps)
    print("敏感性表(训练段夏普,高原=稳健,尖峰=可疑):")
    print(result["table"].round(4).to_string(index=False))
    print(f"最优参数: {result['best']}")
    print(f"训练段夏普 {result['train_sharpe']:.2f} -> "
          f"留出段夏普 {result['holdout_sharpe']:.2f}")
    if result["overfit_flag"]:
        print("⚠️ 过拟合红牌:留出段明显差于训练段,勿直接采用该参数")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(
        OUTPUT_DIR / f"{args.market}_{args.strategy}_optimize.csv", index=False)


if __name__ == "__main__":
    main()
