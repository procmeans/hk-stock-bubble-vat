"""策略有效性检验:等权基准、超额 t 检验、分年对比。

t 检验为 iid 简化(未做自相关修正);基准为逐日等权持有全池。
"""
import argparse

import numpy as np
import pandas as pd

from strategies import backtest, data, metrics
from strategies.run import OUTPUT_DIR, REGISTRY


def benchmark_returns(panel):
    """等权持有全池(逐日等权):截面日收益均值。"""
    return panel["returns"].mean(axis=1)


def yearly_table(net, bench):
    frame = pd.DataFrame({"strategy": net, "benchmark": bench}).fillna(0.0)
    yearly = (1 + frame).groupby(frame.index.year).prod() - 1
    yearly["excess"] = yearly["strategy"] - yearly["benchmark"]
    return yearly


def _annualized(returns):
    if len(returns) == 0:
        return float("nan")
    total = float((1 + returns).prod())
    return total ** (metrics.TRADING_DAYS / len(returns)) - 1


def validate_one(name, panel, market, cost_bps=20.0):
    weights = REGISTRY[name](panel)
    result = backtest.run(weights, panel, cost_bps=cost_bps)
    held = weights.reindex(index=result.index).fillna(0.0).shift(1).abs().sum(axis=1)
    stats = {"strategy": name, "market": market}
    if not held.gt(0).any():
        stats.update({key: float("nan") for key in (
            "strat_annual", "bench_annual", "excess_annual",
            "sharpe", "t_stat", "max_drawdown")})
        stats.update({"live_start": None, "verdict": "从未建仓"})
        return stats

    live = held.gt(0).idxmax()
    net = result["net"].loc[live:]
    bench = benchmark_returns(panel).loc[live:].fillna(0.0)
    excess = net - bench
    std = excess.std(ddof=0)
    t_stat = float(excess.mean() / std * np.sqrt(len(excess))) if std > 0 else float("nan")
    if np.isnan(t_stat):
        verdict = "无法判定"
    elif t_stat >= 2:
        verdict = "显著跑赢基准"
    elif t_stat <= -2:
        verdict = "显著跑输基准"
    else:
        verdict = "超额不显著"
    equity = (1 + net).cumprod()
    stats.update({
        "live_start": live,
        "strat_annual": _annualized(net),
        "bench_annual": _annualized(bench),
        "excess_annual": _annualized(net) - _annualized(bench),
        "sharpe": metrics.daily_sharpe(net),
        "t_stat": t_stat,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "verdict": verdict,
    })
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
        stats = validate_one(name, panel, args.market, cost_bps=args.cost_bps)
        rows.append(stats)
        print(f"{name}: {stats['verdict']}", flush=True)
        if args.strategy:
            weights = REGISTRY[name](panel)
            net = backtest.run(weights, panel, cost_bps=args.cost_bps)["net"]
            print(yearly_table(net, benchmark_returns(panel)).round(4).to_string())
    table = pd.DataFrame(rows).set_index("strategy")
    print(table.round(4).to_string())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / f"{args.market}_validate.csv")


if __name__ == "__main__":
    main()
