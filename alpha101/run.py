"""命令行入口:fetch / eval / select。"""
import sys
import pandas as pd
from alpha101 import data, universe, alphas, backtest, compose, select, report


def cmd_fetch():
    data.fetch_all(years=5)
    print("数据缓存完成")


def cmd_eval():
    P = data.load_panel()
    mask = universe.liquidity_mask(P)
    facs = alphas.compute_all(P)
    results = {}
    for n, f in facs.items():
        fm = universe.apply_mask(f, mask)
        ev = backtest.evaluate(fm, P["close"])
        results[n] = ev
        report.factor_figure(f"alpha_{n}", ev, "output/factor_eval")
    df = report.summary_table(results, "output/factor_eval/factor_summary.csv")
    print(df.sort_values("icir", ascending=False).head(15).to_string(index=False))


def cmd_select(top_n=50):
    P = data.load_panel()
    mask = universe.liquidity_mask(P)
    facs = alphas.compute_all(P)
    score = compose.composite(facs, mask=mask)
    last = score.dropna(how="all").index[-1]
    picks = select.pick(score, last, top_n=top_n)
    out = f"output/picks/{last.date()}.csv"
    import os
    os.makedirs("output/picks", exist_ok=True)
    picks.to_csv(out, index=False)
    print(f"已写 {out}")
    print(picks.head(10).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"fetch": cmd_fetch, "eval": cmd_eval, "select": cmd_select}[cmd]()
