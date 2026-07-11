# 策略有效性验证与优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `strategies/validate.py`(基准/超额/t 检验/分年)+ `strategies/optimize.py`(训练/留出切分的网格调参与过拟合红牌)+ `metrics.daily_sharpe`。

**Architecture:** 复用 `backtest.run` 与 `REGISTRY`;信号均为 point-in-time,全面板算权重、按日期切段评估;留出段只评估最优组合一次。

**Tech Stack:** pandas / numpy / pytest;命令一律 `.venv/bin/python -m pytest strategies/tests -q`。

## Global Constraints

- 预热期(空仓)剔除:统计从首个实际持仓日开始(`weights.shift(1)` 后首个非零日)。
- t 检验:`mean/std(ddof=0)×√n`,iid 简化,阈值 ±2。
- 留出段只用一次:敏感性表只含训练段夏普,唯最优组合评估留出段。
- 不引入新依赖。

---

### Task 1: metrics.daily_sharpe

**Files:**
- Modify: `strategies/metrics.py`
- Test: `strategies/tests/test_metrics.py`(追加)

**Interfaces:**
- Produces: `metrics.daily_sharpe(net: Series) -> float`(日收益年化夏普,std=0 或空返回 NaN)。

- [ ] **Step 1: 追加失败测试**

```python
def test_daily_sharpe_hand_computed():
    net = pd.Series([0.01, -0.01, 0.02])
    expected = net.mean() * 252 / (net.std(ddof=0) * np.sqrt(252))
    assert metrics.daily_sharpe(net) == pytest.approx(expected)
    assert np.isnan(metrics.daily_sharpe(pd.Series([0.01, 0.01])))  # std=0
    assert np.isnan(metrics.daily_sharpe(pd.Series([], dtype=float)))
```

- [ ] **Step 2: 运行确认失败**(AttributeError)
- [ ] **Step 3: 实现**

```python
def daily_sharpe(net) -> float:
    """日收益序列的年化夏普(mean×252 / (std×√252));空或零波动返回 NaN。"""
    if len(net) == 0:
        return float("nan")
    std = net.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(net.mean() * TRADING_DAYS / (std * np.sqrt(TRADING_DAYS)))
```

- [ ] **Step 4: 确认通过;Step 5: Commit** `feat(strategies): metrics.daily_sharpe`

---

### Task 2: validate.py

**Files:**
- Create: `strategies/validate.py`
- Test: `strategies/tests/test_validate.py`

**Interfaces:**
- Consumes: `backtest.run`、`run.REGISTRY`、`run.OUTPUT_DIR`、`metrics.daily_sharpe/TRADING_DAYS`。
- Produces:
  - `benchmark_returns(panel) -> Series`
  - `yearly_table(net: Series, bench: Series) -> DataFrame`(index=年,列 strategy/benchmark/excess)
  - `validate_one(name, panel, market, cost_bps=20.0) -> dict`,键:
    `strategy/live_start/strat_annual/bench_annual/excess_annual/sharpe/t_stat/max_drawdown/verdict`
  - CLI `main()`:`--market --strategy|--all --top --cost-bps`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
import pytest

from strategies import validate


def test_benchmark_is_cross_section_mean(make_panel):
    panel = make_panel({"a": [100, 110], "b": [100, 90]})
    bench = validate.benchmark_returns(panel)
    assert bench.iloc[1] == pytest.approx((0.10 + (-0.10)) / 2)


def test_yearly_table_splits_calendar_years():
    idx = pd.bdate_range("2024-12-30", periods=4)   # 两天 2024,两天 2025
    net = pd.Series([0.01, 0.02, 0.0, 0.01], index=idx)
    bench = pd.Series(0.0, index=idx)
    table = validate.yearly_table(net, bench)
    assert table.loc[2024, "strategy"] == pytest.approx(1.01 * 1.02 - 1)
    assert table.loc[2025, "excess"] == pytest.approx(1.01 - 1)


def test_validate_one_excludes_warmup(make_panel, monkeypatch):
    n = 30
    panel = make_panel({
        "w": [100 * 1.01 ** i for i in range(n)],
        "f": [100 + 0.5 * (-1) ** i for i in range(n)],
    })
    idx = panel["close"].index
    stub = pd.DataFrame(0.0, index=idx, columns=["w", "f"])
    stub.iloc[5:, 0] = 1.0                    # 第 5 日发信号,第 6 日起持仓
    monkeypatch.setitem(validate.REGISTRY, "stub", lambda panel: stub)

    r = validate.validate_one("stub", panel, "us", cost_bps=0.0)

    assert r["live_start"] == idx[6]          # 预热期剔除
    net = pd.Series(panel["close"]["w"].pct_change(), index=idx).loc[idx[6]:]
    bench = validate.benchmark_returns(panel).loc[idx[6]:]
    excess = net - bench
    expected_t = excess.mean() / excess.std(ddof=0) * np.sqrt(len(excess))
    assert r["t_stat"] == pytest.approx(expected_t)
    assert r["verdict"] in {"显著跑赢基准", "超额不显著"}


def test_validate_one_never_live(make_panel, monkeypatch):
    panel = make_panel({"a": [100.0] * 10, "b": [100.0] * 10})
    idx = panel["close"].index
    zero = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    monkeypatch.setitem(validate.REGISTRY, "zero", lambda panel: zero)

    r = validate.validate_one("zero", panel, "us")

    assert r["verdict"] == "从未建仓"
```

- [ ] **Step 2: 运行确认失败**(module 不存在)
- [ ] **Step 3: 实现 validate.py**

```python
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
        stats.update({k: float("nan") for k in (
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
```

- [ ] **Step 4: 确认通过;Step 5: Commit** `feat(strategies): validate 有效性检验(基准/超额/t检验/分年)`

---

### Task 3: optimize.py

**Files:**
- Create: `strategies/optimize.py`
- Test: `strategies/tests/test_optimize.py`

**Interfaces:**
- Consumes: `backtest.run`、`run.REGISTRY`、`metrics.daily_sharpe`。
- Produces:
  - `GRIDS: dict[str, dict[str, list]]`、`combos(name) -> list[dict]`
  - `split_dates(index, ratio=0.6) -> (DatetimeIndex, DatetimeIndex)`
  - `grid_search(name, panel, ratio=0.6, cost_bps=20.0) -> dict`,键:
    `table`(DataFrame: 参数列+train_sharpe,降序)、`best`(dict)、
    `train_sharpe`、`holdout_sharpe`、`overfit_flag`
  - CLI `main()`:`--market --strategy --top --ratio --cost-bps`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
import pytest

from strategies import optimize


def test_combos_respect_constraint(monkeypatch):
    monkeypatch.setitem(optimize.GRIDS, "ma_cross",
                        {"fast": [20, 50, 100], "slow": [60, 120, 200]})
    result = optimize.combos("ma_cross")
    assert len(result) == 8                       # (100,60) 被 fast<slow 滤掉
    assert all(p["fast"] < p["slow"] for p in result)


def test_split_dates_no_overlap():
    idx = pd.bdate_range("2024-01-01", periods=10)
    train, holdout = optimize.split_dates(idx, ratio=0.6)
    assert len(train) == 6 and len(holdout) == 4
    assert train.intersection(holdout).empty
    assert train.union(holdout).equals(idx)


def test_grid_search_picks_active_combo(make_panel, monkeypatch):
    n = 30
    panel = make_panel({"up": [100 * 1.02 ** i for i in range(n)],
                        "dn": [100 * 0.99 ** i for i in range(n)]})
    monkeypatch.setitem(optimize.GRIDS, "ma_cross", {"fast": [2], "slow": [4, 25]})
    r = optimize.grid_search("ma_cross", panel, ratio=0.6)
    assert r["best"] == {"fast": 2, "slow": 4}    # slow=25 在训练段几乎无信号
    assert r["train_sharpe"] > 0


def test_overfit_flag_on_reversal(make_panel, monkeypatch):
    # 训练段动量有效,留出段反转 -> 留出夏普为负 -> 红牌
    n = 50
    up_then_down = [100 * 1.03 ** i for i in range(30)] + \
                   [100 * 1.03 ** 30 * 0.95 ** i for i in range(1, 21)]
    panel = make_panel({"a": up_then_down, "b": [100 + (i % 2) for i in range(n)],
                        "c": [100 - 0.1 * i for i in range(n)]})
    monkeypatch.setitem(optimize.GRIDS, "momentum",
                        {"top_n": [1], "lookback": [10], "skip": [2], "rebalance": [5]})
    r = optimize.grid_search("momentum", panel, ratio=0.6)
    assert r["overfit_flag"] is True
```

- [ ] **Step 2: 运行确认失败**(module 不存在)
- [ ] **Step 3: 实现 optimize.py**

```python
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
```

- [ ] **Step 4: 确认通过(全量 strategies 测试);Step 5: Commit** `feat(strategies): optimize 网格调参+过拟合红牌`

---

### Task 4: README + 真实数据验证运行

**Files:**
- Modify: `strategies/README.md`(追加 validate/optimize 用法)
- 依赖:后台全市场抓取完成(us/hk)

- [ ] **Step 1: README 追加**

```markdown
## 有效性验证与调参

    python -m strategies.validate --market us --all          # 基准/超额/t检验
    python -m strategies.validate --market us --strategy momentum  # 加分年表
    python -m strategies.optimize --market us --strategy ma_cross  # 训练/留出调参

结论口径:t≥2 显著跑赢;|t|<2 超额不显著(不能排除运气);
优化的留出段只用一次,红牌参数勿采用。已退市股缺失 = 残余幸存者偏差。
```

- [ ] **Step 2: 全量测试 + 真实数据运行**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m strategies.validate --market us --all --top 500
.venv/bin/python -m strategies.validate --market hk --all --top 500
.venv/bin/python -m strategies.optimize --market us --strategy ma_cross
.venv/bin/python -m strategies.optimize --market us --strategy momentum
```

- [ ] **Step 3: Commit** `docs(strategies): 验证与调参用法`
