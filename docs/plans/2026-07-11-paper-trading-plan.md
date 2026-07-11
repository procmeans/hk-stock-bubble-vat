# 模拟交易实施计划(paper trading)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `strategies/paper.py`(账户状态、每日步进、CLI)+ `.github/workflows/paper.yml` 每日自动运行美股 momentum 纸面实盘。

**Architecture:** 状态文件化(paper/<account>/);step() 纯函数化便于测试(close 面板注入);T 日信号 → T+1 收盘成交;幂等。

**Tech Stack:** pandas / numpy / yfinance;测试合成数据离线。

## Global Constraints

- 时序与回测一致:pending_targets 次日收盘成交;当日重复运行为空操作。
- 会计恒等:cash + positions_value == nav,每步保持;成本单边 20bp 扣现金。
- 参数:momentum lookback=126 / skip=21 / top_n=40 / rebalance=21(经优化+留出验证)。
- step(state, close, params, cost_bps) 的 params 可注入,测试用小参数。

---

### Task 1: 账户与步进核心

**Files:**
- Create: `strategies/paper.py`
- Test: `strategies/tests/test_paper.py`

**Interfaces:**
- Produces:
  - `PARAMS = {"top_n": 40, "lookback": 126, "skip": 21, "rebalance": 21}`,`COST_BPS=20.0`,`UNIVERSE_SIZE=500`,`FETCH_SIZE=800`,`WINDOW_DAYS=400`,`PAPER_DIR=Path("paper")`
  - `init(account, capital)`(已存在则 SystemExit)、`load_state/save_state`
  - `target_weights(close, top_n, lookback, skip, **_) -> dict[str, float]`
  - `step(state, close, params=PARAMS, cost_bps=COST_BPS) -> (state, nav_row|None, orders)`
    (幂等;先成交挂单、再记净值与基准、再调仓判定)

- [ ] **Step 1: 写失败测试**(test_paper.py,见下方全部用例)

```python
import json

import numpy as np
import pandas as pd
import pytest

from strategies import paper

SMALL = {"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5}


def _close(n=20, up=True):
    idx = pd.bdate_range("2024-01-01", periods=n)
    a = [100 * (1.02 if up else 1.0) ** i for i in range(n)]
    b = [100 + 0.1 * (i % 2) for i in range(n)]
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _state(cash=100000.0):
    return {"account": "t", "capital": cash, "cash": cash, "positions": {},
            "pending_targets": None, "days_since_rebalance": None,
            "bench_nav": cash, "last_run": None}


def test_first_step_creates_pending_without_trading():
    state, nav_row, orders = paper.step(_state(), _close(), params=SMALL)
    assert orders == []
    assert state["pending_targets"] == {"A": 1.0}      # 动量最强者
    assert nav_row["nav"] == pytest.approx(100000.0)
    assert nav_row["cash"] == pytest.approx(100000.0)


def test_pending_fills_next_day_with_cost():
    close = _close()
    state, _, _ = paper.step(_state(), close.iloc[:-1], params=SMALL)
    state, nav_row, orders = paper.step(state, close, params=SMALL)
    assert len(orders) == 1 and orders[0]["ticker"] == "A"
    price = close["A"].iloc[-1]
    assert orders[0]["price"] == pytest.approx(round(float(price), 4))
    fee = 100000.0 * 20 / 1e4 * 1.0                    # 全仓买入的单边成本
    assert nav_row["nav"] == pytest.approx(100000.0 - fee, rel=1e-6)
    assert nav_row["nav"] == pytest.approx(
        nav_row["cash"] + nav_row["positions_value"], rel=1e-9)  # 会计恒等


def test_idempotent_same_day():
    close = _close()
    state, _, _ = paper.step(_state(), close, params=SMALL)
    before = json.dumps(state, sort_keys=True)
    state2, nav_row, orders = paper.step(state, close, params=SMALL)
    assert nav_row is None and orders == []
    assert json.dumps(state2, sort_keys=True) == before


def test_rebalance_cadence():
    n = 22
    close = _close(n)
    state = _state()
    pending_days = []
    for k in range(12, n + 1):                          # 逐日步进
        state, _, _ = paper.step(state, close.iloc[:k], params=SMALL)
        pending_days.append(state["pending_targets"] is not None)
    # 首日出信号,次日成交后归 None,直到第 rebalance 个交易日再次出现
    assert pending_days[0] is True
    assert pending_days[1] is False
    assert any(pending_days[2:])                        # 计数到 5 再调仓


def test_target_weights_equal_weight_top_n():
    close = _close(15)
    w = paper.target_weights(close, top_n=2, lookback=10, skip=2)
    assert w == {"A": 0.5, "B": 0.5}


def test_init_creates_files_and_refuses_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 50000.0)
    state = paper.load_state("acct")
    assert state["cash"] == 50000.0
    assert (tmp_path / "acct" / "nav.csv").exists()
    with pytest.raises(SystemExit):
        paper.init("acct", 50000.0)
```

- [ ] **Step 2: 确认失败**(module 不存在)
- [ ] **Step 3: 实现核心**(init/load/save/target_weights/step,代码见设计;
  positions_value 对 NaN 价格计 0 并在 docstring 注明"退市按 0 冲销")
- [ ] **Step 4: 确认通过;Step 5: Commit** `feat(strategies): paper 账户与每日步进核心`

---

### Task 2: run/status CLI 与数据抓取

**Files:**
- Modify: `strategies/paper.py`(追加 universe_tickers/fetch_close_volume/run/status/main)
- Test: `strategies/tests/test_paper.py`(追加)

**Interfaces:**
- Produces:
  - `universe_tickers(snapshot_dir=Path("data")) -> list[str]`(当日快照市值前 FETCH_SIZE)
  - `fetch_close_volume(codes, window_days) -> (close, volume)`(yfinance,复用
    yf_history.normalize_download/to_yf_ticker)
  - `run(account, fetch=fetch_close_volume)`:池=ADV 前 500 ∪ 持仓 ∪ 挂单;
    step;追加 nav/orders CSV;保存 state
  - CLI:`init --capital` / `run` / `status`,`--account` 默认 us_momentum

- [ ] **Step 1: 追加失败测试**

```python
def test_run_appends_nav_and_saves_state(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 100000.0)
    close = _close()
    volume = close * 0 + 1000.0
    monkeypatch.setattr(paper, "universe_tickers", lambda: ["A", "B"])
    paper.run("acct", fetch=lambda codes, window_days=0: (close, volume))
    nav = pd.read_csv(tmp_path / "acct" / "nav.csv")
    assert len(nav) == 1
    state = paper.load_state("acct")
    assert state["last_run"] == close.index[-1].strftime("%Y-%m-%d")
```

- [ ] **Step 2: 确认失败;Step 3: 实现;Step 4: 全量测试通过**
- [ ] **Step 5: Commit** `feat(strategies): paper run/status CLI 与 yfinance 抓取`

---

### Task 3: Actions 工作流 + 建账 + 冒烟 + 文档

**Files:**
- Create: `.github/workflows/paper.yml`
- Create: `paper/us_momentum/`(init 后提交初始状态)
- Modify: `strategies/README.md`

- [ ] **Step 1: paper.yml**

```yaml
name: paper-trading

on:
  schedule:
    - cron: "0 22 * * 1-5"   # UTC,美股收盘后
  workflow_dispatch: {}

permissions:
  contents: write

concurrency:
  group: paper-trading
  cancel-in-progress: false

jobs:
  step:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install deps
        run: pip install pandas numpy yfinance
      - name: Run paper trading step
        run: python -m strategies.paper run
      - name: Commit & push state
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add paper/
          git diff --cached --quiet || git commit -m "paper: daily step $(date -u +%F)"
          git push
```

- [ ] **Step 2: 本地建账并真实冒烟**

```bash
.venv/bin/python -m strategies.paper init --capital 100000
.venv/bin/python -m strategies.paper run       # 真实抓数,首日出挂单
.venv/bin/python -m strategies.paper status
```

- [ ] **Step 3: README 追加模拟盘用法;Step 4: 全量测试;Step 5: Commit**
  `feat(strategies): 模拟盘 Actions 工作流与 us_momentum 账户`
