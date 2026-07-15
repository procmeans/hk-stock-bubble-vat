# THS Heat Paper Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two auditable A-share paper-trading accounts that buy the iFinD Top 20 by absolute stock heat and by day-over-day heat-rank growth.

**Architecture:** Add a thin iFinD smart-stock-picking client, keep dynamic-column parsing and signal normalization in `strategies/ths_heat.py`, and pass prefetched targets into the network-free paper step. The A-share runner keeps one shared base panel, supplements prices only for heat picks outside it, isolates failures per strategy, and persists the exact signals used.

**Tech Stack:** Python 3.11+, pandas, numpy, requests, pytest, static HTML/JavaScript, GitHub Actions YAML.

## Global Constraints

- Accounts: `a_ths_heat` and `a_ths_heat_rise`.
- Initial capital: ¥100,000 per account.
- Portfolio: Top 20 valid stocks, equal weighted.
- Rebalance every 2 trading days; T-close signal fills at T+1 close.
- Cost: 20bp per side on absolute traded notional.
- Benchmark: existing A-share top-500 liquidity universe, isolated from supplemental prices.
- Use official iFinD `smart_stock_picking`; do not use unofficial App endpoints.
- Query date must equal the latest A-share close-panel date.
- Reuse `THS_HTTP_REFRESH_TOKEN`; never log or persist tokens.
- A signal failure pauses only that strategy's rebalance; other accounts continue.
- Preserve existing strategy parameters and selection logic.
- Paper simplifications remain: fractional shares; no lot-size, price-limit, or impact model.
- Use TDD and keep every commit limited to its named task.

## File Map

- Create `alpha101/tests/test_ths_http.py`; modify `alpha101/ths_http.py` for the HTTP wrapper.
- Create `strategies/ths_heat.py` and `strategies/tests/test_ths_heat.py` for factor logic.
- Modify `strategies/paper.py` and `strategies/tests/test_paper.py` for execution and orchestration.
- Add two account directories plus `paper/ths_heat_signals.csv`.
- Modify `paper/accounts.json`, `paper.html`, `index.html`, `strategies/README.md`, and `.github/workflows/paper-a.yml`.

---

### Task 1: Official iFinD smart-stock-picking wrapper

**Files:**
- Create: `alpha101/tests/test_ths_http.py`
- Modify: `alpha101/ths_http.py:54-90`

**Interfaces:**
- Consumes: existing `post()` and `tables_to_dataframe()`.
- Produces: `smart_stock_picking(searchstring, searchtype="stock", access_token=None, refresh_token=None, timeout=30) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing wrapper test**

Create `alpha101/tests/test_ths_http.py`:

```python
import pandas as pd

from alpha101 import ths_http


def test_smart_stock_picking_posts_query_and_flattens(monkeypatch):
    seen = {}

    def fake_post(endpoint, payload, **kwargs):
        seen.update(endpoint=endpoint, payload=payload, kwargs=kwargs)
        return {"tables": [{"table": {
            "股票代码": ["000001.SZ"],
            "个股热度[20260715]": [123.5],
        }}]}

    monkeypatch.setattr(ths_http, "post", fake_post)
    result = ths_http.smart_stock_picking(
        "2026年7月15日个股热度排名前20", access_token="access", timeout=9
    )

    assert seen == {
        "endpoint": "smart_stock_picking",
        "payload": {
            "searchstring": "2026年7月15日个股热度排名前20",
            "searchtype": "stock",
        },
        "kwargs": {"access_token": "access", "refresh_token": None, "timeout": 9},
    }
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "股票代码"] == "000001.SZ"
    assert result.loc[0, "个股热度[20260715]"] == 123.5
```

- [ ] **Step 2: Verify the test fails**

Run: `.venv/bin/python -m pytest alpha101/tests/test_ths_http.py -q`

Expected: FAIL because `smart_stock_picking` is absent.

- [ ] **Step 3: Add the wrapper after `history_quotation()`**

```python
def smart_stock_picking(
    searchstring: str,
    searchtype: str = "stock",
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Run an iFinD semantic stock query and return a flat DataFrame."""
    data = post(
        "smart_stock_picking",
        {"searchstring": searchstring, "searchtype": searchtype},
        access_token=access_token,
        refresh_token=refresh_token,
        timeout=timeout,
    )
    return tables_to_dataframe(data)
```

- [ ] **Step 4: Run focused iFinD tests**

Run: `.venv/bin/python -m pytest alpha101/tests/test_ths_http.py alpha101/tests/test_ths_today.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add alpha101/ths_http.py alpha101/tests/test_ths_http.py
git commit -m "feat(alpha101): add iFinD smart stock query"
```

---

### Task 2: Heat signal module

**Files:**
- Create: `strategies/ths_heat.py`
- Create: `strategies/tests/test_ths_heat.py`

**Interfaces:**
- Consumes: `alpha101.ths_http.smart_stock_picking()`.
- Produces `build_query()`, `normalize_signal()`, `fetch_signal()`, and `target_weights()`; normalized columns are `date,strategy,rank,ticker,name,factor_value`.

- [ ] **Step 1: Write failing factor tests**

Create `strategies/tests/test_ths_heat.py`:

```python
import pandas as pd
import pytest

from strategies import ths_heat


def test_build_query_uses_explicit_exchange_date():
    day = pd.Timestamp("2026-07-15")
    assert ths_heat.build_query(day, "ths_heat", 20) == \
        "2026年7月15日个股热度排名前20"
    assert ths_heat.build_query(day, "ths_heat_rise", 20) == \
        "2026年7月15日个股热度排名环比增长率排名前20"


@pytest.mark.parametrize("strategy,value_column,values", [
    ("ths_heat", "个股热度[20260715]", [100.0, 300.0, None]),
    ("ths_heat_rise", "个股热度排名环比增长率[20260715]", [2.0, 8.0, None]),
])
def test_normalize_signal_matches_dynamic_column_and_sorts(
    strategy, value_column, values
):
    data = pd.DataFrame({
        value_column: values,
        "股票简称": ["甲", "乙", "坏值"],
        "股票代码": ["000001.SZ", "688001.SH", "bad"],
    })
    result = ths_heat.normalize_signal(
        data, pd.Timestamp("2026-07-15"), strategy, top_n=20
    )
    assert result.columns.tolist() == [
        "date", "strategy", "rank", "ticker", "name", "factor_value"
    ]
    assert result["ticker"].tolist() == ["688001", "000001"]
    assert result["rank"].tolist() == [1, 2]
    assert result["factor_value"].tolist() == [values[1], values[0]]


def test_normalize_signal_rejects_stale_dynamic_date():
    data = pd.DataFrame({
        "股票代码": ["000001.SZ"], "股票简称": ["甲"],
        "个股热度[20260714]": [100.0],
    })
    with pytest.raises(ValueError, match="20260715"):
        ths_heat.normalize_signal(
            data, pd.Timestamp("2026-07-15"), "ths_heat", top_n=20
        )


def test_normalize_signal_rejects_missing_code_column():
    data = pd.DataFrame({"个股热度[20260715]": [100.0]})
    with pytest.raises(ValueError, match="股票代码"):
        ths_heat.normalize_signal(
            data, pd.Timestamp("2026-07-15"), "ths_heat", top_n=20
        )


def test_normalize_signal_rejects_empty_valid_rows():
    data = pd.DataFrame({
        "股票代码": ["bad"], "股票简称": ["坏值"],
        "个股热度[20260715]": [None],
    })
    with pytest.raises(ValueError, match="empty ths_heat signal"):
        ths_heat.normalize_signal(
            data, pd.Timestamp("2026-07-15"), "ths_heat", top_n=20
        )


def test_target_weights_drops_unpriced_names_and_renormalizes():
    signal = pd.DataFrame({"ticker": ["A", "B", "C"]})
    prices = pd.Series({"A": 10.0, "B": float("nan"), "C": 30.0})
    assert ths_heat.target_weights(signal, prices) == {"A": 0.5, "C": 0.5}


def test_fetch_signal_passes_explicit_query(monkeypatch):
    seen = {}
    raw = pd.DataFrame({
        "股票代码": ["000001.SZ"], "股票简称": ["甲"],
        "个股热度[20260715]": [100.0],
    })

    def fake_query(searchstring, **kwargs):
        seen.update(searchstring=searchstring, kwargs=kwargs)
        return raw

    monkeypatch.setattr(ths_heat.ths_http, "smart_stock_picking", fake_query)
    result = ths_heat.fetch_signal(
        pd.Timestamp("2026-07-15"), "ths_heat", top_n=20,
        access_token="access",
    )
    assert seen["searchstring"] == "2026年7月15日个股热度排名前20"
    assert seen["kwargs"] == {"access_token": "access"}
    assert result.loc[0, "ticker"] == "000001"
```

- [ ] **Step 2: Verify import failure**

Run: `.venv/bin/python -m pytest strategies/tests/test_ths_heat.py -q`

Expected: FAIL because `strategies.ths_heat` does not exist.

- [ ] **Step 3: Implement `strategies/ths_heat.py`**

```python
"""同花顺个股热度与热度排名上升单因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101 import ths_http

STRATEGIES = {"ths_heat", "ths_heat_rise"}
SIGNAL_COLUMNS = ["date", "strategy", "rank", "ticker", "name", "factor_value"]
VALUE_PREFIX = {
    "ths_heat": "个股热度",
    "ths_heat_rise": "个股热度排名环比增长率",
}


def _day(signal_date) -> pd.Timestamp:
    return pd.Timestamp(signal_date).normalize()


def build_query(signal_date, strategy: str, top_n: int) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported THS heat strategy: {strategy}")
    day = _day(signal_date)
    return f"{day.year}年{day.month}月{day.day}日{VALUE_PREFIX[strategy]}排名前{int(top_n)}"


def _value_column(data: pd.DataFrame, signal_date, strategy: str) -> str:
    stamp = _day(signal_date).strftime("%Y%m%d")
    expected = f"{VALUE_PREFIX[strategy]}[{stamp}]"
    if expected not in data.columns:
        raise ValueError(f"missing {VALUE_PREFIX[strategy]} column for {stamp}")
    return expected


def normalize_signal(data, signal_date, strategy: str, top_n: int = 20):
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported THS heat strategy: {strategy}")
    if "股票代码" not in data.columns:
        raise ValueError("missing 股票代码 column")
    value_column = _value_column(data, signal_date, strategy)
    result = pd.DataFrame({
        "ticker": data["股票代码"].astype(str).str.extract(r"(\d{6})", expand=False),
        "name": data["股票简称"].astype(str)
        if "股票简称" in data.columns else data["股票代码"].astype(str),
        "factor_value": pd.to_numeric(data[value_column], errors="coerce"),
    }).dropna(subset=["ticker", "factor_value"])
    result = result[np.isfinite(result["factor_value"])]
    result = result.drop_duplicates("ticker").sort_values(
        ["factor_value", "ticker"], ascending=[False, True]
    ).head(int(top_n)).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"empty {strategy} signal for {_day(signal_date).date()}")
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "strategy", strategy)
    result.insert(0, "date", _day(signal_date).strftime("%Y-%m-%d"))
    return result[SIGNAL_COLUMNS]


def fetch_signal(signal_date, strategy: str, top_n: int = 20, access_token=None):
    raw = ths_http.smart_stock_picking(
        build_query(signal_date, strategy, top_n), access_token=access_token
    )
    return normalize_signal(raw, signal_date, strategy, top_n=top_n)


def target_weights(signal: pd.DataFrame, prices: pd.Series) -> dict[str, float]:
    tickers = signal["ticker"].astype(str).tolist()
    quoted = pd.to_numeric(prices.reindex(tickers), errors="coerce")
    valid = [ticker for ticker, price in quoted.items()
             if pd.notna(price) and np.isfinite(price) and price > 0]
    if not valid:
        return {}
    return {ticker: 1.0 / len(valid) for ticker in valid}
```

- [ ] **Step 4: Run tests and diff checks**

Run:

```bash
.venv/bin/python -m pytest strategies/tests/test_ths_heat.py -q
git diff --check
```

Expected: tests PASS and diff check is silent.

- [ ] **Step 5: Commit**

```bash
git add strategies/ths_heat.py strategies/tests/test_ths_heat.py
git commit -m "feat(strategies): add THS heat factors"
```

---

### Task 3: Paper core support for prefetched targets and isolated benchmark

**Files:**
- Modify: `strategies/paper.py:16-162`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: normalized target dictionaries from Task 2.
- Produces `THS_HEAT_PARAMS`, `HEAT_STRATEGIES`, `rebalance_due()`, a `target_override` argument on `compute_targets()` and `step()`, and optional `panel["benchmark_close"]` support.

- [ ] **Step 1: Add failing paper-core tests**

Append to `strategies/tests/test_paper.py`:

```python
@pytest.mark.parametrize("strategy", ["ths_heat", "ths_heat_rise"])
def test_step_uses_prefetched_heat_targets(strategy):
    state = _state()
    state["strategy"] = strategy
    state["params"] = {"top_n": 20, "rebalance": 2}
    state, _, _ = paper.step(
        state, {"close": _close()}, target_override={"B": 1.0}
    )
    assert state["pending_targets"] == {"B": 1.0}


def test_heat_rebalance_two_holds_for_two_closes():
    close = _close(15)
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 1, "rebalance": 2}
    state, _, _ = paper.step(
        state, {"close": close.iloc[:12]}, target_override={"A": 1.0}
    )
    state, _, _ = paper.step(state, {"close": close.iloc[:13]})
    assert state["pending_targets"] is None and "A" in state["positions"]
    state, _, _ = paper.step(
        state, {"close": close.iloc[:14]}, target_override={"B": 1.0}
    )
    assert state["pending_targets"] == {"B": 1.0}
    state, _, _ = paper.step(state, {"close": close.iloc[:15]})
    assert "B" in state["positions"] and "A" not in state["positions"]


def test_benchmark_close_excludes_supplemental_hot_stock():
    idx = pd.bdate_range("2026-07-14", periods=2)
    close = pd.DataFrame({"A": [10.0, 10.0], "HOT": [10.0, 20.0]}, index=idx)
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 1, "rebalance": 2}
    state, nav_row, _ = paper.step(
        state,
        {"close": close, "benchmark_close": close[["A"]]},
        target_override={},
    )
    assert nav_row["bench_nav"] == 100000.0


def test_failed_due_heat_signal_stays_due_for_next_day():
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 20, "rebalance": 2}
    state["days_since_rebalance"] = 2
    state, _, _ = paper.step(state, {"close": _close()}, target_override={})
    assert state["pending_targets"] is None
    assert paper.rebalance_due(state) is True
```

- [ ] **Step 2: Verify the new tests fail**

Run the four named tests with:

```bash
.venv/bin/python -m pytest strategies/tests/test_paper.py -k \
  "prefetched_heat or heat_rebalance_two or benchmark_close or failed_due_heat" -q
```

Expected: FAIL because `step()` has no `target_override` and `rebalance_due()` is absent.

- [ ] **Step 3: Register both strategies and pure target overrides**

Update the constants and replace `compute_targets()`:

```python
THS_HEAT_PARAMS = {"top_n": 20, "rebalance": 2}
HEAT_STRATEGIES = {"ths_heat", "ths_heat_rise"}
DEFAULT_PARAMS = {
    "momentum": PARAMS, "alpha101": A101_PARAMS,
    "equal_weight": EW_PARAMS, "ths_heat": THS_HEAT_PARAMS,
    "ths_heat_rise": THS_HEAT_PARAMS,
}


def compute_targets(strategy, panel, params, target_override=None) -> dict:
    if target_override is not None:
        return target_override
    if strategy in HEAT_STRATEGIES:
        return {}
    if strategy == "alpha101":
        from strategies.alpha101_composite import targets
        return targets(panel, **params)
    if strategy == "equal_weight":
        from strategies.equal_weight import targets
        return targets(panel, **params)
    return target_weights(panel["close"], **params)


def rebalance_due(state: dict, params=None) -> bool:
    params = params or state.get("params") or PARAMS
    days = state.get("days_since_rebalance")
    return days is None or days + 1 >= params["rebalance"]
```

- [ ] **Step 4: Extend `step()` while preserving accounting code**

Use this signature:

```python
def step(state, panel, params=None, cost_bps=COST_BPS, target_override=None):
```

Replace only benchmark calculation and the due block:

```python
    benchmark_close = panel.get("benchmark_close", close)
    daily = benchmark_close.ffill().pct_change().iloc[-1]
    bench_ret = float(np.nanmean(daily)) if not daily.isna().all() else 0.0
    state["bench_nav"] = float(state["bench_nav"] * (1.0 + bench_ret))

    if state["days_since_rebalance"] is None:
        due = True
    else:
        state["days_since_rebalance"] += 1
        due = state["days_since_rebalance"] >= params["rebalance"]
    if due:
        new_targets = compute_targets(
            state.get("strategy", "momentum"), panel, params,
            target_override=target_override,
        )
        if new_targets:
            state["pending_targets"] = new_targets
            state["days_since_rebalance"] = 0
```

Do not change fill ordering, cost deductions, NAV accounting, or same-day idempotency.

- [ ] **Step 5: Run all paper tests**

Run: `.venv/bin/python -m pytest strategies/tests/test_paper.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add strategies/paper.py strategies/tests/test_paper.py
git commit -m "feat(strategies): support prefetched paper targets"
```

---

### Task 4: A-share heat orchestration, supplemental prices, and audit log

**Files:**
- Modify: `strategies/paper.py:164-314`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: Task 2 signal functions and Task 3 target overrides.
- Produces `_merge_panel()`, `append_heat_signals()`, `prepare_heat_targets()`, plus optional `heat_fetch` injection on `run()` and `run_market()`.

- [ ] **Step 1: Add failing orchestration tests**

Append to `strategies/tests/test_paper.py`:

```python
def _a_panel(columns=("A", "B"), n=20):
    base = _close(n)
    close = pd.DataFrame(index=base.index)
    for column in columns:
        close[column] = base[column] if column in base else 50.0
    return {
        "close": close, "volume": close * 0 + 1000.0,
        "amount": close * 1000.0, "returns": close.pct_change(),
    }


def _signal(strategy, ticker, value):
    return pd.DataFrame([{
        "date": "2024-01-26", "strategy": strategy, "rank": 1,
        "ticker": ticker, "name": ticker, "factor_value": value,
    }])


def test_run_market_fetches_due_heat_once_and_supplements_prices(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    paper.init("rise", 100000.0, strategy="ths_heat_rise", market="a")
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    panel_calls, signal_calls = [], []

    def fake_panel(codes, window_days=0):
        panel_calls.append(set(codes))
        return _a_panel(tuple(sorted(codes))) if set(codes) <= {"C", "D"} else _a_panel()

    def fake_signal(day, strategy, top_n=20):
        signal_calls.append((strategy, top_n))
        return _signal(strategy, "C" if strategy == "ths_heat" else "D", 9.0)

    paper.run_market("a", fetch=fake_panel, heat_fetch=fake_signal)
    assert len(panel_calls) == 2 and panel_calls[1] == {"C", "D"}
    assert {item[0] for item in signal_calls} == {"ths_heat", "ths_heat_rise"}
    assert paper.load_state("heat")["pending_targets"] == {"C": 1.0}
    assert paper.load_state("rise")["pending_targets"] == {"D": 1.0}


def test_heat_query_failure_isolated_per_account(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    paper.init("rise", 100000.0, strategy="ths_heat_rise", market="a")
    paper.init(
        "plain", 100000.0, strategy="momentum", market="a",
        params={"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5},
    )
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])

    def fake_signal(day, strategy, top_n=20):
        if strategy == "ths_heat":
            raise RuntimeError("temporary API failure")
        return _signal(strategy, "A", 8.0)

    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(),
        heat_fetch=fake_signal,
    )
    assert paper.load_state("heat")["pending_targets"] is None
    assert paper.load_state("rise")["pending_targets"] == {"A": 1.0}
    assert paper.load_state("plain")["last_run"] == "2024-01-26"
    audit = pd.read_csv(tmp_path / "ths_heat_signals.csv")
    assert set(audit["status"]) == {"ok", "error"}


def test_non_due_heat_account_does_not_query(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    state = paper.load_state("heat")
    state["days_since_rebalance"] = 0
    paper.save_state("heat", state)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(),
        heat_fetch=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("heat endpoint called on non-due day")
        ),
    )


def test_append_heat_signals_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    rows = [{
        "date": "2026-07-15", "strategy": "ths_heat", "rank": 1,
        "ticker": "000001", "name": "甲", "factor_value": 100.0,
        "status": "ok", "error": "",
    }]
    paper.append_heat_signals(rows)
    paper.append_heat_signals(rows)
    assert len(pd.read_csv(tmp_path / "ths_heat_signals.csv")) == 1
```

- [ ] **Step 2: Verify missing interfaces fail**

Run the four new tests with `pytest ... -k "supplements_prices or failure_isolated or non_due_heat or idempotent" -q`.

Expected: FAIL because `heat_fetch` and `append_heat_signals()` are absent.

- [ ] **Step 3: Preserve the benchmark before supplemental prices**

Replace the A branch in `_market_panel()`:

```python
    if market == "a":
        codes = sorted(set(a_universe_tickers()) | held)
        panel = (fetch or fetch_a_panel)(codes, window_days=WINDOW_DAYS)
        benchmark_codes = panel["amount"].tail(60).mean() \
            .nlargest(UNIVERSE_SIZE).index.tolist()
        keep = set(benchmark_codes) | held
        result = {
            key: value[[column for column in value.columns if column in keep]]
            if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
        result["benchmark_close"] = panel["close"].reindex(columns=benchmark_codes)
        return result
```

- [ ] **Step 4: Add merge and idempotent audit helpers**

```python
HEAT_SIGNAL_COLUMNS = [
    "date", "strategy", "rank", "ticker", "name", "factor_value",
    "status", "error",
]


def _merge_panel(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, pd.DataFrame) and isinstance(merged.get(key), pd.DataFrame):
            merged[key] = merged[key].combine_first(value)
        else:
            merged.setdefault(key, value)
    return merged


def _error_row(date, strategy, error):
    return {
        "date": date, "strategy": strategy, "rank": "", "ticker": "",
        "name": "", "factor_value": "", "status": "error",
        "error": " ".join(str(error).split())[:300],
    }


def append_heat_signals(rows, path=None):
    if not rows:
        return
    path = path or PAPER_DIR / "ths_heat_signals.csv"
    incoming = pd.DataFrame(rows, columns=HEAT_SIGNAL_COLUMNS)
    existing = pd.read_csv(path, dtype={"ticker": str}) \
        if path.exists() and path.stat().st_size else pd.DataFrame()
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(
        ["date", "strategy", "rank", "ticker", "status"], keep="first"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
```

- [ ] **Step 5: Add `prepare_heat_targets()`**

Import `ths_heat` beside `momentum`, then add:

```python
def prepare_heat_targets(states, panel, fetch_panel, fetch_signal=None):
    signal_date = panel["close"].index[-1]
    date_text = signal_date.strftime("%Y-%m-%d")
    due = {
        account: state for account, state in states.items()
        if state.get("strategy") in HEAT_STRATEGIES
        and state.get("last_run") != date_text and rebalance_due(state)
    }
    if not due:
        return panel, {}, []
    loader = fetch_signal or ths_heat.fetch_signal
    signals, audit = {}, []
    for strategy in sorted({state["strategy"] for state in due.values()}):
        top_n = next(state["params"]["top_n"] for state in due.values()
                     if state["strategy"] == strategy)
        try:
            signals[strategy] = loader(signal_date, strategy, top_n=top_n)
        except Exception as error:
            audit.append(_error_row(date_text, strategy, error))
    missing = sorted({ticker for signal in signals.values()
                      for ticker in signal["ticker"].astype(str)
                      if ticker not in panel["close"].columns})
    if missing:
        try:
            panel = _merge_panel(panel, fetch_panel(missing, window_days=WINDOW_DAYS))
        except Exception as error:
            for strategy in sorted(signals):
                audit.append(_error_row(date_text, strategy, error))
    prices = panel["close"].ffill().iloc[-1]
    overrides = {}
    for account, state in due.items():
        strategy = state["strategy"]
        signal = signals.get(strategy)
        if signal is None:
            overrides[account] = {}
            continue
        weights = ths_heat.target_weights(signal, prices)
        overrides[account] = weights
        used = signal[signal["ticker"].isin(weights)]
        if used.empty:
            audit.append(_error_row(
                date_text, strategy, RuntimeError("no heat picks have valid prices")
            ))
        else:
            audit.extend({**row, "status": "ok", "error": ""}
                         for row in used.to_dict("records"))
    return panel, overrides, audit
```

- [ ] **Step 6: Thread overrides through `_step_and_persist()`, `run()`, and `run_market()`**

Replace the three functions with:

```python
def _step_and_persist(account, state, panel, target_override=None):
    state, nav_row, orders = step(
        state, panel, target_override=target_override
    )
    if nav_row is None:
        print(f"{account}: {state['last_run']} 已运行过,跳过")
        return
    directory = account_dir(account)
    pd.DataFrame([nav_row]).to_csv(
        directory / "nav.csv", mode="a", header=False, index=False
    )
    if orders:
        pd.DataFrame(orders).to_csv(
            directory / "orders.csv", mode="a", header=False, index=False
        )
    save_state(account, state)
    pending = len(state["pending_targets"] or {})
    print(f"{account} {nav_row['date']}: NAV {nav_row['nav']:,.2f} "
          f"(基准 {nav_row['bench_nav']:,.2f}),成交 {len(orders)} 笔,"
          f"挂单 {pending} 只")


def run(account="us_momentum", fetch=None, heat_fetch=None):
    state = load_state(account)
    market = state.get("market", "us")
    panel = _market_panel(market, _held(state), fetch=fetch)
    overrides, audit = {}, []
    if market == "a":
        panel, overrides, audit = prepare_heat_targets(
            {account: state}, panel, fetch or fetch_a_panel,
            fetch_signal=heat_fetch,
        )
        append_heat_signals(audit)
    _step_and_persist(account, state, panel, overrides.get(account))


def run_market(market, fetch=None, heat_fetch=None):
    """同市场账户共用基础抓数，热度策略仅在到期日预取信号。"""
    manifest = PAPER_DIR / "accounts.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    states = {
        entry["account"]: load_state(entry["account"])
        for entry in entries
        if load_state(entry["account"]).get("market", "us") == market
    }
    if not states:
        raise SystemExit(f"无 {market} 市场账户")
    held = set()
    for state in states.values():
        held |= _held(state)
    panel = _market_panel(market, held, fetch=fetch)
    overrides, audit = {}, []
    if market == "a":
        panel, overrides, audit = prepare_heat_targets(
            states, panel, fetch or fetch_a_panel, fetch_signal=heat_fetch
        )
        append_heat_signals(audit)
    for account, state in states.items():
        _step_and_persist(account, state, panel, overrides.get(account))
```

- [ ] **Step 7: Run all paper tests**

Run: `.venv/bin/python -m pytest strategies/tests/test_paper.py -q`

Expected: all tests PASS, including the pre-existing one-base-fetch test when no heat account is due.

- [ ] **Step 8: Commit**

```bash
git add strategies/paper.py strategies/tests/test_paper.py
git commit -m "feat(strategies): run THS heat paper accounts"
```

---

### Task 5: Initialize accounts and expose both strategies in the UI

**Files:**
- Modify: `paper/accounts.json`
- Create: `paper/a_ths_heat/{state.json,nav.csv,orders.csv}`
- Create: `paper/a_ths_heat_rise/{state.json,nav.csv,orders.csv}`
- Create: `paper/ths_heat_signals.csv`
- Modify: `paper.html:6,85-95`
- Modify: `index.html:59`
- Modify: `strategies/README.md:32-60`
- Modify: `.github/workflows/paper-a.yml:37-44`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: strategy names and parameters from Tasks 3-4.
- Produces: five manifest entries, two initialized stores, a shared audit header, dashboard descriptions, and generic workflow copy.

- [ ] **Step 1: Add the failing repository-artifact test**

Append to `strategies/tests/test_paper.py`:

```python
def test_repository_manifest_has_two_ths_heat_accounts():
    root = paper.Path(__file__).resolve().parents[2]
    entries = json.loads((root / "paper" / "accounts.json").read_text())
    by_account = {entry["account"]: entry for entry in entries}
    assert by_account["a_ths_heat"] == {
        "account": "a_ths_heat", "title": "A股 同花顺热度", "currency": "¥"
    }
    assert by_account["a_ths_heat_rise"] == {
        "account": "a_ths_heat_rise",
        "title": "A股 同花顺热度上升", "currency": "¥",
    }
    for account, strategy in [
        ("a_ths_heat", "ths_heat"),
        ("a_ths_heat_rise", "ths_heat_rise"),
    ]:
        state = json.loads((root / "paper" / account / "state.json").read_text())
        assert state["strategy"] == strategy and state["market"] == "a"
        assert state["params"] == {"top_n": 20, "rebalance": 2}
```

- [ ] **Step 2: Verify missing accounts fail**

Run: `.venv/bin/python -m pytest strategies/tests/test_paper.py::test_repository_manifest_has_two_ths_heat_accounts -q`

Expected: FAIL with `KeyError: 'a_ths_heat'`.

- [ ] **Step 3: Append both entries to `paper/accounts.json`**

```json
 {
  "account": "a_ths_heat",
  "title": "A股 同花顺热度",
  "currency": "¥"
 },
 {
  "account": "a_ths_heat_rise",
  "title": "A股 同花顺热度上升",
  "currency": "¥"
 }
```

- [ ] **Step 4: Create exact account and CSV files**

Create `paper/a_ths_heat/state.json`:

```json
{
 "account": "a_ths_heat", "capital": 100000.0, "cash": 100000.0,
 "positions": {}, "pending_targets": null, "days_since_rebalance": null,
 "bench_nav": 100000.0, "last_run": null, "strategy": "ths_heat",
 "market": "a", "params": {"top_n": 20, "rebalance": 2}
}
```

Create `paper/a_ths_heat_rise/state.json`:

```json
{
 "account": "a_ths_heat_rise", "capital": 100000.0, "cash": 100000.0,
 "positions": {}, "pending_targets": null, "days_since_rebalance": null,
 "bench_nav": 100000.0, "last_run": null, "strategy": "ths_heat_rise",
 "market": "a", "params": {"top_n": 20, "rebalance": 2}
}
```

Both `nav.csv` files:

```csv
date,nav,cash,positions_value,bench_nav
```

Both `orders.csv` files:

```csv
date,ticker,side,shares,price,value,cost
```

Create `paper/ths_heat_signals.csv`:

```csv
date,strategy,rank,ticker,name,factor_value,status,error
```

- [ ] **Step 5: Add dashboard copy**

Change the paper page title:

```html
<title>模拟盘 · 五策略</title>
```

Add to `DESC`:

```javascript
  a_ths_heat: '同花顺个股热度 top 20 等权 · 每 2 交易日调仓 · 虚拟资金 ¥100,000 · 单边 20bp · ' +
    '注:热度反映市场关注而非已验证正向 alpha,高关注可能伴随过度反应',
  a_ths_heat_rise: '同花顺个股热度排名环比增长率 top 20 等权 · 每 2 交易日调仓 · ' +
    '虚拟资金 ¥100,000 · 单边 20bp · 注:信号短期且换手可能较高',
```

Replace the index link with:

```html
<p style="pointer-events:auto"><a href="paper.html" style="color:#8da2c0;font-size:12px">模拟盘 · 五策略 →</a></p>
```

- [ ] **Step 6: Update workflow and README copy**

Use this workflow commit line:

```yaml
git diff --cached --quiet || git commit -m "paper: A-share daily step $(date -u +%F)"
```

Append to the README simulation section:

```markdown
同花顺热度双账户由同一 A 股工作流步进：

- `a_ths_heat`：指定交易日个股热度 top 20；
- `a_ths_heat_rise`：指定交易日个股热度排名环比增长率 top 20。

两者均等权、每 2 个交易日调仓、T 日信号次日收盘成交、单边 20bp。
实际使用的信号写入 `paper/ths_heat_signals.csv`。热度代表关注度而非经过验证的
正向 alpha；高关注可能来自价格过度反应，且热度上升组合可能产生较高换手。
```

- [ ] **Step 7: Validate artifacts and tests**

Run:

```bash
python3 -m json.tool paper/accounts.json >/dev/null
python3 -m json.tool paper/a_ths_heat/state.json >/dev/null
python3 -m json.tool paper/a_ths_heat_rise/state.json >/dev/null
.venv/bin/python -m pytest strategies/tests/test_paper.py -q
```

Expected: JSON commands exit 0 and all paper tests PASS.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/paper-a.yml index.html paper.html paper/ \
  strategies/README.md strategies/tests/test_paper.py
git commit -m "feat(paper): add THS heat accounts to dashboard"
```

---

### Task 6: Full verification and live read-only smoke test

**Files:**
- Verify only; no expected source changes.

**Interfaces:**
- Consumes all Task 1-5 deliverables.
- Produces test, live-schema, credential-safety, and clean-worktree evidence.

- [ ] **Step 1: Run all Alpha101 and strategy tests**

Run: `.venv/bin/python -m pytest alpha101/tests strategies/tests -q`

Expected: zero failures.

- [ ] **Step 2: Validate formatting and artifacts**

```bash
git diff --check
python3 -m json.tool paper/accounts.json >/dev/null
python3 -m json.tool paper/a_ths_heat/state.json >/dev/null
python3 -m json.tool paper/a_ths_heat_rise/state.json >/dev/null
python3 -c "import pandas as pd; pd.read_csv('paper/ths_heat_signals.csv'); print('signal csv ok')"
```

Expected: no diff errors, all JSON parses, and `signal csv ok` prints.

- [ ] **Step 3: Run one live read-only query per factor**

With network approval and without printing credentials, run:

```bash
set -a; source .env; set +a; .venv/bin/python -c "
import pandas as pd
from strategies.ths_heat import fetch_signal
day = pd.Timestamp('2026-07-15')
for strategy in ('ths_heat', 'ths_heat_rise'):
    frame = fetch_signal(day, strategy, top_n=20)
    print(strategy, len(frame), frame.columns.tolist(), frame['date'].unique().tolist())
"
```

Expected: each reports 20 rows, six normalized columns, and only `2026-07-15`. This command must not write account state.

- [ ] **Step 4: Confirm no credentials entered tracked files**

```bash
git status --short
rg -n "THS_HTTP_REFRESH_TOKEN=|refresh_token.*[A-Za-z0-9]{20}" \
  alpha101 strategies paper docs .github || true
```

Expected: `.env` is absent; matches contain only variable names or documentation, never token values.

- [ ] **Step 5: Inspect history and report evidence**

```bash
git log --oneline -7
git status --short
```

Expected: Task 1-5 commits are present and the worktree is clean. Report account names, parameters, focused/full test counts, and live response shapes.
