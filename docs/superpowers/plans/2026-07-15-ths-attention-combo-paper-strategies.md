# THS Attention Combo Paper Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two auditable A-share paper accounts that compare a weighted three-factor attention strategy with a funnel-style strategy using attention-rank growth, 7-trading-day momentum, and low free-float ratio.

**Architecture:** A focused `strategies/ths_attention_combo.py` module owns the single dated iFinD query, dynamic-column normalization, eligibility filters, factor percentiles, and both deterministic selectors. The existing A-share paper runner calls that query once whenever either new account is due, reuses the shared supplemental price panel, passes prefetched equal-weight targets into the network-free step function, and writes a separate idempotent audit CSV. Account/UI artifacts extend the existing manifest-driven dashboard from five to seven strategies without changing the five existing accounts.

**Tech Stack:** Python 3.11, pandas, NumPy, requests, pytest, JSON/CSV paper stores, static HTML, GitHub Actions, official iFinD HTTP API.

## Global Constraints

- Execute implementation in `.worktrees/feat-ths-attention-combo`; commands below
  therefore use the repository's shared `../../.venv/bin/python` and
  `source ../../.env` rather than assuming a second virtual environment or
  copied secret file inside the worktree.
- New accounts are `a_ths_attention_weighted` / `A股 注意力加权组合` and `a_ths_attention_funnel` / `A股 注意力逐层筛选`, both currency `¥`.
- Each new account starts with `100000.0`, holds at most 20 names equal-weighted, rebalances every 2 trading days, signals at T close, fills only at a valid T+1 close, and pays 20bp one-way cost.
- The shared candidate pool is the explicitly dated iFinD Top 100 by `个股热度排名环比增长率`; one due-day query must also request `流通A股` and `总股本`.
- Eligibility excludes `ST`/`*ST`, fewer than 60 valid close observations, missing/nonpositive T or T-7 close, nonfinite factors, and `float_a / total_shares` outside `(0, 1]`.
- Weighted score is exactly `0.50 * attention_pct + 0.30 * momentum_pct + 0.20 * low_float_pct`.
- Funnel order is exactly positive 7-day momentum, lowest 50% float ratio with `ceil(N/2)`, then Top 20 by attention growth.
- Both accounts share one candidate response and supplemental prices; supplemental candidates never contaminate the original A-liquidity-500 benchmark.
- A shared query failure pauses both new accounts only; a selector failure pauses only its account; existing five accounts continue and failed accounts remain due for the next trading day.
- Audit path is `paper/ths_attention_combo_signals.csv`; secrets and complete API payloads must never be persisted.
- Existing atomic state replacement, keyed NAV/order upserts, failure aggregation, and credential sanitizer remain authoritative.
- No historical attention database, parameter optimization, fourth factor, lot-size/limit-up model, or changes to the existing five strategies.

## File Map

- Create `strategies/ths_attention_combo.py`: query construction, normalization, factor frame, deterministic selectors, and equal-weight targets.
- Create `strategies/tests/test_ths_attention_combo.py`: unit contracts for query/schema, filters, factor math, selectors, and target weights.
- Modify `strategies/paper.py`: strategy registration, due-account orchestration, shared candidate/supplemental fetch, audit rows, and runner wiring.
- Modify `strategies/tests/test_paper.py`: integration, failure isolation, retry, benchmark, audit, and repository-artifact tests.
- Modify `paper/accounts.json`: append the two manifest entries.
- Create `paper/a_ths_attention_weighted/{state.json,nav.csv,orders.csv}` and `paper/a_ths_attention_funnel/{state.json,nav.csv,orders.csv}`.
- Create `paper/ths_attention_combo_signals.csv`: audit header.
- Modify `paper.html` and `index.html`: seven-strategy copy and two descriptions.
- Verify `.github/workflows/paper-a.yml`: keep its schedule/permissions/secret and ensure existing `git add paper/` covers new artifacts; no workflow source change is required.
- Modify `strategies/README.md`: formulas, timing, failure semantics, audit path, and risk warning.

---

### Task 1: Dated iFinD candidate query and normalization

**Files:**
- Create: `strategies/ths_attention_combo.py`
- Create: `strategies/tests/test_ths_attention_combo.py`

**Interfaces:**
- Consumes: `alpha101.ths_http.smart_stock_picking(searchstring, searchtype="stock", access_token=None)`.
- Produces: `STRATEGIES`, `CANDIDATE_COLUMNS`, `build_query(signal_date, candidate_n=100) -> str`, `normalize_candidates(data, signal_date, candidate_n=100) -> pd.DataFrame`, and `fetch_candidates(signal_date, candidate_n=100, access_token=None) -> pd.DataFrame`.
- Normalized columns are exactly `date,ticker,name,attention_rise,float_a,total_shares`.

- [ ] **Step 1: Write failing query and normalization tests**

Create `strategies/tests/test_ths_attention_combo.py` with:

```python
import pandas as pd
import pytest

from strategies import ths_attention_combo as combo


def _raw(day="20260715"):
    return pd.DataFrame({
        "股票代码": ["000002.SZ", "000001.SZ", "BAD"],
        "股票简称": ["乙", "甲", "坏代码"],
        f"流通a股[{day}]": [80.0, 20.0, 10.0],
        f"总股本[{day}]": [100.0, 100.0, 100.0],
        f"个股热度排名环比增长率[{day}]": [5.0, 10.0, 99.0],
    })


def test_build_query_requests_all_fields_once():
    assert combo.build_query(pd.Timestamp("2026-07-15"), 100) == (
        "2026年7月15日个股热度排名环比增长率排名前100，"
        "2026年7月15日流通A股，2026年7月15日总股本"
    )


def test_normalize_candidates_uses_dated_columns_and_stable_order():
    result = combo.normalize_candidates(_raw(), "2026-07-15", candidate_n=100)
    assert result.columns.tolist() == combo.CANDIDATE_COLUMNS
    assert result["ticker"].tolist() == ["000001", "000002"]
    assert result["attention_rise"].tolist() == [10.0, 5.0]
    assert result["date"].unique().tolist() == ["2026-07-15"]


@pytest.mark.parametrize("missing", [
    "股票代码", "股票简称", "流通a股[20260715]",
    "总股本[20260715]", "个股热度排名环比增长率[20260715]",
])
def test_normalize_candidates_rejects_missing_required_column(missing):
    with pytest.raises(ValueError, match="missing"):
        combo.normalize_candidates(_raw().drop(columns=[missing]), "2026-07-15")


def test_fetch_candidates_calls_smart_query_once(monkeypatch):
    seen = []

    def fake_query(query, access_token=None):
        seen.append((query, access_token))
        return _raw()

    monkeypatch.setattr(combo.ths_http, "smart_stock_picking", fake_query)
    result = combo.fetch_candidates("2026-07-15", 100, access_token="token")
    assert len(result) == 2
    assert seen == [(combo.build_query("2026-07-15", 100), "token")]
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_ths_attention_combo.py -q
```

Expected: collection fails with `ImportError` because `strategies.ths_attention_combo` does not exist.

- [ ] **Step 3: Implement the query and schema contract**

Create `strategies/ths_attention_combo.py` with this complete first slice:

```python
"""同花顺注意力上升、7日动量与低流通比例组合策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101 import ths_http

STRATEGIES = {"ths_attention_weighted", "ths_attention_funnel"}
CANDIDATE_COLUMNS = [
    "date", "ticker", "name", "attention_rise", "float_a", "total_shares",
]


def _day(signal_date) -> pd.Timestamp:
    return pd.Timestamp(signal_date).normalize()


def build_query(signal_date, candidate_n: int = 100) -> str:
    day = _day(signal_date)
    prefix = f"{day.year}年{day.month}月{day.day}日"
    return (
        f"{prefix}个股热度排名环比增长率排名前{int(candidate_n)}，"
        f"{prefix}流通A股，{prefix}总股本"
    )


def _dated_column(data: pd.DataFrame, prefix: str, signal_date) -> str:
    expected = f"{prefix}[{_day(signal_date).strftime('%Y%m%d')}]"
    matches = [column for column in data.columns
               if str(column).lower() == expected.lower()]
    if not matches:
        raise ValueError(f"missing {expected} column")
    return matches[0]


def normalize_candidates(data, signal_date, candidate_n: int = 100):
    for column in ("股票代码", "股票简称"):
        if column not in data.columns:
            raise ValueError(f"missing {column} column")
    attention = _dated_column(data, "个股热度排名环比增长率", signal_date)
    float_a = _dated_column(data, "流通a股", signal_date)
    total = _dated_column(data, "总股本", signal_date)
    result = pd.DataFrame({
        "ticker": data["股票代码"].astype(str).str.extract(
            r"(\d{6})", expand=False
        ),
        "name": data["股票简称"].astype(str),
        "attention_rise": pd.to_numeric(data[attention], errors="coerce"),
        "float_a": pd.to_numeric(data[float_a], errors="coerce"),
        "total_shares": pd.to_numeric(data[total], errors="coerce"),
    }).dropna(subset=["ticker"])
    result = result.drop_duplicates("ticker").sort_values(
        ["attention_rise", "ticker"], ascending=[False, True], na_position="last"
    ).head(int(candidate_n)).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"empty attention candidates for {_day(signal_date).date()}")
    result.insert(0, "date", _day(signal_date).strftime("%Y-%m-%d"))
    return result[CANDIDATE_COLUMNS]


def fetch_candidates(signal_date, candidate_n: int = 100, access_token=None):
    raw = ths_http.smart_stock_picking(
        build_query(signal_date, candidate_n), access_token=access_token
    )
    return normalize_candidates(raw, signal_date, candidate_n=candidate_n)
```

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_ths_attention_combo.py alpha101/tests/test_ths_http.py strategies/tests/test_ths_heat.py -q
```

Expected: all tests pass; the existing iFinD wrapper and heat parser remain green.

- [ ] **Step 5: Inspect and commit Task 1**

Run:

```bash
git diff --check
git status --short
git add strategies/ths_attention_combo.py strategies/tests/test_ths_attention_combo.py
git commit -m "feat(strategies): add THS attention candidate query"
```

Expected: one commit containing only the new module/query slice and its tests.

---

### Task 2: Eligibility, factor math, and both deterministic selectors

**Files:**
- Modify: `strategies/ths_attention_combo.py`
- Modify: `strategies/tests/test_ths_attention_combo.py`

**Interfaces:**
- Consumes: Task 1 normalized candidates and a `close: pd.DataFrame` whose final row is T.
- Produces: `factor_frame(candidates, close, min_history=60) -> pd.DataFrame`, `select_weighted(factors, top_n=20) -> pd.DataFrame`, `select_funnel(factors, top_n=20) -> pd.DataFrame`, and `target_weights(selected, prices) -> dict[str, float]`.
- Factor columns are `date,ticker,name,attention_rise,float_a,total_shares,momentum_7d,float_ratio,attention_pct,momentum_pct,low_float_pct`.
- Selector output adds `strategy,rank,score` and preserves all factor columns.

- [ ] **Step 1: Add failing factor and selector tests**

Append to `strategies/tests/test_ths_attention_combo.py`:

```python
import numpy as np


def _close_for_factors():
    index = pd.bdate_range("2026-04-20", periods=60)
    return pd.DataFrame({
        "000001": np.linspace(10.0, 20.0, len(index)),
        "000002": np.linspace(20.0, 10.0, len(index)),
        "000003": np.linspace(8.0, 12.0, len(index)),
        "000004": np.linspace(5.0, 9.0, len(index)),
    }, index=index)


def _candidates_for_factors():
    return pd.DataFrame([
        ["2026-07-10", "000001", "甲", 100.0, 10.0, 100.0],
        ["2026-07-10", "000002", "乙", 90.0, 80.0, 100.0],
        ["2026-07-10", "000003", "*ST丙", 200.0, 5.0, 100.0],
        ["2026-07-10", "000004", "丁", 80.0, 20.0, 100.0],
    ], columns=combo.CANDIDATE_COLUMNS)


def test_factor_frame_filters_st_and_scores_all_directions():
    factors = combo.factor_frame(
        _candidates_for_factors(), _close_for_factors(), min_history=60
    )
    assert factors["ticker"].tolist() == ["000001", "000002", "000004"]
    by_ticker = factors.set_index("ticker")
    assert by_ticker.loc["000001", "momentum_7d"] > 0
    assert by_ticker.loc["000002", "momentum_7d"] < 0
    assert by_ticker.loc["000001", "attention_pct"] == 1.0
    assert by_ticker.loc["000001", "low_float_pct"] == 1.0
    assert by_ticker.loc["000002", "low_float_pct"] == pytest.approx(1 / 3)


def test_factor_frame_requires_current_t_minus_7_and_60_valid_closes():
    close = _close_for_factors()
    close.loc[close.index[-1], "000001"] = np.nan
    close.loc[close.index[-8], "000002"] = np.nan
    close.iloc[0, close.columns.get_loc("000004")] = -1.0
    factors = combo.factor_frame(_candidates_for_factors(), close, min_history=60)
    assert factors["ticker"].tolist() == []


def test_factor_frame_rejects_nonfinite_and_invalid_share_inputs():
    candidates = _candidates_for_factors().copy()
    candidates.loc[candidates["ticker"] == "000001", "attention_rise"] = np.nan
    candidates.loc[candidates["ticker"] == "000002", "float_a"] = 0.0
    candidates.loc[candidates["ticker"] == "000003", "name"] = "丙"
    candidates.loc[candidates["ticker"] == "000003", "total_shares"] = 0.0
    candidates.loc[candidates["ticker"] == "000004", "float_a"] = 120.0
    factors = combo.factor_frame(candidates, _close_for_factors(), min_history=60)
    assert factors.empty


def test_weighted_selector_uses_exact_weights_and_tie_breaks():
    factors = combo.factor_frame(
        _candidates_for_factors(), _close_for_factors(), min_history=60
    )
    selected = combo.select_weighted(factors, top_n=2)
    expected = (
        0.50 * selected.iloc[0]["attention_pct"]
        + 0.30 * selected.iloc[0]["momentum_pct"]
        + 0.20 * selected.iloc[0]["low_float_pct"]
    )
    assert selected.iloc[0]["score"] == pytest.approx(expected)
    assert selected["strategy"].unique().tolist() == ["ths_attention_weighted"]
    assert selected["rank"].tolist() == [1, 2]


def test_weighted_selector_breaks_score_ties_by_attention_then_ticker():
    factors = combo.factor_frame(
        _candidates_for_factors(), _close_for_factors(), min_history=60
    ).set_index("ticker").loc[["000002", "000001", "000004"]].reset_index()
    factors[["attention_pct", "momentum_pct", "low_float_pct"]] = 0.5
    factors["attention_rise"] = [10.0, 10.0, 9.0]
    selected = combo.select_weighted(factors, top_n=3)
    assert selected["ticker"].tolist() == ["000001", "000002", "000004"]


def test_funnel_keeps_positive_momentum_then_lowest_half_ceiling():
    factors = pd.DataFrame({
        "date": ["2026-07-15"] * 5,
        "ticker": ["A", "B", "C", "D", "E"],
        "name": list("ABCDE"),
        "attention_rise": [10.0, 50.0, 30.0, 40.0, 99.0],
        "float_a": [10.0, 20.0, 30.0, 40.0, 5.0],
        "total_shares": [100.0] * 5,
        "momentum_7d": [0.1, 0.2, 0.3, 0.4, -0.1],
        "float_ratio": [0.1, 0.2, 0.3, 0.4, 0.05],
        "attention_pct": [0.2, 1.0, 0.6, 0.8, 0.4],
        "momentum_pct": [0.4, 0.6, 0.8, 1.0, 0.2],
        "low_float_pct": [0.8, 0.6, 0.4, 0.2, 1.0],
    })
    selected = combo.select_funnel(factors, top_n=20)
    assert selected["ticker"].tolist() == ["B", "A"]
    assert selected["rank"].tolist() == [1, 2]
    assert selected["strategy"].unique().tolist() == ["ths_attention_funnel"]


def test_target_weights_rechecks_t_close_and_renormalizes():
    selected = pd.DataFrame({"ticker": ["A", "B", "C"]})
    prices = pd.Series({"A": 10.0, "B": np.nan, "C": 30.0})
    assert combo.target_weights(selected, prices) == {"A": 0.5, "C": 0.5}
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_ths_attention_combo.py -q
```

Expected: failures report missing `factor_frame`, `select_weighted`, `select_funnel`, and `target_weights`.

- [ ] **Step 3: Implement factor eligibility and percentiles**

Append these definitions to `strategies/ths_attention_combo.py`:

```python
FACTOR_COLUMNS = CANDIDATE_COLUMNS + [
    "momentum_7d", "float_ratio", "attention_pct",
    "momentum_pct", "low_float_pct",
]


def factor_frame(candidates, close: pd.DataFrame, min_history: int = 60):
    if len(close.index) < 8:
        return pd.DataFrame(columns=FACTOR_COLUMNS)
    base = candidates.drop_duplicates("ticker").copy()
    base = base[~base["name"].astype(str).str.match(
        r"^\*?ST", case=False, na=False
    )]
    tickers = base["ticker"].astype(str).tolist()
    history = close.reindex(columns=tickers).apply(
        pd.to_numeric, errors="coerce"
    )
    current = pd.to_numeric(history.iloc[-1], errors="coerce")
    past = pd.to_numeric(history.iloc[-8], errors="coerce")
    valid_history = history.where(np.isfinite(history) & (history > 0))
    valid_count = valid_history.notna().sum()
    base = base.set_index("ticker")
    base["momentum_7d"] = current / past - 1.0
    base["float_ratio"] = base["float_a"] / base["total_shares"]
    finite = np.isfinite(base[[
        "attention_rise", "float_a", "total_shares",
        "momentum_7d", "float_ratio",
    ]]).all(axis=1)
    valid = (
        finite
        & (valid_count.reindex(base.index).fillna(0) >= int(min_history))
        & (current.reindex(base.index) > 0)
        & (past.reindex(base.index) > 0)
        & (base["float_a"] > 0)
        & (base["total_shares"] > 0)
        & (base["float_ratio"] > 0)
        & (base["float_ratio"] <= 1)
    )
    result = base.loc[valid].reset_index()
    if result.empty:
        return pd.DataFrame(columns=FACTOR_COLUMNS)
    result["attention_pct"] = result["attention_rise"].rank(
        method="average", pct=True, ascending=True
    )
    result["momentum_pct"] = result["momentum_7d"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_float_pct"] = result["float_ratio"].rank(
        method="average", pct=True, ascending=False
    )
    return result.sort_values("ticker").reset_index(drop=True)[FACTOR_COLUMNS]
```

- [ ] **Step 4: Implement both selectors and targets**

Append:

```python
SELECTED_COLUMNS = ["strategy", "rank", *FACTOR_COLUMNS, "score"]


def _ranked(frame, strategy: str, top_n: int, score) -> pd.DataFrame:
    selected = frame.copy()
    selected["score"] = score.reindex(selected.index)
    selected = selected.sort_values(
        ["score", "attention_rise", "ticker"],
        ascending=[False, False, True],
    ).head(int(top_n)).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected.insert(0, "strategy", strategy)
    return selected[SELECTED_COLUMNS]


def select_weighted(factors, top_n: int = 20):
    score = (
        0.50 * factors["attention_pct"]
        + 0.30 * factors["momentum_pct"]
        + 0.20 * factors["low_float_pct"]
    )
    return _ranked(factors, "ths_attention_weighted", top_n, score)


def select_funnel(factors, top_n: int = 20):
    import math

    positive = factors[factors["momentum_7d"] > 0].sort_values(
        ["float_ratio", "ticker"], ascending=[True, True]
    )
    kept = positive.head(math.ceil(len(positive) / 2)).copy()
    return _ranked(
        kept, "ths_attention_funnel", top_n, kept["attention_rise"]
    )


def target_weights(selected, prices: pd.Series) -> dict[str, float]:
    tickers = selected["ticker"].astype(str).tolist()
    quoted = pd.to_numeric(prices.reindex(tickers), errors="coerce")
    valid = [ticker for ticker, price in quoted.items()
             if pd.notna(price) and np.isfinite(price) and price > 0]
    if not valid:
        return {}
    return {ticker: 1.0 / len(valid) for ticker in valid}
```

- [ ] **Step 5: Run focused and full strategy tests**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_ths_attention_combo.py -q
../../.venv/bin/python -m pytest strategies/tests -q
```

Expected: all tests pass, including the explicit ST, invalid-share, missing-price,
60-observation, formula, and deterministic-tie contracts.

- [ ] **Step 6: Inspect and commit Task 2**

Run:

```bash
git diff --check
git add strategies/ths_attention_combo.py strategies/tests/test_ths_attention_combo.py
git commit -m "feat(strategies): score THS attention combo factors"
```

Expected: one factor/selector commit with no paper-runner changes.

---

### Task 3: Paper orchestration, shared fetch, audit, and failure isolation

**Files:**
- Modify: `strategies/paper.py`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: `ths_attention_combo.fetch_candidates`, `factor_frame`, `select_weighted`, `select_funnel`, and `target_weights` from Tasks 1–2.
- Produces: `ATTENTION_PARAMS`, `ATTENTION_STRATEGIES`, `ATTENTION_SIGNAL_COLUMNS`, `_attention_error_row`, `append_attention_signals`, and `prepare_attention_targets(states, panel, fetch_panel, fetch_candidates=None)`.
- Extends `run(..., attention_fetch=None)` and `run_market(..., attention_fetch=None)` while keeping all existing call sites valid.

- [ ] **Step 1: Add failing shared-query and due-day integration tests**

In `strategies/tests/test_paper.py`, import the module and add a helper:

```python
from strategies import ths_attention_combo


def _attention_candidates(tickers=("A", "B")):
    return pd.DataFrame([
        {
            "date": "2024-04-19", "ticker": ticker, "name": ticker,
            "attention_rise": 100.0 - index,
            "float_a": 10.0 + index * 10.0, "total_shares": 100.0,
        }
        for index, ticker in enumerate(tickers)
    ], columns=ths_attention_combo.CANDIDATE_COLUMNS)


def test_run_market_shares_one_attention_query_between_two_accounts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    params = {"top_n": 20, "candidate_n": 100,
              "rebalance": 2, "min_history": 60}
    paper.init("weighted", 100000.0, strategy="ths_attention_weighted",
               market="a", params=params)
    paper.init("funnel", 100000.0, strategy="ths_attention_funnel",
               market="a", params=params)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    calls = []

    def fake_candidates(day, candidate_n=100):
        calls.append((day.strftime("%Y-%m-%d"), candidate_n))
        return _attention_candidates()

    panel = _a_panel(n=80)
    paper.run_market(
        "a", fetch=lambda codes, window_days=0: panel,
        attention_fetch=fake_candidates,
    )
    assert calls == [(panel["close"].index[-1].strftime("%Y-%m-%d"), 100)]
    assert paper.load_state("weighted")["pending_targets"]
    assert paper.load_state("funnel")["pending_targets"]


def test_non_due_attention_accounts_do_not_query(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    for account, strategy in [
        ("weighted", "ths_attention_weighted"),
        ("funnel", "ths_attention_funnel"),
    ]:
        paper.init(account, 100000.0, strategy=strategy, market="a")
        state = paper.load_state(account)
        state["days_since_rebalance"] = 0
        paper.save_state(account, state)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(n=80),
        attention_fetch=lambda *args, **kwargs: pytest.fail(
            "attention query called off-cycle"
        ),
    )
```

- [ ] **Step 2: Add failing failure-isolation and audit tests**

Append:

```python
def test_attention_query_failure_pauses_new_accounts_but_plain_continues(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    for account, strategy in [
        ("weighted", "ths_attention_weighted"),
        ("funnel", "ths_attention_funnel"),
    ]:
        paper.init(account, 100000.0, strategy=strategy, market="a")
    paper.init("plain", 100000.0, strategy="momentum", market="a",
               params={"top_n": 1, "lookback": 10,
                       "skip": 2, "rebalance": 5})
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])

    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(n=80),
        attention_fetch=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("temporary attention failure")
        ),
    )
    assert paper.load_state("weighted")["pending_targets"] is None
    assert paper.load_state("funnel")["pending_targets"] is None
    assert paper.load_state("plain")["last_run"] is not None
    audit = pd.read_csv(tmp_path / "ths_attention_combo_signals.csv")
    assert set(audit["account"]) == {"weighted", "funnel"}
    assert set(audit["status"]) == {"error"}


def test_attention_selector_failure_isolated_per_account(monkeypatch):
    states = {}
    for account, strategy in [
        ("weighted", "ths_attention_weighted"),
        ("funnel", "ths_attention_funnel"),
    ]:
        state = _state()
        state.update({"strategy": strategy, "params": {
            "top_n": 20, "candidate_n": 100,
            "rebalance": 2, "min_history": 60,
        }})
        states[account] = state
    monkeypatch.setattr(
        ths_attention_combo, "select_weighted",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("weighted calculation failed")
        ),
    )
    _, overrides, audit = paper.prepare_attention_targets(
        states, _a_panel(n=80),
        lambda *args, **kwargs: pytest.fail("no supplement expected"),
        fetch_candidates=lambda *args, **kwargs: _attention_candidates(),
    )
    assert overrides["weighted"] == {}
    assert overrides["funnel"]
    errors = [row for row in audit if row["status"] == "error"]
    assert {row["account"] for row in errors} == {"weighted"}


def test_attention_supplement_is_shared_partial_failure_audited_and_benchmark_pure():
    states = {}
    for account, strategy in [
        ("weighted", "ths_attention_weighted"),
        ("funnel", "ths_attention_funnel"),
    ]:
        state = _state()
        state.update({"strategy": strategy, "params": {
            "top_n": 20, "candidate_n": 100,
            "rebalance": 2, "min_history": 60,
        }})
        states[account] = state
    panel = _a_panel(n=80)
    panel["benchmark_close"] = panel["close"][["A", "B"]].copy()
    original_benchmark = panel["benchmark_close"].copy()
    calls = []

    def fake_panel(codes, window_days=0):
        calls.append(tuple(codes))
        if "BAD" in codes:
            raise RuntimeError("bad supplemental ticker")
        return _a_panel(tuple(codes), n=80)

    result, overrides, audit = paper.prepare_attention_targets(
        states, panel, fake_panel,
        fetch_candidates=lambda *args, **kwargs: _attention_candidates(
            ("A", "C", "BAD")
        ),
    )
    assert calls == [("BAD", "C"), ("BAD",), ("C",)]
    pd.testing.assert_frame_equal(result["benchmark_close"], original_benchmark)
    assert overrides["weighted"] and overrides["funnel"]
    assert all("BAD" not in weights for weights in overrides.values())
    errors = [row for row in audit if row["status"] == "error"]
    assert {row["account"] for row in errors} == {"weighted", "funnel"}
    assert all("BAD" in row["error"] for row in errors)


def test_attention_silent_supplement_omission_is_audited_for_both_accounts():
    states = {}
    for account, strategy in [
        ("weighted", "ths_attention_weighted"),
        ("funnel", "ths_attention_funnel"),
    ]:
        state = _state()
        state.update({"strategy": strategy, "params": {
            "top_n": 20, "candidate_n": 100,
            "rebalance": 2, "min_history": 60,
        }})
        states[account] = state

    _, overrides, audit = paper.prepare_attention_targets(
        states, _a_panel(n=80),
        lambda codes, window_days=0: _a_panel(("C",), n=80),
        fetch_candidates=lambda *args, **kwargs: _attention_candidates(
            ("A", "C", "BAD")
        ),
    )
    assert overrides["weighted"] and overrides["funnel"]
    errors = [row for row in audit if row["status"] == "error"]
    assert {row["account"] for row in errors} == {"weighted", "funnel"}
    assert all("BAD" in row["error"] for row in errors)


def test_failed_due_attention_target_stays_due_for_next_day():
    state = _state()
    state["strategy"] = "ths_attention_weighted"
    state["params"] = {
        "top_n": 20, "candidate_n": 100,
        "rebalance": 2, "min_history": 60,
    }
    state["days_since_rebalance"] = 2
    state, _, _ = paper.step(
        state, {"close": _close()}, target_override={}
    )
    assert state["pending_targets"] is None
    assert paper.rebalance_due(state) is True


@pytest.mark.parametrize(
    "strategy", ["ths_attention_weighted", "ths_attention_funnel"]
)
def test_attention_compute_targets_requires_prefetched_override(strategy):
    assert paper.compute_targets(
        strategy, _a_panel(n=80), {
            "top_n": 20, "candidate_n": 100,
            "rebalance": 2, "min_history": 60,
        }
    ) == {}


def test_append_attention_signals_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    rows = [{
        "date": "2026-07-15", "account": "weighted",
        "strategy": "ths_attention_weighted", "rank": 1,
        "ticker": "000001", "name": "甲", "attention_rise": 10.0,
        "momentum_7d": 0.05, "float_ratio": 0.2,
        "attention_pct": 1.0, "momentum_pct": 0.8,
        "low_float_pct": 0.9, "score": 0.92,
        "status": "ok", "error": "",
    }]
    paper.append_attention_signals(rows)
    paper.append_attention_signals(rows)
    assert len(pd.read_csv(tmp_path / "ths_attention_combo_signals.csv")) == 1
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_paper.py -k "attention" -q
```

Expected: failures for missing attention strategy registration, `attention_fetch` parameters, audit helper, and `prepare_attention_targets`.

- [ ] **Step 4: Register the strategies and audit schema**

Modify the top of `strategies/paper.py`:

```python
from strategies import momentum, ths_attention_combo, ths_heat

ATTENTION_PARAMS = {
    "top_n": 20, "candidate_n": 100, "rebalance": 2, "min_history": 60,
}
ATTENTION_STRATEGIES = {
    "ths_attention_weighted", "ths_attention_funnel",
}
NETWORK_STRATEGIES = HEAT_STRATEGIES | ATTENTION_STRATEGIES
ATTENTION_SIGNAL_COLUMNS = [
    "date", "account", "strategy", "rank", "ticker", "name",
    "attention_rise", "momentum_7d", "float_ratio",
    "attention_pct", "momentum_pct", "low_float_pct", "score",
    "status", "error",
]
```

Extend `DEFAULT_PARAMS` with both attention strategies pointing to `ATTENTION_PARAMS`, and change the early return in `compute_targets` from `HEAT_STRATEGIES` to `NETWORK_STRATEGIES`. Do not alter the existing heat constants.

After `_upsert_csv`, add:

```python
def _attention_error_row(date, account, strategy, error):
    return {
        "date": date, "account": account, "strategy": strategy,
        "rank": "", "ticker": "", "name": "", "attention_rise": "",
        "momentum_7d": "", "float_ratio": "", "attention_pct": "",
        "momentum_pct": "", "low_float_pct": "", "score": "",
        "status": "error", "error": sanitize_error(error),
    }


def append_attention_signals(rows, path=None):
    path = path or PAPER_DIR / "ths_attention_combo_signals.csv"
    _upsert_csv(
        path, rows, ["date", "account", "rank", "ticker", "status"],
        ATTENTION_SIGNAL_COLUMNS,
    )
```

- [ ] **Step 5: Implement shared attention preparation**

Add this orchestration after `prepare_heat_targets`:

```python
def prepare_attention_targets(
    states, panel, fetch_panel, fetch_candidates=None
):
    signal_date = panel["close"].index[-1]
    date_text = signal_date.strftime("%Y-%m-%d")
    due = {
        account: state for account, state in states.items()
        if state.get("strategy") in ATTENTION_STRATEGIES
        and state.get("last_run") != date_text and rebalance_due(state)
    }
    if not due:
        return panel, {}, []
    loader = fetch_candidates or ths_attention_combo.fetch_candidates
    candidate_n = max(state["params"]["candidate_n"] for state in due.values())
    audit = []
    try:
        candidates = loader(signal_date, candidate_n=candidate_n)
    except Exception as error:
        return panel, {account: {} for account in due}, [
            _attention_error_row(
                date_text, account, state["strategy"], error
            ) for account, state in due.items()
        ]

    missing = sorted(set(candidates["ticker"].astype(str))
                     - set(panel["close"].columns))
    quote_errors = {}
    if missing:
        panel, quote_errors, _ = _fetch_supplemental(
            panel, missing, fetch_panel
        )
    quote_issues = dict(quote_errors)
    latest = panel["close"].iloc[-1]
    for ticker in missing:
        price = pd.to_numeric(latest.get(ticker, np.nan), errors="coerce")
        if ticker not in panel["close"].columns:
            quote_issues.setdefault(
                ticker, RuntimeError("supplemental response omitted ticker")
            )
        elif not np.isfinite(price) or price <= 0:
            quote_issues.setdefault(
                ticker, RuntimeError("no valid signal-date close")
            )
    min_history = max(state["params"]["min_history"] for state in due.values())
    try:
        factors = ths_attention_combo.factor_frame(
            candidates, panel["close"], min_history=min_history
        )
    except Exception as error:
        return panel, {account: {} for account in due}, [
            _attention_error_row(
                date_text, account, state["strategy"], error
            ) for account, state in due.items()
        ]

    overrides = {}
    current_prices = panel["close"].iloc[-1]
    for account, state in due.items():
        strategy = state["strategy"]
        account_errors = []
        if quote_issues:
            account_errors.append("; ".join(
                f"{ticker}: {quote_issues[ticker]}"
                for ticker in sorted(quote_issues)
            ))
        try:
            selector = (
                ths_attention_combo.select_weighted
                if strategy == "ths_attention_weighted"
                else ths_attention_combo.select_funnel
            )
            selected = selector(factors, top_n=state["params"]["top_n"])
            weights = ths_attention_combo.target_weights(
                selected, current_prices
            )
            overrides[account] = weights
            if not weights:
                raise RuntimeError("no attention candidates have valid targets")
            used = selected[selected["ticker"].isin(weights)]
            audit.extend({
                **{column: row.get(column, "")
                   for column in ATTENTION_SIGNAL_COLUMNS},
                "date": date_text, "account": account,
                "status": "ok", "error": "",
            } for row in used.to_dict("records"))
        except Exception as error:
            overrides[account] = {}
            account_errors.append(str(error))
        if account_errors:
            audit.append(_attention_error_row(
                date_text, account, strategy,
                RuntimeError("; ".join(account_errors)),
            ))
    return panel, overrides, audit
```

During implementation, preserve two important semantics: an exception or silent
omission in the shared supplemental fetch creates one sanitized account-level error
summary for each due comparison account while valid candidates still proceed; every
selected target must have a valid current close. Never forward-fill eligibility or
target prices.

- [ ] **Step 6: Wire single-account and market runners**

Change signatures to:

```python
def run(account="us_momentum", fetch=None, heat_fetch=None,
        attention_fetch=None):
```

and:

```python
def run_market(market, fetch=None, heat_fetch=None, attention_fetch=None):
```

In each A-market path, keep heat preparation unchanged, then run attention preparation against the possibly supplemented panel:

```python
panel, attention_overrides, attention_audit = prepare_attention_targets(
    states, panel, fetch or fetch_a_panel,
    fetch_candidates=attention_fetch,
)
overrides.update(attention_overrides)
append_attention_signals(attention_audit)
```

For `run`, pass `{account: state}` instead of `states`. Do not call attention preparation outside market `a`. Keep account stepping and aggregate failure handling unchanged.

- [ ] **Step 7: Run paper and full tests**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_paper.py -q
../../.venv/bin/python -m pytest alpha101/tests strategies/tests -q
```

Expected: all tests pass, including all pre-existing heat, stale-price, benchmark-purity, crash-idempotency, and sanitizer tests.

- [ ] **Step 8: Inspect and commit Task 3**

Run:

```bash
git diff --check
git add strategies/paper.py strategies/tests/test_paper.py
git commit -m "feat(paper): run THS attention combo accounts"
```

Expected: one orchestration commit with no account/UI artifacts.

---

### Task 4: Seed two accounts and expose seven strategies

**Files:**
- Modify: `paper/accounts.json`
- Create: `paper/a_ths_attention_weighted/state.json`
- Create: `paper/a_ths_attention_weighted/nav.csv`
- Create: `paper/a_ths_attention_weighted/orders.csv`
- Create: `paper/a_ths_attention_funnel/state.json`
- Create: `paper/a_ths_attention_funnel/nav.csv`
- Create: `paper/a_ths_attention_funnel/orders.csv`
- Create: `paper/ths_attention_combo_signals.csv`
- Modify: `paper.html`
- Modify: `index.html`
- Modify: `strategies/README.md`
- Verify: `.github/workflows/paper-a.yml`
- Modify: `strategies/tests/test_paper.py`

**Interfaces:**
- Consumes: Task 3 strategy names and exact parameter dictionary.
- Produces: seven manifest entries, two zero-history stores, audit header, two dashboard descriptions, and documented workflow coverage.

- [ ] **Step 1: Add a failing repository-artifact test**

Append to `strategies/tests/test_paper.py`:

```python
def test_repository_has_two_attention_combo_accounts_and_seven_strategy_ui():
    root = paper.Path(__file__).resolve().parents[2]
    entries = json.loads((root / "paper" / "accounts.json").read_text())
    assert len(entries) == 7
    by_account = {entry["account"]: entry for entry in entries}
    expected = {
        "a_ths_attention_weighted": (
            "A股 注意力加权组合", "ths_attention_weighted"
        ),
        "a_ths_attention_funnel": (
            "A股 注意力逐层筛选", "ths_attention_funnel"
        ),
    }
    for account, (title, strategy) in expected.items():
        assert by_account[account] == {
            "account": account, "title": title, "currency": "¥",
        }
        state = json.loads((root / "paper" / account / "state.json").read_text())
        assert state["capital"] == state["cash"] == 100000.0
        assert state["strategy"] == strategy and state["market"] == "a"
        assert state["params"] == {
            "top_n": 20, "candidate_n": 100,
            "rebalance": 2, "min_history": 60,
        }
        assert (root / "paper" / account / "nav.csv").read_text() == (
            "date,nav,cash,positions_value,bench_nav\n"
        )
        assert (root / "paper" / account / "orders.csv").read_text() == (
            "date,ticker,side,shares,price,value,cost\n"
        )
    audit = pd.read_csv(root / "paper" / "ths_attention_combo_signals.csv")
    assert audit.columns.tolist() == paper.ATTENTION_SIGNAL_COLUMNS
    html = (root / "paper.html").read_text()
    index = (root / "index.html").read_text()
    assert "模拟盘 · 七策略" in html and "模拟盘 · 七策略 →" in index
    assert "a_ths_attention_weighted" in html
    assert "a_ths_attention_funnel" in html
    workflow = (root / ".github/workflows/paper-a.yml").read_text()
    assert "THS_HTTP_REFRESH_TOKEN" in workflow
    assert "run-market --market a" in workflow
    assert "git add paper/" in workflow
```

- [ ] **Step 2: Run the artifact test and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest strategies/tests/test_paper.py::test_repository_has_two_attention_combo_accounts_and_seven_strategy_ui -q
```

Expected: failure because the manifest has five accounts and the new stores do not exist.

- [ ] **Step 3: Append the manifest entries and seed states**

Append to `paper/accounts.json`:

```json
{
 "account": "a_ths_attention_weighted",
 "title": "A股 注意力加权组合",
 "currency": "¥"
},
{
 "account": "a_ths_attention_funnel",
 "title": "A股 注意力逐层筛选",
 "currency": "¥"
}
```

Create `paper/a_ths_attention_weighted/state.json`:

```json
{
 "account": "a_ths_attention_weighted",
 "capital": 100000.0,
 "cash": 100000.0,
 "positions": {},
 "pending_targets": null,
 "days_since_rebalance": null,
 "bench_nav": 100000.0,
 "last_run": null,
 "strategy": "ths_attention_weighted",
 "market": "a",
 "params": {"top_n": 20, "candidate_n": 100, "rebalance": 2, "min_history": 60}
}
```

Create `paper/a_ths_attention_funnel/state.json` with the same values except:

```json
{
 "account": "a_ths_attention_funnel",
 "capital": 100000.0,
 "cash": 100000.0,
 "positions": {},
 "pending_targets": null,
 "days_since_rebalance": null,
 "bench_nav": 100000.0,
 "last_run": null,
 "strategy": "ths_attention_funnel",
 "market": "a",
 "params": {"top_n": 20, "candidate_n": 100, "rebalance": 2, "min_history": 60}
}
```

For each account create `nav.csv` and `orders.csv` with exactly:

```csv
date,nav,cash,positions_value,bench_nav
```

and:

```csv
date,ticker,side,shares,price,value,cost
```

Create `paper/ths_attention_combo_signals.csv` with exactly:

```csv
date,account,strategy,rank,ticker,name,attention_rise,momentum_7d,float_ratio,attention_pct,momentum_pct,low_float_pct,score,status,error
```

- [ ] **Step 4: Update dashboard and homepage copy**

In `paper.html`, change the document title to:

```html
<title>模拟盘 · 七策略</title>
```

Add these `DESC` entries:

```javascript
a_ths_attention_weighted: '注意力上升50% + 7日动量30% + 低流通比例20% · top 20等权 · ' +
  '每2交易日调仓 · 虚拟资金 ¥100,000 · 注:组合权重未经历史优化,仅做前向检验',
a_ths_attention_funnel: '注意力上升前100 → 7日动量为正 → 低流通比例50% → top 20等权 · ' +
  '每2交易日调仓 · 虚拟资金 ¥100,000 · 注:硬门槛可能导致持仓不足20只',
```

In `index.html`, change only the link copy from `模拟盘 · 五策略 →` to `模拟盘 · 七策略 →`.

- [ ] **Step 5: Document the two strategies**

Append a section to `strategies/README.md` containing the exact formulas, Top 100 candidate pool, ST/60-day/current-and-T-7 filters, Top 20 equal weighting, two-day cadence, T/T+1 timing, 20bp cost, audit path, failure/retry behavior, and this warning:

```text
注意力、短期动量和低流通比例均可能放大拥挤与波动；两个账户只比较组合方法，
不代表三因子或任一权重已经通过历史显著性验证。
```

Do not modify `.github/workflows/paper-a.yml`; its existing `git add paper/` already stages both new account directories and the audit CSV.

- [ ] **Step 6: Validate artifacts and run tests**

Run:

```bash
python3 -m json.tool paper/accounts.json >/dev/null
python3 -m json.tool paper/a_ths_attention_weighted/state.json >/dev/null
python3 -m json.tool paper/a_ths_attention_funnel/state.json >/dev/null
../../.venv/bin/python -c "import pandas as pd; pd.read_csv('paper/ths_attention_combo_signals.csv'); print('attention audit ok')"
../../.venv/bin/python -m pytest strategies/tests/test_paper.py -q
../../.venv/bin/python -m pytest alpha101/tests strategies/tests -q
```

Expected: JSON commands exit 0, `attention audit ok` prints, and all tests pass.

- [ ] **Step 7: Inspect and commit Task 4**

Run:

```bash
git diff --check
git status --short
git add paper/accounts.json paper/a_ths_attention_weighted \
  paper/a_ths_attention_funnel paper/ths_attention_combo_signals.csv \
  paper.html index.html strategies/README.md strategies/tests/test_paper.py
git commit -m "feat(paper): add THS attention comparison accounts"
```

Expected: one account/UI/docs commit; `.github/workflows/paper-a.yml` is unchanged.

---

### Task 5: Full verification and official read-only smoke test

**Files:**
- Verify only; no expected source changes.

**Interfaces:**
- Consumes all Task 1–4 deliverables.
- Produces offline test, live-schema, target-weight, credential-safety, history, and clean-worktree evidence.

- [ ] **Step 1: Run the complete offline suite**

Run:

```bash
../../.venv/bin/python -m pytest alpha101/tests strategies/tests -q
../../.venv/bin/python -m compileall -q alpha101 strategies
git diff --check
```

Expected: zero test failures, compile exit 0, and no diff errors.

- [ ] **Step 2: Validate repository artifacts**

Run:

```bash
python3 -m json.tool paper/accounts.json >/dev/null
python3 -m json.tool paper/a_ths_attention_weighted/state.json >/dev/null
python3 -m json.tool paper/a_ths_attention_funnel/state.json >/dev/null
../../.venv/bin/python -c "import pandas as pd; f=pd.read_csv('paper/ths_attention_combo_signals.csv'); print(f.columns.tolist(), len(f))"
```

Expected: all JSON parses; the CSV prints the 15-column attention audit schema and zero seed rows.

- [ ] **Step 3: Run one live read-only combined query and both selectors**

With network approval and without printing credentials, run:

```bash
set -a; source ../../.env; set +a; ../../.venv/bin/python -c "
import pandas as pd
from strategies import paper, ths_attention_combo as combo
seed = paper.fetch_a_panel(['000001', '600000'], window_days=120)
day = seed['close'].index[-1]
candidates = combo.fetch_candidates(day, candidate_n=100)
supplement = paper.fetch_a_panel(candidates['ticker'].tolist(), window_days=120)
panel = paper._merge_panel(seed, supplement)
factors = combo.factor_frame(candidates, panel['close'], min_history=60)
for selector in (combo.select_weighted, combo.select_funnel):
    selected = selector(factors, top_n=20)
    weights = combo.target_weights(selected, panel['close'].iloc[-1])
    print(selector.__name__, len(candidates), len(factors), len(selected),
          round(sum(weights.values()), 12), selected.columns.tolist())
"
```

Expected: candidate columns are normalized, each selector returns at least one row for the live sample, each nonempty target weight sum is exactly 1 within floating tolerance, and no paper state file changes.

- [ ] **Step 4: Confirm credential safety and clean history**

Run:

```bash
rg -n "THS_HTTP_REFRESH_TOKEN=|refresh_token.*[A-Za-z0-9]{20}" \
  alpha101 strategies paper docs .github || true
git status --short
git log --oneline -8
```

Expected: matches contain only variable names, scan commands, or documented placeholders; `.env` is absent; Task 1–4 commits are present; the worktree is clean.

- [ ] **Step 5: Report final evidence**

Record the exact full/focused test counts, live response columns and row counts, selector/weight results, account IDs and parameters, audit schema, credential-scan result, HEAD hash, and worktree status. Do not write live results into account state or claim historical profitability.
