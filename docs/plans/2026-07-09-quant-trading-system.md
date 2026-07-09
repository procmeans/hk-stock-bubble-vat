# Quant Trading System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a repeatable A-share quantitative workflow that compares multiple models, turns scores into buy/sell signals, backtests portfolios, and reports drawdowns and performance.

**Architecture:** Keep the system pandas-first and file-based inside `alpha101/`. Reuse the existing iFinD historical panel, Alpha101 formulas, universe filters, and compose helpers. Add thin modules for model scoring, trade signal generation, portfolio backtesting, and reporting.

**Tech Stack:** Python 3.9, pandas, numpy, pytest, existing `alpha101` modules, CSV outputs under `output/`.

---

### Task 1: Model Registry And Score Schema

**Files:**
- Create: `alpha101/models.py`
- Test: `alpha101/tests/test_models.py`

**Step 1: Write the failing tests**

Create tests for:

```python
def test_model_registry_lists_initial_models():
    from alpha101 import models

    assert "alpha101_equal_weight" in models.available_models()
    assert "alpha101_single_101" in models.available_models()


def test_scores_to_long_frame_has_stable_schema():
    import pandas as pd
    from alpha101.models import scores_to_long_frame

    score = pd.DataFrame(
        {"000001": [1.0], "000002": [2.0]},
        index=[pd.Timestamp("2026-07-08")],
    )
    names = {"000001": "平安银行", "000002": "万科A"}

    result = scores_to_long_frame(score, "demo", names)

    assert list(result.columns) == [
        "date", "code", "name", "model", "score", "rank", "eligible", "reason"
    ]
    assert result.iloc[0]["code"] == "000002"
    assert result.iloc[0]["rank"] == 1
```

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest alpha101/tests/test_models.py -v
```

Expected: FAIL because `alpha101.models` does not exist.

**Step 3: Implement minimal module**

Implement:

- `available_models() -> list[str]`
- `scores_to_long_frame(score, model, names=None, eligible=True, reason="")`
- `score_alpha101_equal_weight(panel, mask=None)`
- `score_alpha101_single_101(panel, mask=None)`

Use existing:

- `alpha101.alphas.compute_all`
- `alpha101.compose.composite`
- `alpha101.alphas.alpha_101`

**Step 4: Run tests**

Run:

```bash
python3 -m pytest alpha101/tests/test_models.py alpha101/tests/test_alphas.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add alpha101/models.py alpha101/tests/test_models.py
git commit -m "feat: add alpha model registry"
```

---

### Task 2: Score CLI For Multi-Model Comparison

**Files:**
- Modify: `alpha101/models.py`
- Test: `alpha101/tests/test_models.py`

**Step 1: Write failing CLI-oriented test**

Test `run_scores(panel, models, names, output)` with a synthetic score function or small panel. It should write a CSV with multiple model names.

**Step 2: Run failure**

```bash
python3 -m pytest alpha101/tests/test_models.py::test_run_scores_writes_multiple_models -v
```

Expected: FAIL because `run_scores` does not exist.

**Step 3: Implement**

Add:

- `run_scores(panel, model_names, names=None, output=Path(...))`
- CLI `python3 -m alpha101.models score --cache alpha101/cache/ths_panel.pkl --universe data/a-2026-07-07.json --models alpha101_equal_weight,alpha101_single_101`

Output:

```text
output/model_scores/latest_scores.csv
```

**Step 4: Verify**

Run:

```bash
python3 -m pytest alpha101/tests/test_models.py -v
python3 -m alpha101.models score --cache alpha101/cache/ths_panel.pkl --universe data/a-2026-07-07.json --models alpha101_equal_weight,alpha101_single_101 --output output/model_scores/latest_scores.csv
```

Expected: tests PASS and CSV exists.

**Step 5: Commit**

```bash
git add alpha101/models.py alpha101/tests/test_models.py output/model_scores/.gitkeep
git commit -m "feat: add multi-model score output"
```

---

### Task 3: Trade Signal Rules

**Files:**
- Create: `alpha101/trade.py`
- Test: `alpha101/tests/test_trade.py`

**Step 1: Write failing tests**

Cover:

- Buy top N new names.
- Hold existing names inside sell buffer.
- Sell names outside sell buffer.
- Equal target weights capped by max position.

Example:

```python
def test_generate_signals_uses_hold_buffer():
    from alpha101.trade import generate_signals
    scores = ...
    current = {"000003": 0.05}
    result = generate_signals(scores, current, buy_top_n=2, sell_below_rank=4)
    assert ...
```

**Step 2: Run failure**

```bash
python3 -m pytest alpha101/tests/test_trade.py -v
```

Expected: FAIL because `alpha101.trade` does not exist.

**Step 3: Implement**

Add:

- `generate_signals(scores, current_positions=None, buy_top_n=20, sell_below_rank=60, max_weight=0.05)`
- CLI `python3 -m alpha101.trade signal --scores output/model_scores/latest_scores.csv --model alpha101_equal_weight`

Output schema:

```text
date, model, code, name, action, current_weight, target_weight, score, rank, reason
```

**Step 4: Verify**

```bash
python3 -m pytest alpha101/tests/test_trade.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add alpha101/trade.py alpha101/tests/test_trade.py
git commit -m "feat: add trade signal rules"
```

---

### Task 4: Portfolio Backtest Engine

**Files:**
- Create: `alpha101/portfolio_backtest.py`
- Test: `alpha101/tests/test_portfolio_backtest.py`

**Step 1: Write failing tests**

Cover:

- One-period return with equal weights.
- Transaction cost drag from turnover.
- Missing price data keeps position value unchanged or drops position according to documented rule.
- Weekly rebalance dates.

**Step 2: Run failure**

```bash
python3 -m pytest alpha101/tests/test_portfolio_backtest.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement**

Add:

- `rebalance_dates(index, frequency="W-FRI")`
- `simulate(close, score_table, model, top_n=20, sell_below_rank=60, cost_bps=20)`
- output tables: equity curve, positions, orders.

Use close-to-close returns and costs:

```text
cost = sum(abs(target_weight - current_weight)) * cost_bps / 10000
```

**Step 4: Verify**

```bash
python3 -m pytest alpha101/tests/test_portfolio_backtest.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add alpha101/portfolio_backtest.py alpha101/tests/test_portfolio_backtest.py
git commit -m "feat: add portfolio backtest engine"
```

---

### Task 5: Drawdown And Performance Report

**Files:**
- Create: `alpha101/portfolio_report.py`
- Test: `alpha101/tests/test_portfolio_report.py`

**Step 1: Write failing tests**

Cover:

- Max drawdown calculation.
- Annualized return and volatility.
- Sharpe.
- Yearly return table.

**Step 2: Run failure**

```bash
python3 -m pytest alpha101/tests/test_portfolio_report.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement**

Add:

- `max_drawdown(equity)`
- `performance_summary(equity, periods_per_year=252)`
- `yearly_returns(equity)`
- CLI to build `output/backtests/model_compare.csv`.

**Step 4: Verify**

```bash
python3 -m pytest alpha101/tests/test_portfolio_report.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add alpha101/portfolio_report.py alpha101/tests/test_portfolio_report.py
git commit -m "feat: add portfolio performance report"
```

---

### Task 6: End-To-End Research Command

**Files:**
- Create: `alpha101/research.py`
- Test: `alpha101/tests/test_research.py`
- Modify: `alpha101/README.md`

**Step 1: Write failing smoke test**

Use a tiny synthetic panel and assert the pipeline creates:

- score table
- signal table
- equity curve
- summary table

**Step 2: Run failure**

```bash
python3 -m pytest alpha101/tests/test_research.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement**

Add command:

```bash
python3 -m alpha101.research run \
  --cache alpha101/cache/ths_panel.pkl \
  --universe data/a-2026-07-07.json \
  --models alpha101_equal_weight,alpha101_single_101 \
  --top-n 20 \
  --sell-below-rank 60 \
  --cost-bps 20
```

**Step 4: Verify**

```bash
python3 -m pytest alpha101/tests -v
python3 -m alpha101.research run --cache alpha101/cache/ths_panel.pkl --universe data/a-2026-07-07.json --models alpha101_equal_weight,alpha101_single_101
```

Expected: tests PASS, output files generated.

**Step 5: Commit**

```bash
git add alpha101/research.py alpha101/tests/test_research.py alpha101/README.md
git commit -m "feat: add end-to-end research pipeline"
```
