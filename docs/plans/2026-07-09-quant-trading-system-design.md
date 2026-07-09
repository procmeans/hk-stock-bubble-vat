# Quant Trading System Design

## Goal

Turn the current A-share Alpha101 scoring scripts into a repeatable research loop:
data update, multi-model scoring, trade signal generation, portfolio backtest,
risk review, and model comparison.

The first production target is a research-grade weekly rebalancing system, not
direct broker execution.

## Current State

The repository already supports:

- Static market snapshots for HK, A-share, and US valuation visualization.
- An `alpha101/` framework with 100 implemented WorldQuant Alpha101 formulas
  (`alpha_56` is not implemented).
- iFinD HTTP integration for today quotation and historical A-share OHLCV data.
- A full iFinD historical panel cache at `alpha101/cache/ths_panel.pkl`.
- A full Alpha101 composite output at `output/ths_full_alpha101_picks.csv`.

The repository does not yet support:

- Comparing multiple models under one schema.
- Translating scores into buy/sell/hold decisions.
- Portfolio-level backtesting with transaction costs and turnover.
- Drawdown, yearly return, and trade-level review.
- A daily/weekly optimization workflow.

## Architecture

The system should stay file-based and pandas-first. Each stage writes a stable
CSV artifact so results can be inspected, diffed, and reused without a database.

Pipeline:

1. `alpha101.ths_history fetch`: update iFinD historical OHLCV cache.
2. `alpha101.models score`: compute model scores for a date range.
3. `alpha101.trade signal`: convert scores into target holdings and orders.
4. `alpha101.portfolio_backtest run`: simulate rebalancing with costs.
5. `alpha101.portfolio_report build`: produce review tables and charts.

## Model Layer

Add a model registry where each model returns the same long-form score table:

```text
date, code, name, model, score, rank, eligible, reason
```

Initial models:

- `alpha101_equal_weight`: existing 100 Alpha101 factors, winsorized z-score,
  equal-weight composite.
- `alpha101_ic_weighted`: same factors, weighted by trailing IC/ICIR.
- `alpha101_single_101`: WQ Alpha101 factor 101 only, retained as a baseline.
- `value_quality`: valuation plus profitability and growth fields from the
  A-share snapshot when available.
- `hybrid`: alpha101 score plus value/quality score with liquidity and industry
  constraints.

## Trading Rules

The first trading system uses simple, auditable rules:

- Rebalance weekly by default, using signals from the latest available close.
- Buy the model top N, default `20`.
- Hold existing positions until they fall outside a sell buffer, default `60`.
- Equal weight, with maximum single-stock weight of `5%`.
- Exclude ST stocks, delisted names, suspended stocks, newly listed names with
  fewer than 60 valid trading days, and low-liquidity names.
- Apply transaction cost and slippage, default one-way `0.20%`.

The output schema for signals:

```text
date, model, code, name, action, current_weight, target_weight, score, rank, reason
```

## Backtest

The first backtest engine should be deterministic and conservative:

- Use close-to-close returns.
- Rebalance at the next available close after signal date.
- Enforce no-trade when a stock is missing price data.
- Charge costs on weight turnover.
- Track cash implicitly through normalized portfolio weights.

Output artifacts:

- `output/backtests/<model>/equity_curve.csv`
- `output/backtests/<model>/positions.csv`
- `output/backtests/<model>/orders.csv`
- `output/backtests/model_compare.csv`

## Risk And Review

Report the metrics that determine whether a model is tradable:

- Cumulative return, annualized return, volatility, Sharpe.
- Maximum drawdown, drawdown start, trough, recovery date.
- Yearly returns.
- Monthly returns.
- Win rate by rebalance period.
- Average turnover and estimated cost drag.
- Benchmark excess return.
- Worst trades and best trades.
- Industry concentration when industry data is present.

## Error Handling

- Data fetching should support resume and write incremental cache files.
- Model scoring should log failed factors and continue when possible.
- Backtests should fail fast if required price fields are missing.
- Reports should include warnings for missing benchmark data, missing industry
  data, or incomplete latest trading dates.

## Testing

Use pytest and small synthetic panels:

- Model registry returns consistent score schemas.
- Signal rules produce buy, sell, hold, and buffer behavior.
- Backtest applies returns and transaction costs correctly.
- Drawdown metrics identify peak, trough, and recovery.
- End-to-end smoke test runs synthetic scores through signals and backtest.

## Milestones

1. Model registry and multi-model score output.
2. Trade signal generation.
3. Portfolio backtest and metrics.
4. Model comparison report.
5. Optional HTML report and scheduled workflow.
