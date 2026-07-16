import numpy as np
import pandas as pd
import pytest

from intraday.portfolio import (
    build_benchmark_targets,
    build_targets,
    is_one_price_limit,
    portfolio_metrics,
    simulate,
)


def _raw_ohlc(opens):
    long = (
        opens.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="code", value_name="open")
        .set_index(["date", "code"])
        .sort_index()
    )
    long["high"] = long["open"] * 1.01
    long["low"] = long["open"] * 0.99
    long["close"] = long["open"]
    return long


def test_one_price_limit_direction_rounding_and_unverifiable_rows():
    up = pd.Series({"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0})
    down = pd.Series({"open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0})
    rounded_up = pd.Series({
        "open": 10.996,
        "high": 11.004,
        "low": 11.001,
        "close": 10.999,
    })
    suspended = pd.Series({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0})

    assert is_one_price_limit(up, 10.0) == (True, False)
    assert is_one_price_limit(down, 10.0) == (False, True)
    assert is_one_price_limit(rounded_up, 10.0) == (True, False)
    assert is_one_price_limit(suspended, 10.0) == (True, True)
    assert is_one_price_limit(up, np.nan) == (True, True)


@pytest.mark.parametrize(
    ("price", "expected"),
    [(10.45, (True, False)), (9.55, (False, True))],
)
def test_one_price_limit_includes_exact_four_point_five_percent_boundary(
    price,
    expected,
):
    row = pd.Series({"open": price, "high": price, "low": price, "close": price})

    assert is_one_price_limit(row, 10.0) == expected


@pytest.mark.parametrize("price", [10.44, 9.56])
def test_one_price_limit_below_boundary_remains_bidirectionally_blocked(price):
    row = pd.Series({"open": price, "high": price, "low": price, "close": price})

    assert is_one_price_limit(row, 10.0) == (True, True)


@pytest.mark.parametrize("previous_close", [0.0, -1.0, None, "bad"])
def test_one_price_limit_blocks_both_sides_for_invalid_previous_close(
    previous_close,
):
    row = pd.Series({"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0})

    assert is_one_price_limit(row, previous_close) == (True, True)


def test_signal_trades_next_open_and_charges_exact_cost():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame({"A": [10.0, 10.0, 11.0, 11.0]}, index=dates)
    targets = {dates[0]: pd.Series({"A": 1.0})}

    result = simulate(targets, opens, _raw_ohlc(opens), cost_bps=20)

    trade = result["trades"].iloc[0]
    assert trade["signal_date"] == dates[0]
    assert trade["date"] == dates[1]
    assert not result["trades"]["date"].eq(dates[0]).any()
    assert trade["cost"] == pytest.approx(trade["notional"] * 0.002)
    assert result["nav"].loc[dates[0]] == 1.0
    assert result["nav"].loc[dates[1]] == pytest.approx(1.0 - trade["cost"])


def test_blocked_buy_stays_cash():
    dates = pd.bdate_range("2026-01-01", periods=5)
    opens = pd.DataFrame({"A": [10.0, 11.0, 10.0, 9.0, 9.0]}, index=dates)
    raw = _raw_ohlc(opens)
    raw.loc[(dates[1], "A"), ["high", "low", "close"]] = 11.0

    result = simulate(
        {dates[0]: pd.Series({"A": 1.0})},
        opens,
        raw,
        cost_bps=0,
    )

    assert result["trades"].empty
    assert result["nav"].eq(1.0).all()


def test_blocked_sell_keeps_existing_position():
    dates = pd.bdate_range("2026-01-01", periods=5)
    opens = pd.DataFrame({"A": [10.0, 10.0, 9.0, 9.0, 9.0]}, index=dates)
    raw = _raw_ohlc(opens)
    raw.loc[(dates[2], "A"), ["high", "low", "close"]] = 9.0
    targets = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[1]: pd.Series(dtype=float),
    }

    result = simulate(targets, opens, raw, cost_bps=0)

    execution_day = dates[2]
    sells = result["trades"].query(
        "date == @execution_day and side == 'sell'"
    )
    assert sells.empty
    assert result["nav"].loc[dates[3]] == pytest.approx(0.9)


@pytest.mark.parametrize("case", ["missing-row", "nan-ohlc", "missing-previous"])
def test_unverifiable_raw_data_blocks_buy(case):
    dates = pd.bdate_range("2026-01-01", periods=3)
    opens = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    raw = _raw_ohlc(opens)
    if case == "missing-row":
        raw = raw.drop(index=(dates[1], "A"))
    elif case == "nan-ohlc":
        raw.loc[(dates[1], "A"), "high"] = np.nan
    else:
        raw = raw.drop(index=(dates[0], "A"))

    result = simulate(
        {dates[0]: pd.Series({"A": 1.0})},
        opens,
        raw,
        cost_bps=0,
    )

    assert result["trades"].empty
    assert result["nav"].eq(1.0).all()


def test_missing_execution_raw_row_blocks_sell_too():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=dates)
    raw = _raw_ohlc(opens).drop(index=(dates[2], "A"))
    targets = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[1]: pd.Series(dtype=float),
    }

    result = simulate(targets, opens, raw, cost_bps=0)

    execution_trades = result["trades"].loc[
        result["trades"]["date"].eq(dates[2])
    ]
    assert execution_trades.empty
    assert result["nav"].loc[dates[3]] == 1.0


def test_up_limit_allows_sell_and_down_limit_allows_buy():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame(
        {"A": [10.0, 10.0, 11.0, 11.0], "B": [10.0, 10.0, 9.0, 9.0]},
        index=dates,
    )
    raw = _raw_ohlc(opens)
    raw.loc[(dates[2], "A"), ["high", "low", "close"]] = 11.0
    raw.loc[(dates[2], "B"), ["high", "low", "close"]] = 9.0
    targets = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[1]: pd.Series({"B": 1.0}),
    }

    result = simulate(targets, opens, raw, cost_bps=0)

    execution = result["trades"].loc[
        result["trades"]["date"].eq(dates[2])
    ]
    assert execution[["code", "side"]].to_dict("records") == [
        {"code": "A", "side": "sell"},
        {"code": "B", "side": "buy"},
    ]


def test_missing_adjusted_open_ffills_holding_value_but_does_not_trade():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame({"A": [10.0, 10.0, np.nan, 12.0]}, index=dates)
    raw = _raw_ohlc(opens.fillna(10.0))
    targets = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[1]: pd.Series(dtype=float),
    }

    result = simulate(targets, opens, raw, cost_bps=0)

    assert result["trades"].loc[
        result["trades"]["date"].eq(dates[2])
    ].empty
    assert result["nav"].loc[dates[2]] == 1.0
    assert result["nav"].loc[dates[3]] == pytest.approx(1.2)


def test_blocked_sell_leaves_only_cash_available_for_scaled_buys():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    raw = _raw_ohlc(opens)
    raw.loc[(dates[2], "A"), ["open", "high", "low", "close"]] = 9.0
    targets = {
        dates[0]: pd.Series({"A": 0.5}),
        dates[1]: pd.Series({"B": 1.0}),
    }

    result = simulate(targets, opens, raw, cost_bps=0)

    execution = result["trades"].loc[
        result["trades"]["date"].eq(dates[2])
    ]
    assert not execution["side"].eq("sell").any()
    buy = execution.query("code == 'B' and side == 'buy'").iloc[0]
    assert buy["notional"] == pytest.approx(0.5)
    assert result["turnover"].loc[dates[2]] == pytest.approx(0.5)


def test_rebalance_sells_before_buys_and_charges_both_sides_exactly():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    raw = _raw_ohlc(opens)
    targets = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[1]: pd.Series({"B": 1.0}),
    }

    result = simulate(targets, opens, raw, cost_bps=20)

    execution = result["trades"].loc[
        result["trades"]["date"].eq(dates[2])
    ]
    assert execution["side"].tolist() == ["sell", "buy"]
    assert execution["cost"].to_numpy() == pytest.approx(
        execution["notional"].to_numpy() * 0.002
    )
    assert result["cost"].loc[dates[2]] == pytest.approx(
        execution["cost"].sum()
    )
    assert result["turnover"].loc[dates[2]] == pytest.approx(
        execution["notional"].sum() / result["nav"].loc[dates[1]]
    )


def test_simulate_returns_stable_empty_trade_schema():
    dates = pd.bdate_range("2026-01-01", periods=3)
    opens = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)

    result = simulate({}, opens, _raw_ohlc(opens), cost_bps=20)

    assert set(result) == {"nav", "returns", "turnover", "cost", "trades"}
    assert result["trades"].empty
    assert result["trades"].columns.tolist() == [
        "signal_date",
        "date",
        "code",
        "side",
        "shares",
        "price",
        "notional",
        "cost",
        "status",
    ]
    assert result["nav"].eq(1.0).all()
    assert result["returns"].eq(0.0).all()
    assert result["turnover"].eq(0.0).all()
    assert result["cost"].eq(0.0).all()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate-date", "adjusted_open index must be unique"),
        ("unsorted-date", "adjusted_open index must be increasing"),
        ("duplicate-code", "adjusted_open columns must be unique"),
        ("unsorted-code", "adjusted_open columns must be increasing"),
        ("duplicate-raw", "raw_daily index must be unique"),
        ("duplicate-target", "target index must be unique"),
    ],
)
def test_simulate_rejects_duplicate_or_unsorted_inputs(case, message):
    dates = pd.bdate_range("2026-01-01", periods=3)
    opens = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    raw = _raw_ohlc(opens)
    targets = {dates[0]: pd.Series({"A": 1.0})}
    if case == "duplicate-date":
        opens.index = [dates[0], dates[0], dates[2]]
    elif case == "unsorted-date":
        opens = opens.iloc[[1, 0, 2]]
    elif case == "duplicate-code":
        opens.columns = ["A", "A"]
    elif case == "unsorted-code":
        opens = opens[["B", "A"]]
    elif case == "duplicate-raw":
        raw = pd.concat([raw, raw.iloc[[0]]])
    else:
        targets[dates[0]] = pd.Series(
            [0.5, 0.5],
            index=["A", "A"],
        )

    with pytest.raises(ValueError, match=message):
        simulate(targets, opens, raw, cost_bps=20)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unsorted", "raw_daily index must be increasing"),
        ("reversed-levels", "raw_daily index levels must be date,code"),
    ],
)
def test_simulate_rejects_unsorted_or_reversed_raw_multiindex(case, message):
    dates = pd.bdate_range("2026-01-01", periods=3)
    opens = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    raw = _raw_ohlc(opens)
    if case == "unsorted":
        raw = raw.iloc[::-1]
    else:
        raw = raw.reorder_levels(["code", "date"]).sort_index()

    with pytest.raises(ValueError, match=message):
        simulate({}, opens, raw, cost_bps=20)


@pytest.mark.parametrize("cost_bps", [-1.0, np.nan, np.inf])
def test_simulate_rejects_invalid_cost(cost_bps):
    dates = pd.bdate_range("2026-01-01", periods=2)
    opens = pd.DataFrame({"A": [10.0, 10.0]}, index=dates)

    with pytest.raises(ValueError, match="cost_bps must be finite and nonnegative"):
        simulate({}, opens, _raw_ohlc(opens), cost_bps=cost_bps)


def test_simulate_executes_from_the_same_normalized_numeric_target_copy():
    dates = pd.bdate_range("2026-01-01", periods=3)
    opens = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    targets = {dates[0]: pd.Series({"A": "1.0"})}

    result = simulate(targets, opens, _raw_ohlc(opens), cost_bps=0)

    assert result["trades"][["code", "side"]].to_dict("records") == [
        {"code": "A", "side": "buy"}
    ]
    assert result["trades"].loc[0, "notional"] == 1.0


@pytest.mark.parametrize("weight", ["not-a-number", np.inf, -0.1])
def test_simulate_rejects_invalid_target_weights(weight):
    dates = pd.bdate_range("2026-01-01", periods=2)
    opens = pd.DataFrame({"A": [10.0, 10.0]}, index=dates)

    with pytest.raises(ValueError, match="target weights must be finite and nonnegative"):
        simulate(
            {dates[0]: pd.Series({"A": weight})},
            opens,
            _raw_ohlc(opens),
            cost_bps=0,
        )


def test_build_targets_anchors_first_live_pool_day_and_keeps_five_day_cadence():
    dates = pd.bdate_range("2026-01-01", periods=12)
    codes = [f"{number:06d}" for number in range(1, 7)]
    score = pd.DataFrame(np.nan, index=dates, columns=codes)
    score.loc[dates[0], codes[:4]] = [4.0, 3.0, 2.0, 1.0]
    score.loc[dates[1], codes[:5]] = [10.0, 10.0, 9.0, 8.0, 100.0]
    score.loc[dates[6], codes[:3]] = [3.0, 2.0, 1.0]
    score.loc[dates[11], codes[:4]] = [1.0, 4.0, 3.0, 2.0]
    pool_rows = []
    for day in dates:
        members = codes[:3] if day == dates[0] else codes[:4]
        pool_rows.extend({"date": str(day.date()), "code": code} for code in members)
    pools = pd.DataFrame(pool_rows)

    targets = build_targets(
        score,
        pools,
        every=5,
        top_n=2,
        min_count=4,
    )

    assert list(targets) == [dates[1], dates[11]]
    assert targets[dates[1]].index.tolist() == ["000001", "000002"]
    assert targets[dates[1]].tolist() == [0.5, 0.5]
    assert targets[dates[11]].index.tolist() == ["000002", "000003"]
    assert dates[6] not in targets
    assert "000005" not in targets[dates[1]]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate-score-date", "score index must be unique"),
        ("unsorted-score-date", "score index must be increasing"),
        ("duplicate-score-code", "score columns must be unique"),
        ("duplicate-pool", "pools contains duplicate date/code"),
    ],
)
def test_build_targets_rejects_duplicate_or_unsorted_inputs(case, message):
    dates = pd.bdate_range("2026-01-01", periods=2)
    score = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=dates,
        columns=["A", "B"],
    )
    pools = pd.DataFrame({"date": dates.repeat(2), "code": ["A", "B"] * 2})
    if case == "duplicate-score-date":
        score.index = [dates[0], dates[0]]
    elif case == "unsorted-score-date":
        score = score.iloc[::-1]
    elif case == "duplicate-score-code":
        score.columns = ["A", "A"]
    else:
        pools = pd.concat([pools, pools.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=message):
        build_targets(score, pools, every=1, top_n=1, min_count=1)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("every", 0, "every must be a positive integer"),
        ("top_n", 0, "top_n must be a positive integer"),
        ("min_count", 0, "min_count must be a positive integer"),
    ],
)
def test_build_targets_rejects_invalid_integer_parameters(
    parameter,
    value,
    message,
):
    day = pd.Timestamp("2026-01-05")
    score = pd.DataFrame([[1.0]], index=[day], columns=["A"])
    pools = pd.DataFrame({"date": [day], "code": ["A"]})
    kwargs = {"every": 1, "top_n": 1, "min_count": 1, parameter: value}

    with pytest.raises(ValueError, match=message):
        build_targets(score, pools, **kwargs)


def test_build_benchmark_targets_equal_weights_same_day_final_pool():
    dates = pd.bdate_range("2026-01-01", periods=3)
    pools = pd.DataFrame({
        "date": [dates[0], dates[0], dates[2]],
        "code": ["B", "A", "C"],
    })

    targets = build_benchmark_targets(pools, [dates[0], dates[1]])

    assert list(targets) == [dates[0]]
    assert targets[dates[0]].index.tolist() == ["A", "B"]
    assert targets[dates[0]].tolist() == [0.5, 0.5]


def test_build_benchmark_targets_rejects_duplicate_signal_dates():
    day = pd.Timestamp("2026-01-05")
    pools = pd.DataFrame({"date": [day], "code": ["A"]})

    with pytest.raises(ValueError, match="signal_dates must be unique"):
        build_benchmark_targets(pools, [day, day])


METRIC_KEYS = {
    "strategy_total",
    "benchmark_total",
    "strategy_annual",
    "benchmark_annual",
    "annual_excess",
    "sharpe",
    "information_ratio",
    "max_drawdown",
    "monthly_win_rate",
    "excess_nw_t",
    "annual_turnover",
}


def test_portfolio_metrics_handles_zero_volatility_without_warning():
    dates = pd.bdate_range("2026-01-01", periods=3)
    strategy = {
        "returns": pd.Series(0.0, index=dates),
        "turnover": pd.Series([0.0, 0.1, 0.2], index=dates),
    }
    benchmark = {"returns": pd.Series(0.0, index=dates)}

    result = portfolio_metrics(strategy, benchmark)

    assert set(result) == METRIC_KEYS
    assert result["strategy_total"] == 0.0
    assert result["benchmark_total"] == 0.0
    assert result["strategy_annual"] == 0.0
    assert result["annual_excess"] == 0.0
    assert np.isnan(result["sharpe"])
    assert np.isnan(result["information_ratio"])
    assert result["max_drawdown"] == 0.0
    assert result["monthly_win_rate"] == 0.0
    assert np.isnan(result["excess_nw_t"])
    assert result["annual_turnover"] == pytest.approx(0.3 * 252 / 3)


def test_portfolio_metrics_returns_stable_nan_schema_for_empty_sample():
    empty = pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    strategy = {"returns": empty, "turnover": empty}
    benchmark = {"returns": empty}

    result = portfolio_metrics(strategy, benchmark)

    assert set(result) == METRIC_KEYS
    assert all(np.isnan(value) for value in result.values())


def test_portfolio_metrics_matches_compounding_monthly_wins_and_turnover():
    dates = pd.to_datetime([
        "2026-01-30",
        "2026-01-31",
        "2026-02-02",
        "2026-02-03",
    ])
    strategy_returns = pd.Series([0.0, 0.1, 0.0, -0.01], index=dates)
    benchmark_returns = pd.Series([0.0, 0.05, 0.0, 0.02], index=dates)
    turnover = pd.Series([0.0, 0.1, 0.0, 0.2], index=dates)

    result = portfolio_metrics(
        {"returns": strategy_returns, "turnover": turnover},
        {"returns": benchmark_returns},
    )

    strategy_total = float((1 + strategy_returns).prod() - 1)
    benchmark_total = float((1 + benchmark_returns).prod() - 1)
    excess = strategy_returns - benchmark_returns
    assert result["strategy_total"] == pytest.approx(strategy_total)
    assert result["benchmark_total"] == pytest.approx(benchmark_total)
    assert result["strategy_annual"] == pytest.approx(
        (1 + strategy_total) ** (252 / len(dates)) - 1
    )
    assert result["sharpe"] == pytest.approx(
        strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252)
    )
    assert result["information_ratio"] == pytest.approx(
        excess.mean() / excess.std(ddof=0) * np.sqrt(252)
    )
    assert result["max_drawdown"] == pytest.approx(-0.01)
    assert result["monthly_win_rate"] == 0.5
    assert np.isfinite(result["excess_nw_t"])
    assert result["annual_turnover"] == pytest.approx(0.3 * 252 / 4)
