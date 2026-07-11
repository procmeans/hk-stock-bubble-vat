import pandas as pd
import pytest

from strategies import backtest


def test_signal_takes_effect_next_day_with_costs(make_panel):
    panel = make_panel({"a": [100, 110, 121], "b": [100, 100, 100]})
    weights = pd.DataFrame(
        {"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}, index=panel["close"].index
    )

    result = backtest.run(weights, panel, cost_bps=20)

    assert result["gross"].iloc[0] == 0.0            # T 日信号当日不产生收益
    assert result["gross"].iloc[1] == pytest.approx(0.10)
    assert result["turnover"].iloc[1] == pytest.approx(1.0)   # 次日建仓计换手
    assert result["net"].iloc[1] == pytest.approx(0.10 - 1.0 * 20 / 1e4)
    assert result["equity"].iloc[2] == pytest.approx(
        (1 + result["net"].iloc[1]) * (1 + result["net"].iloc[2])
    )


def test_slippage_adds_to_cost(make_panel):
    panel = make_panel({"a": [100, 100, 100]})
    weights = pd.DataFrame({"a": [1.0, 1.0, 1.0]}, index=panel["close"].index)

    result = backtest.run(weights, panel, cost_bps=20, slippage_bps=10)

    assert result["net"].iloc[1] == pytest.approx(-1.0 * 30 / 1e4)
