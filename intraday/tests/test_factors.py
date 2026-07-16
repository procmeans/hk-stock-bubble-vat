import numpy as np
import pandas as pd
import pytest

from intraday.factors import factor_panels, minute_day_factors


def _minute_frame(day, close=None, volume=None):
    close = np.asarray(
        close if close is not None else [10.0, 10.2, 10.1, 10.4],
        dtype=float,
    )
    volume = np.asarray(
        volume if volume is not None else [10.0, 20.0, 30.0, 40.0],
        dtype=float,
    )
    return pd.DataFrame({
        "time": pd.date_range(
            pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30),
            periods=len(close),
            freq="min",
        ),
        "close": close,
        "volume": volume,
        "amount": close * volume,
    })


def test_minute_day_factors_match_formulas_and_interface():
    close = np.exp(np.cumsum([0.0, 0.01, -0.02, 0.03]))
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-12 09:30", periods=4, freq="min"),
        "close": close,
        "volume": [10.0, 20.0, 30.0, 40.0],
        "amount": close * [10.0, 20.0, 30.0, 40.0],
    })

    result = minute_day_factors(frame)

    returns = np.array([0.01, -0.02, 0.03])
    expected_rskew = (
        np.sqrt(3)
        * (returns ** 3).sum()
        / ((returns ** 2).sum() ** 1.5)
    )
    assert set(result) == {"rskew_day", "cpv_day", "smart_q_day"}
    assert result["rskew_day"] == pytest.approx(expected_rskew)
    assert result["cpv_day"] == pytest.approx(
        frame["close"].corr(frame["volume"])
    )


def test_smart_money_includes_threshold_crossing_minute():
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-12 09:30", periods=3, freq="min"),
        "close": [10.0, 12.0, 12.1],
        "volume": [1.0, 80.0, 19.0],
        "amount": [10.0, 960.0, 229.9],
    })

    result = minute_day_factors(frame)

    expected = 12.0 / (1199.9 / 100.0)
    assert result["smart_q_day"] == pytest.approx(expected)


def test_smart_money_stops_when_selected_volume_exactly_reaches_target():
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-12 09:30", periods=3, freq="min"),
        "close": [10.0, 11.0, 11.1],
        "volume": [1.0, 20.0, 80.0],
        "amount": [10.0, 220.0, 888.0],
    })

    result = minute_day_factors(frame)

    expected_all_vwap = frame["amount"].sum() / frame["volume"].sum()
    assert result["smart_q_day"] == pytest.approx(11.0 / expected_all_vwap)


def test_smart_money_breaks_smartness_ties_by_earliest_time():
    times = pd.date_range("2026-01-12 09:30", periods=6, freq="min")
    close = 2.0 ** np.arange(6)
    chronological = pd.DataFrame({
        "time": times,
        "close": close,
        "volume": np.ones(6),
        "amount": close,
    })
    shuffled = chronological.iloc[[0, 5, 4, 3, 2, 1]].reset_index(drop=True)

    result = minute_day_factors(shuffled)

    expected_all_vwap = chronological["amount"].sum() / 6.0
    assert result["smart_q_day"] == pytest.approx(close[1] / expected_all_vwap)


def test_minute_day_factors_returns_nan_for_empty_and_single_row_inputs():
    empty_result = minute_day_factors(pd.DataFrame())
    single_result = minute_day_factors(_minute_frame("2026-01-12").iloc[[0]])

    for result in [empty_result, single_result]:
        assert set(result) == {"rskew_day", "cpv_day", "smart_q_day"}
        assert all(np.isnan(value) for value in result.values())


def test_minute_day_factors_handles_constant_prices_without_warnings():
    frame = _minute_frame(
        "2026-01-12",
        close=[10.0, 10.0, 10.0, 10.0],
    )

    result = minute_day_factors(frame)

    assert np.isnan(result["rskew_day"])
    assert np.isnan(result["cpv_day"])
    assert result["smart_q_day"] == pytest.approx(1.0)


def test_minute_day_factors_excludes_zero_volume_from_cpv_and_smart_money():
    frame = _minute_frame(
        "2026-01-12",
        close=[10.0, 11.0, 12.0],
        volume=[0.0, 0.0, 0.0],
    )

    result = minute_day_factors(frame)

    assert np.isfinite(result["rskew_day"])
    assert np.isnan(result["cpv_day"])
    assert np.isnan(result["smart_q_day"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("close", np.inf),
        ("close", np.nan),
        ("volume", np.inf),
        ("amount", np.nan),
    ],
)
def test_minute_day_factors_rejects_non_finite_inputs(column, value):
    frame = _minute_frame("2026-01-12")
    frame.loc[1, column] = value

    with pytest.raises(ValueError, match="non-finite"):
        minute_day_factors(frame)


def test_factor_panels_exposes_only_bound_output_keys():
    result = factor_panels([], ["000001"], pd.DatetimeIndex([]))

    assert set(result) == {"rskew", "cpv_mean", "cpv_std", "smart"}


def test_factor_panels_returns_aligned_empty_panels_for_empty_partitions():
    dates = pd.bdate_range("2026-01-05", periods=2)
    codes = ["000002", "000001"]

    result = factor_panels([], codes, dates)

    for panel in result.values():
        assert panel.index.equals(dates)
        assert panel.columns.tolist() == codes
        assert panel.isna().all().all()


def test_factor_panels_aligns_missing_code_days_before_rolling():
    dates = pd.bdate_range("2026-01-05", periods=2)
    frame = _minute_frame(dates[0]).assign(code="000001")

    result = factor_panels(
        [(dates[0], frame)],
        ["000001", "000002"],
        dates,
        window=2,
        min_periods=1,
    )

    expected_day = minute_day_factors(frame)
    for key, daily_key in [
        ("rskew", "rskew_day"),
        ("cpv_mean", "cpv_day"),
        ("smart", "smart_q_day"),
    ]:
        assert result[key].loc[dates[1], "000001"] == pytest.approx(
            expected_day[daily_key]
        )
        assert result[key]["000002"].isna().all()
    assert result["cpv_std"].isna().all().all()


def test_factor_panels_applies_20_day_15_observation_boundaries_and_cpv_ddof():
    dates = pd.bdate_range("2026-01-05", periods=21)
    partitions = []
    daily_values = []
    for offset, day in enumerate(dates):
        close = [
            10.0,
            10.3 + 0.01 * offset,
            10.1 - 0.02 * offset,
            10.5 + 0.03 * offset,
        ]
        frame = _minute_frame(day, close=close).assign(code="000001")
        partitions.append((day, frame))
        daily_values.append(minute_day_factors(frame))

    result = factor_panels(partitions, ["000001"], dates)

    expected_rskew = pd.Series(
        [value["rskew_day"] for value in daily_values], index=dates
    ).rolling(20, min_periods=15).mean()
    expected_cpv = pd.Series(
        [value["cpv_day"] for value in daily_values], index=dates
    )
    expected_smart = pd.Series(
        [value["smart_q_day"] for value in daily_values], index=dates
    ).rolling(20, min_periods=15).mean()

    assert result["rskew"]["000001"].equals(expected_rskew)
    assert result["cpv_mean"]["000001"].equals(
        expected_cpv.rolling(20, min_periods=15).mean()
    )
    assert result["cpv_std"]["000001"].equals(
        expected_cpv.rolling(20, min_periods=15).std(ddof=1)
    )
    assert result["smart"]["000001"].equals(expected_smart)
    assert np.isnan(result["cpv_mean"].iloc[13, 0])
    assert np.isfinite(result["cpv_mean"].iloc[14, 0])
    assert result["cpv_mean"].iloc[20, 0] == pytest.approx(
        expected_cpv.iloc[1:].mean()
    )
