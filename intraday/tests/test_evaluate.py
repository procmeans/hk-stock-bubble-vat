import numpy as np
import pandas as pd
import pytest

from intraday.evaluate import (
    evaluate_factors,
    forward_open_return,
    newey_west_t,
    quantile_cohorts,
    rank_ic,
)


def test_forward_return_is_t1_to_t6():
    dates = pd.bdate_range("2026-01-01", periods=8)
    prices = pd.DataFrame({"A": np.arange(10.0, 18.0)}, index=dates)

    result = forward_open_return(prices, horizon=5)

    assert result.loc[dates[0], "A"] == pytest.approx(16.0 / 11.0 - 1)
    assert result.iloc[-6:]["A"].isna().all()


def test_newey_west_matches_bartlett_definition():
    values = pd.Series([0.01, -0.02, 0.03, 0.01])
    array = values.to_numpy()
    demeaned = array - array.mean()
    variance = demeaned @ demeaned / len(array)
    variance += 2 * 0.5 * (demeaned[1:] @ demeaned[:-1] / len(array))
    expected = array.mean() / np.sqrt(max(variance, 0) / len(array))

    result = newey_west_t(values, lags=1)

    assert result == pytest.approx(expected)


def test_rank_ic_and_quantiles_are_monotonic():
    dates = pd.bdate_range("2026-01-01", periods=10)
    codes = list("ABCDE")
    factor = pd.DataFrame(
        np.tile(np.arange(5), (10, 1)),
        index=dates,
        columns=codes,
    )
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    for position in range(1, len(dates)):
        opens.iloc[position] = opens.iloc[position - 1] * (
            1 + np.arange(5) * 0.001
        )

    forward = forward_open_return(opens, horizon=2)
    ic = rank_ic(factor, forward, min_count=5)
    quantiles = quantile_cohorts(
        factor,
        opens,
        q=5,
        horizon=2,
        min_count=5,
    )

    assert ic.dropna().eq(1.0).all()
    assert quantiles.columns.tolist() == list(range(5))
    assert quantiles.mean().is_monotonic_increasing


def test_forward_return_excludes_invalid_entry_and_exit_prices():
    dates = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [11.0, 0.0, np.inf, -1.0, 11.0],
            [12.0, 12.0, 12.0, 12.0, 0.0],
        ],
        index=dates,
        columns=list("ABCDE"),
    )

    result = forward_open_return(prices, horizon=1)

    assert result.loc[dates[0], "A"] == pytest.approx(12.0 / 11.0 - 1)
    assert result.loc[dates[0], list("BCDE")].isna().all()


@pytest.mark.parametrize(
    ("axis_case", "message"),
    [
        ("duplicate-date", "open_prices index must be unique"),
        ("unsorted-date", "open_prices index must be increasing"),
        ("duplicate-code", "open_prices columns must be unique"),
        ("unsorted-code", "open_prices columns must be increasing"),
    ],
)
def test_forward_return_rejects_duplicate_or_unsorted_axes(axis_case, message):
    dates = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame(
        np.arange(6.0).reshape(3, 2) + 10.0,
        index=dates,
        columns=["A", "B"],
    )
    if axis_case == "duplicate-date":
        prices.index = [dates[0], dates[0], dates[2]]
    elif axis_case == "unsorted-date":
        prices = prices.iloc[[1, 0, 2]]
    elif axis_case == "duplicate-code":
        prices.columns = ["A", "A"]
    else:
        prices = prices[["B", "A"]]

    with pytest.raises(ValueError, match=message):
        forward_open_return(prices, horizon=1)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_forward_return_requires_positive_integer_horizon(horizon):
    prices = pd.DataFrame(
        {"A": [10.0, 11.0]},
        index=pd.bdate_range("2026-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        forward_open_return(prices, horizon=horizon)


def test_newey_west_handles_short_nan_and_nonfinite_samples():
    assert np.isnan(newey_west_t([], lags=4))
    assert np.isnan(newey_west_t([np.nan, 0.01], lags=4))
    assert np.isfinite(
        newey_west_t([np.nan, 0.01, 0.02, np.inf], lags=0)
    )
    assert np.isnan(newey_west_t([0.01, 0.01, 0.01], lags=2))


@pytest.mark.parametrize("lags", [-1, 1.5, True])
def test_newey_west_requires_nonnegative_integer_lags(lags):
    with pytest.raises(ValueError, match="lags must be a nonnegative integer"):
        newey_west_t([0.01, 0.02, 0.03], lags=lags)


def test_rank_ic_aligns_dates_and_codes_by_label():
    dates = pd.bdate_range("2026-01-01", periods=3)
    factor = pd.DataFrame(
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        index=dates[:2],
        columns=["A", "B", "C"],
    )
    forward = pd.DataFrame(
        [[3.0, 4.0, 5.0], [5.0, 4.0, 3.0]],
        index=dates[1:],
        columns=["B", "C", "D"],
    )

    result = rank_ic(factor, forward, min_count=2)

    assert result.index.tolist() == [dates[1]]
    assert result.iloc[0] == 1.0


def test_rank_ic_returns_nan_for_constant_cross_section_without_warning():
    day = pd.Timestamp("2026-01-05")
    factor = pd.DataFrame([[1.0, 1.0, 1.0]], index=[day], columns=list("ABC"))
    forward = pd.DataFrame([[1.0, 2.0, 3.0]], index=[day], columns=list("ABC"))

    result = rank_ic(factor, forward, min_count=3)

    assert result.index.tolist() == [day]
    assert np.isnan(result.iloc[0])


def test_rank_ic_excludes_nonfinite_values_before_min_count_and_correlation():
    day = pd.Timestamp("2026-01-05")
    factor = pd.DataFrame([[1.0, 3.0, np.inf]], index=[day], columns=list("ABC"))
    forward = pd.DataFrame([[1.0, 2.0, 0.0]], index=[day], columns=list("ABC"))

    result = rank_ic(factor, forward, min_count=2)

    assert result.iloc[0] == 1.0
    assert rank_ic(factor, forward, min_count=3).empty


def test_rank_ic_rejects_unsorted_factor_codes():
    day = pd.Timestamp("2026-01-05")
    factor = pd.DataFrame([[2.0, 1.0]], index=[day], columns=["B", "A"])
    forward = pd.DataFrame([[1.0, 2.0]], index=[day], columns=["A", "B"])

    with pytest.raises(ValueError, match="factor columns must be increasing"):
        rank_ic(factor, forward, min_count=2)


def test_quantile_ties_break_by_code_deterministically():
    dates = pd.bdate_range("2026-01-01", periods=3)
    codes = [f"{number:06d}" for number in range(10)]
    factor = pd.DataFrame(1.0, index=dates[:1], columns=codes)
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.iloc[2] = 100.0 * (1 + np.arange(10) * 0.01)

    result = quantile_cohorts(
        factor,
        opens,
        q=5,
        horizon=1,
        min_count=10,
    )

    assert result.loc[dates[2]].to_numpy() == pytest.approx(
        [0.005, 0.025, 0.045, 0.065, 0.085]
    )


def test_quantile_membership_requires_valid_t1_entry_open():
    dates = pd.bdate_range("2026-01-01", periods=3)
    codes = list("ABCDEF")
    factor = pd.DataFrame([np.arange(6.0)], index=dates[:1], columns=codes)
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.loc[dates[1], "A"] = 0.0
    opens.loc[dates[2]] = [200.0, 101.0, 102.0, 103.0, 104.0, 105.0]

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=1,
        min_count=5,
    )

    assert result.loc[dates[2], 0] == pytest.approx(0.02)
    assert result.loc[dates[2], 1] == pytest.approx(0.045)


def test_quantile_skips_cohort_when_buyable_count_is_below_minimum():
    dates = pd.bdate_range("2026-01-01", periods=3)
    codes = list("ABCDEF")
    factor = pd.DataFrame([np.arange(6.0)], index=dates[:1], columns=codes)
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.loc[dates[1], ["A", "B"]] = [0.0, np.nan]

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=1,
        min_count=5,
    )

    assert result.empty
    assert result.columns.tolist() == [0, 1]
    assert result.index.name == "date"
    assert result.columns.name == "group"


def test_quantile_uses_available_group_returns_when_later_price_is_missing():
    dates = pd.bdate_range("2026-01-01", periods=3)
    codes = list("ABCD")
    factor = pd.DataFrame([[0.0, 1.0, 2.0, 3.0]], index=dates[:1], columns=codes)
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.loc[dates[2]] = [np.nan, 102.0, 103.0, 104.0]

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=1,
        min_count=4,
    )

    assert result.loc[dates[2], 0] == pytest.approx(0.02)
    assert result.loc[dates[2], 1] == pytest.approx(0.035)


def test_quantile_skips_tail_without_full_horizon_with_stable_schema():
    dates = pd.bdate_range("2026-01-01", periods=3)
    factor = pd.DataFrame([[0.0, 1.0]], index=dates[1:2], columns=["A", "B"])
    opens = pd.DataFrame(100.0, index=dates, columns=["A", "B"])

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=1,
        min_count=2,
    )

    assert result.empty
    assert result.columns.tolist() == [0, 1]


def test_quantile_averages_five_day_style_overlapping_cohorts_by_date():
    dates = pd.bdate_range("2026-01-01", periods=5)
    codes = list("ABCD")
    factor = pd.DataFrame(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        index=dates[:2],
        columns=codes,
    )
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.loc[dates[3]] = [101.0, 101.0, 103.0, 103.0]
    opens.loc[dates[4]] = opens.loc[dates[3]] * [1.02, 1.02, 1.04, 1.04]

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=2,
        min_count=4,
    )

    assert result.index.tolist() == dates[2:].tolist()
    assert result.loc[dates[2]].to_numpy() == pytest.approx([0.0, 0.0])
    assert result.loc[dates[3]].to_numpy() == pytest.approx([0.02, 0.02])
    assert result.loc[dates[4]].to_numpy() == pytest.approx([0.04, 0.02])


def test_quantile_equally_averages_five_simultaneously_active_cohorts():
    dates = pd.bdate_range("2026-01-01", periods=11)
    codes = list("ABCD")
    ascending = [0.0, 1.0, 2.0, 3.0]
    descending = ascending[::-1]
    factor = pd.DataFrame(
        [ascending, descending, ascending, descending, ascending],
        index=dates[:5],
        columns=codes,
    )
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    opens.loc[dates[6]:] = [101.0, 101.0, 105.0, 105.0]

    result = quantile_cohorts(
        factor,
        opens,
        q=2,
        horizon=5,
        min_count=4,
    )

    assert result.loc[dates[6], 0] == pytest.approx(
        (0.01 * 3 + 0.05 * 2) / 5
    )
    assert result.loc[dates[6], 1] == pytest.approx(
        (0.05 * 3 + 0.01 * 2) / 5
    )


def test_quantile_rejects_unsorted_factor_codes():
    dates = pd.bdate_range("2026-01-01", periods=3)
    factor = pd.DataFrame([[1.0, 0.0]], index=dates[:1], columns=["B", "A"])
    opens = pd.DataFrame(100.0, index=dates, columns=["A", "B"])

    with pytest.raises(ValueError, match="factor columns must be increasing"):
        quantile_cohorts(factor, opens, q=2, horizon=1, min_count=2)


SUMMARY_COLUMNS = [
    "factor",
    "ic_mean",
    "ic_std",
    "icir",
    "positive_ic_rate",
    "ic_nw_t",
    "q5_q1_mean",
    "monotonicity",
]
QUANTILE_COLUMNS = ["factor", "date", "group", "return"]


def test_evaluate_factors_reports_bound_statistics_and_long_quantiles():
    dates = pd.bdate_range("2026-01-01", periods=12)
    codes = list("ABCDE")
    factor = pd.DataFrame(
        np.tile(np.arange(5.0), (len(dates), 1)),
        index=dates,
        columns=codes,
    )
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    for position in range(1, len(dates)):
        opens.iloc[position] = opens.iloc[position - 1] * (
            1 + np.arange(5) * 0.001
        )

    summary, daily_ic, quantiles = evaluate_factors(
        {"score": factor},
        opens,
        min_count=5,
    )

    assert summary.columns.tolist() == SUMMARY_COLUMNS
    assert summary["factor"].tolist() == ["score"]
    assert summary.loc[0, "ic_mean"] == 1.0
    assert summary.loc[0, "ic_std"] == 0.0
    assert np.isnan(summary.loc[0, "icir"])
    assert summary.loc[0, "positive_ic_rate"] == 1.0
    assert np.isnan(summary.loc[0, "ic_nw_t"])
    assert summary.loc[0, "q5_q1_mean"] == pytest.approx(0.004)
    assert summary.loc[0, "monotonicity"] == 1.0
    assert daily_ic.columns.tolist() == ["score"]
    assert daily_ic.index.name == "date"
    assert daily_ic["score"].dropna().eq(1.0).all()
    assert quantiles.columns.tolist() == QUANTILE_COLUMNS
    assert quantiles["factor"].eq("score").all()
    assert sorted(quantiles["group"].unique()) == list(range(5))


def test_evaluate_factors_returns_stable_schema_for_empty_factor_mapping():
    opens = pd.DataFrame(
        {"A": [10.0, 11.0]},
        index=pd.bdate_range("2026-01-01", periods=2),
    )

    summary, daily_ic, quantiles = evaluate_factors({}, opens, min_count=1)

    assert summary.empty
    assert summary.columns.tolist() == SUMMARY_COLUMNS
    assert daily_ic.empty
    assert daily_ic.index.name == "date"
    assert quantiles.empty
    assert quantiles.columns.tolist() == QUANTILE_COLUMNS


def test_evaluate_factors_uses_sample_ic_std():
    dates = pd.bdate_range("2026-01-01", periods=8)
    codes = list("ABCDE")
    factor = pd.DataFrame(np.nan, index=dates, columns=codes)
    factor.loc[dates[0]] = np.arange(5.0)
    factor.loc[dates[1]] = np.arange(5.0)[::-1]
    opens = pd.DataFrame(100.0, index=dates, columns=codes)
    for position in range(1, len(dates)):
        opens.iloc[position] = opens.iloc[position - 1] * (
            1 + np.arange(5) * 0.001
        )

    summary, daily_ic, _ = evaluate_factors(
        {"mixed": factor},
        opens,
        min_count=5,
    )

    assert daily_ic["mixed"].tolist() == [1.0, -1.0]
    assert summary.loc[0, "ic_mean"] == 0.0
    assert summary.loc[0, "ic_std"] == pytest.approx(np.sqrt(2.0))
    assert summary.loc[0, "icir"] == 0.0
    assert summary.loc[0, "positive_ic_rate"] == 0.5


def test_evaluate_factors_handles_empty_ic_and_quantiles_without_warning():
    dates = pd.bdate_range("2026-01-01", periods=8)
    factor = pd.DataFrame(np.nan, index=dates, columns=list("ABCDE"))
    opens = pd.DataFrame(100.0, index=dates, columns=list("ABCDE"))

    summary, daily_ic, quantiles = evaluate_factors(
        {"empty": factor},
        opens,
        min_count=5,
    )

    assert summary["factor"].tolist() == ["empty"]
    assert summary.loc[0, SUMMARY_COLUMNS[1:]].isna().all()
    assert daily_ic.empty
    assert daily_ic.columns.tolist() == ["empty"]
    assert quantiles.empty
    assert quantiles.columns.tolist() == QUANTILE_COLUMNS
