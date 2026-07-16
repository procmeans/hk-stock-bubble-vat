import numpy as np
import pandas as pd
import pytest

from intraday.preprocess import (
    DIRECTIONS,
    compose,
    neutralize_day,
    preprocess_panels,
    winsorize_mad,
)


def _cross_section(codes, day):
    cap = pd.Series([10, 20, 30, 10, 20, 30], index=codes, dtype=float)
    industry = pd.Series(["x", "x", "x", "y", "y", "y"], index=codes)
    values = 2.0 * np.log(cap) + industry.map({"x": 1.0, "y": -1.0})
    values += pd.Series([-0.2, 0.1, 0.1, 0.2, -0.1, -0.1], index=codes)
    attributes = pd.DataFrame({
        "date": day,
        "code": codes,
        "float_cap": cap.to_numpy(),
        "industry": industry.to_numpy(),
    })
    return values, cap, industry, attributes


def _four_factor_panels(dates, codes, values):
    panel = pd.DataFrame(
        [values.reindex(codes).to_numpy() for _ in dates],
        index=pd.DatetimeIndex(dates),
        columns=codes,
    )
    return {name: panel.copy() for name in DIRECTIONS}


def test_winsorize_mad_uses_scaled_five_mad_bounds():
    values = pd.Series([0.0, 1.0, 2.0, 100.0, np.nan])

    result = winsorize_mad(values)

    assert result.iloc[:3].tolist() == [0.0, 1.0, 2.0]
    assert result.iloc[3] == pytest.approx(1.5 + 5.0 * 1.4826)
    assert np.isnan(result.iloc[4])


def test_winsorize_mad_preserves_mad_zero_and_nan_inputs():
    zero_mad = pd.Series([1.0, 1.0, 1.0, 100.0, np.nan])
    all_nan = pd.Series([np.nan, np.nan])

    zero_result = winsorize_mad(zero_mad)
    nan_result = winsorize_mad(all_nan)

    pd.testing.assert_series_equal(zero_result, zero_mad)
    pd.testing.assert_series_equal(nan_result, all_nan)
    assert zero_result is not zero_mad
    assert nan_result is not all_nan


def test_neutralize_removes_size_and_industry_exposure():
    index = list("ABCDEF")
    cap = pd.Series([10, 20, 30, 10, 20, 30], index=index, dtype=float)
    industry = pd.Series(["x", "x", "x", "y", "y", "y"], index=index)
    values = np.log(cap) * 2 + industry.map({"x": 1.0, "y": -1.0})
    values = values + pd.Series(
        [-0.2, 0.1, 0.1, 0.2, -0.1, -0.1],
        index=index,
    )

    result = neutralize_day(values, cap, industry, min_count=6)

    assert result.corr(np.log(cap)) == pytest.approx(0.0, abs=1e-10)
    assert result.groupby(industry).mean().abs().max() < 1e-10
    assert result.std(ddof=0) == pytest.approx(1.0)


def test_neutralize_returns_nan_when_valid_count_is_too_small():
    index = ["A", "B", "C"]
    result = neutralize_day(
        pd.Series([1.0, 2.0, 3.0], index=index),
        pd.Series([10.0, 20.0, 30.0], index=index),
        pd.Series(["x", "x", "y"], index=index),
        min_count=4,
    )

    assert result.index.tolist() == index
    assert result.isna().all()


def test_neutralize_excludes_invalid_caps_and_missing_industries():
    index = list("ABCDEFGHIJK")
    cap = pd.Series(
        [
            10.0,
            20.0,
            30.0,
            10.0,
            20.0,
            30.0,
            0.0,
            -1.0,
            np.inf,
            np.nan,
            40.0,
        ],
        index=index,
    )
    industry = pd.Series(
        ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", np.nan],
        index=index,
    )
    values = np.log(cap.where(cap > 0, 10.0))
    values += industry.map({"x": 1.0, "y": -1.0}).fillna(0.0)
    values += pd.Series(
        [
            -0.2,
            0.1,
            0.1,
            0.2,
            -0.1,
            -0.1,
            0.3,
            -0.3,
            0.4,
            -0.4,
            0.5,
        ],
        index=index,
    )

    result = neutralize_day(values, cap, industry, min_count=6)

    assert result.loc[list("ABCDEF")].notna().all()
    assert result.loc[list("GHIJK")].isna().all()
    assert result.dropna().std(ddof=0) == pytest.approx(1.0)


def test_neutralize_handles_rank_deficient_design_matrix():
    index = list("ABCDE")
    values = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=index)
    cap = pd.Series(10.0, index=index)
    industry = pd.Series("only", index=index)

    result = neutralize_day(values, cap, industry, min_count=5)

    assert np.isfinite(result).all()
    assert result.std(ddof=0) == pytest.approx(1.0)


def test_neutralize_invalidates_zero_variance_input():
    index = list("ABCDE")
    result = neutralize_day(
        pd.Series(3.0, index=index),
        pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=index),
        pd.Series(["x", "x", "x", "y", "y"], index=index),
        min_count=5,
    )

    assert result.isna().all()


def test_neutralize_invalidates_numerically_zero_residual_variance():
    index = list("ABCDEF")
    log_cap = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=index)
    industry = pd.Series(["x", "x", "x", "y", "y", "y"], index=index)
    values = 2.0 * log_cap + industry.map({"x": 1.0, "y": -1.0})

    result = neutralize_day(
        values,
        np.exp(log_cap),
        industry,
        min_count=6,
    )

    assert result.isna().all()


def test_factor_directions_are_fixed_and_negative():
    assert DIRECTIONS == {
        "rskew": -1.0,
        "cpv_mean": -1.0,
        "cpv_std": -1.0,
        "smart": -1.0,
    }


def test_compose_weights_three_logic_blocks_equally():
    factor = pd.DataFrame([[1.0, -1.0]], columns=["A", "B"])
    processed = {
        "rskew": factor,
        "cpv_mean": factor,
        "cpv_std": factor,
        "smart": -factor,
    }

    result = compose(processed)

    assert result["cpv_block"].iloc[0, 0] > 0
    assert result["score"].iloc[0, 0] == pytest.approx(1 / 3)


def test_compose_requires_all_four_bound_factor_keys():
    frame = pd.DataFrame([[1.0, -1.0]], columns=["A", "B"])

    with pytest.raises(ValueError, match="missing required factors: smart"):
        compose({
            "rskew": frame,
            "cpv_mean": frame,
            "cpv_std": frame,
        })


def test_compose_equally_weights_cpv_subfactors_before_zscore():
    cpv_mean = pd.DataFrame([[-2.0, 0.0, 2.0]], columns=["A", "B", "C"])
    cpv_std = pd.DataFrame([[0.0, 2.0, -2.0]], columns=["A", "B", "C"])
    zeros = cpv_mean * 0.0

    result = compose({
        "rskew": zeros,
        "cpv_mean": cpv_mean,
        "cpv_std": cpv_std,
        "smart": zeros,
    })

    averaged = pd.Series([-1.0, 1.0, 0.0], index=["A", "B", "C"])
    expected = (averaged - averaged.mean()) / averaged.std(ddof=0)
    assert result["cpv_block"].iloc[0].to_numpy() == pytest.approx(
        expected.to_numpy()
    )


def test_compose_aligns_axes_and_does_not_renormalize_missing_blocks():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    rskew = pd.DataFrame(
        [[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]],
        index=dates,
        columns=["A", "B", "C"],
    )
    cpv_mean = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=dates[:1],
        columns=["A", "B", "C"],
    )
    cpv_std = pd.DataFrame(
        [[1.0, 2.0]],
        index=dates[:1],
        columns=["A", "B"],
    )
    smart = pd.DataFrame(
        [[1.0, 0.0, -1.0]],
        index=dates[:1],
        columns=["A", "B", "C"],
    )

    result = compose({
        "rskew": rskew,
        "cpv_mean": cpv_mean,
        "cpv_std": cpv_std,
        "smart": smart,
    })

    assert result["cpv_block"].index.equals(dates)
    assert result["cpv_block"].columns.tolist() == ["A", "B", "C"]
    assert result["cpv_block"].loc[dates[0], "A"] == pytest.approx(-1.0)
    assert result["cpv_block"].loc[dates[0], "B"] == pytest.approx(1.0)
    assert np.isnan(result["cpv_block"].loc[dates[0], "C"])
    assert np.isnan(result["score"].loc[dates[0], "C"])
    assert result["score"].loc[dates[1]].isna().all()


def test_preprocess_panels_aligns_factor_axes_and_applies_negative_directions():
    dates = pd.bdate_range("2026-01-05", periods=2)
    codes = [f"{number:06d}" for number in range(1, 7)]
    values, cap, industry, attributes = _cross_section(codes, dates[0])
    factors = _four_factor_panels(dates, codes, values)
    factors["cpv_mean"] = factors["cpv_mean"].loc[
        dates[:1], codes[::-1]
    ]
    factors["cpv_std"] = factors["cpv_std"].loc[dates[:1]]
    factors["smart"] = factors["smart"].loc[dates[:1]]
    pools = pd.DataFrame({
        "date": [str(dates[0].date())] * len(codes),
        "code": codes[::-1],
    })

    result = preprocess_panels(
        factors,
        pools,
        attributes.assign(date=dates[0] + pd.Timedelta(hours=16)),
        min_count=6,
    )

    expected = neutralize_day(-values, cap, industry, min_count=6)
    for name in DIRECTIONS:
        assert result[name].index.equals(dates)
        assert result[name].columns.tolist() == codes
        assert result[name].loc[dates[0]].to_numpy() == pytest.approx(
            expected.reindex(codes).to_numpy()
        )
        assert result[name].loc[dates[1]].isna().all()


def test_preprocess_panels_uses_factor_calendar_for_four_five_day_attr_boundary():
    dates = pd.bdate_range("2026-01-05", periods=6)
    codes = [f"{number:06d}" for number in range(1, 7)]
    values, _, _, attributes = _cross_section(codes, dates[0])
    factors = _four_factor_panels(dates, codes, values)
    pools = pd.DataFrame({
        "date": np.repeat(dates[[4, 5]], len(codes)),
        "code": codes * 2,
    })

    result = preprocess_panels(
        factors,
        pools,
        attributes,
        min_count=6,
    )

    for panel in result.values():
        assert panel.loc[dates[4]].notna().all()
        assert panel.loc[dates[5]].isna().all()


def test_preprocess_panels_returns_nan_without_prior_attributes():
    day = pd.Timestamp("2026-01-05")
    codes = [f"{number:06d}" for number in range(1, 7)]
    values, _, _, attributes = _cross_section(codes, day)
    factors = _four_factor_panels([day], codes, values)
    pools = pd.DataFrame({"date": day, "code": codes})

    result = preprocess_panels(
        factors,
        pools,
        attributes.assign(date=day + pd.Timedelta(days=1)),
        min_count=6,
    )

    assert all(panel.isna().all().all() for panel in result.values())


@pytest.mark.parametrize(
    "case",
    ["too-few-members", "missing-member-attribute", "missing-industry"],
)
def test_preprocess_panels_invalidates_incomplete_cross_sections(case):
    day = pd.Timestamp("2026-01-05")
    codes = [f"{number:06d}" for number in range(1, 7)]
    values, _, _, attributes = _cross_section(codes, day)
    factors = _four_factor_panels([day], codes, values)
    pool_codes = codes
    if case == "too-few-members":
        pool_codes = codes[:-1]
    elif case == "missing-member-attribute":
        attributes = attributes.iloc[:-1]
    else:
        attributes.loc[attributes.index[-1], "industry"] = pd.NA
    pools = pd.DataFrame({"date": day, "code": pool_codes})

    result = preprocess_panels(
        factors,
        pools,
        attributes,
        min_count=6,
    )

    assert all(panel.isna().all().all() for panel in result.values())


def test_preprocess_panels_requires_exact_four_factor_keys():
    day = pd.Timestamp("2026-01-05")
    codes = [f"{number:06d}" for number in range(1, 7)]
    values, _, _, attributes = _cross_section(codes, day)
    factors = _four_factor_panels([day], codes, values)
    factors.pop("smart")

    with pytest.raises(ValueError, match="missing required factors: smart"):
        preprocess_panels(
            factors,
            pd.DataFrame({"date": day, "code": codes}),
            attributes,
            min_count=6,
        )


@pytest.mark.parametrize(
    ("pools", "message"),
    [
        (
            pd.DataFrame({
                "date": ["2026-01-05", "2026-01-05 16:00"],
                "code": ["000001", "000001"],
            }),
            "pools contains duplicate date/code",
        ),
        (
            pd.DataFrame({"date": ["not-a-date"], "code": ["000001"]}),
            "pools contains invalid date",
        ),
    ],
)
def test_preprocess_panels_rejects_invalid_or_duplicate_pool_rows(
    pools,
    message,
):
    empty_factors = {
        name: pd.DataFrame(index=pd.DatetimeIndex([]))
        for name in DIRECTIONS
    }
    empty_attributes = pd.DataFrame(
        columns=["date", "code", "float_cap", "industry"]
    )

    with pytest.raises(ValueError, match=message):
        preprocess_panels(empty_factors, pools, empty_attributes, min_count=1)


def test_preprocess_panels_requires_pool_columns():
    empty_factors = {
        name: pd.DataFrame(index=pd.DatetimeIndex([]))
        for name in DIRECTIONS
    }
    empty_attributes = pd.DataFrame(
        columns=["date", "code", "float_cap", "industry"]
    )

    with pytest.raises(ValueError, match="pools missing required columns: code"):
        preprocess_panels(
            empty_factors,
            pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}),
            empty_attributes,
            min_count=1,
        )


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        (
            pd.DataFrame({
                "date": ["2026-01-05", "2026-01-05 16:00"],
                "code": ["000001", "000001"],
                "float_cap": [10.0, 20.0],
                "industry": ["x", "x"],
            }),
            "attributes contains duplicate date/code",
        ),
        (
            pd.DataFrame({
                "date": ["not-a-date"],
                "code": ["000001"],
                "float_cap": [10.0],
                "industry": ["x"],
            }),
            "attributes contains invalid date",
        ),
    ],
)
def test_preprocess_panels_rejects_invalid_or_duplicate_attributes(
    attributes,
    message,
):
    empty_factors = {
        name: pd.DataFrame(index=pd.DatetimeIndex([]))
        for name in DIRECTIONS
    }
    empty_pools = pd.DataFrame(columns=["date", "code"])

    with pytest.raises(ValueError, match=message):
        preprocess_panels(empty_factors, empty_pools, attributes, min_count=1)


def test_preprocess_panels_requires_attribute_columns():
    empty_factors = {
        name: pd.DataFrame(index=pd.DatetimeIndex([]))
        for name in DIRECTIONS
    }
    empty_pools = pd.DataFrame(columns=["date", "code"])

    with pytest.raises(
        ValueError,
        match="attributes missing required columns: industry",
    ):
        preprocess_panels(
            empty_factors,
            empty_pools,
            pd.DataFrame(columns=["date", "code", "float_cap"]),
            min_count=1,
        )
