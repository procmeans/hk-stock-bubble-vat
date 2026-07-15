import numpy as np
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


@pytest.mark.parametrize(("selector", "top_n", "expected"), [
    pytest.param(combo.select_weighted, 99, 20, id="weighted-cap"),
    pytest.param(combo.select_funnel, 99, 20, id="funnel-cap"),
    pytest.param(combo.select_weighted, -1, 0, id="weighted-negative"),
    pytest.param(combo.select_funnel, -1, 0, id="funnel-negative"),
])
def test_selectors_cap_at_twenty_and_treat_negative_top_n_as_zero(
    selector, top_n, expected
):
    size = 42
    order = np.arange(1, size + 1, dtype=float)
    factors = pd.DataFrame({
        "date": ["2026-07-15"] * size,
        "ticker": [f"{index:06d}" for index in range(size)],
        "name": [f"stock-{index}" for index in range(size)],
        "attention_rise": order,
        "float_a": order,
        "total_shares": [100.0] * size,
        "momentum_7d": order / 100.0,
        "float_ratio": order / 100.0,
        "attention_pct": order / size,
        "momentum_pct": order / size,
        "low_float_pct": order[::-1] / size,
    })
    assert len(selector(factors, top_n=top_n)) == expected
