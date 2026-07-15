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
