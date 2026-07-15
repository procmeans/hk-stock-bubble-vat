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
