import pandas as pd
import pytest

from strategies import data


def _raw(codes, days=3):
    rows = []
    for i, code in enumerate(codes):
        for d in range(days):
            price = 10.0 * (i + 1)
            rows.append({
                "code": code, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "open": price, "high": price, "low": price, "close": price,
                "volume": 100.0 * (i + 1),
            })
    return pd.DataFrame(rows)


def test_load_panel_us_from_cache(tmp_path):
    cache = tmp_path / "yf_panel_us.pkl"
    _raw(["NVDA", "AAPL"]).to_pickle(cache)

    panel = data.load_panel("us", cache=cache)

    assert set(panel["close"].columns) == {"NVDA", "AAPL"}
    assert "vwap" in panel and "amount" in panel


def test_load_panel_top_filters_by_amount(tmp_path):
    cache = tmp_path / "yf_panel_us.pkl"
    _raw(["SMALL", "BIG"]).to_pickle(cache)

    panel = data.load_panel("us", top=1, cache=cache)

    assert list(panel["close"].columns) == ["BIG"]   # 成交额更大的留下


def test_load_panel_missing_cache_hints_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="yf_history fetch --market hk"):
        data.load_panel("hk", cache=tmp_path / "nope.pkl")
