import pandas as pd

from alpha101 import ths_http


def test_smart_stock_picking_posts_query_and_flattens(monkeypatch):
    seen = {}

    def fake_post(endpoint, payload, **kwargs):
        seen.update(endpoint=endpoint, payload=payload, kwargs=kwargs)
        return {"tables": [{"table": {
            "股票代码": ["000001.SZ"],
            "个股热度[20260715]": [123.5],
        }}]}

    monkeypatch.setattr(ths_http, "post", fake_post)
    result = ths_http.smart_stock_picking(
        "2026年7月15日个股热度排名前20", access_token="access", timeout=9
    )

    assert seen == {
        "endpoint": "smart_stock_picking",
        "payload": {
            "searchstring": "2026年7月15日个股热度排名前20",
            "searchtype": "stock",
        },
        "kwargs": {"access_token": "access", "refresh_token": None, "timeout": 9},
    }
    assert isinstance(result, pd.DataFrame)
    assert result.loc[0, "股票代码"] == "000001.SZ"
    assert result.loc[0, "个股热度[20260715]"] == 123.5
