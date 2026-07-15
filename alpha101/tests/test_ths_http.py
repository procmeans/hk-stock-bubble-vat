import pandas as pd
import pytest

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


def test_api_error_without_message_never_stringifies_secret_payload():
    secret = "refresh-secret-that-must-not-escape"
    payload = {
        "errorcode": 401,
        "data": {"refresh_token": secret, "access_token": "access-secret"},
    }

    with pytest.raises(RuntimeError) as caught:
        ths_http.raise_for_api_error(payload)

    assert str(caught.value) == "iFinD HTTP API error 401"
    assert secret not in str(caught.value)


def test_api_error_never_stringifies_nested_credential_bearing_error_code():
    secret = "nested-refresh-secret"
    payload = {
        "errorcode": {"code": 401, "refresh_token": secret},
        "data": {"reason": "malformed response"},
    }

    with pytest.raises(RuntimeError) as caught:
        ths_http.raise_for_api_error(payload)

    assert str(caught.value) == "iFinD HTTP API error unknown"
    assert secret not in str(caught.value)


@pytest.mark.parametrize(("errorcode", "rendered"), [
    (10 ** 5000, "unknown"),
    ("9" * 40, "unknown"),
    ("429", "429"),
], ids=["large-integer", "overlong-numeric-string", "short-numeric-string"])
def test_api_error_code_rendering_is_total_and_bounded(errorcode, rendered):
    with pytest.raises(RuntimeError) as caught:
        ths_http.raise_for_api_error({"errorcode": errorcode})

    assert str(caught.value) == f"iFinD HTTP API error {rendered}"
