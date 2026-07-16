import pandas as pd
import pytest

from alpha101 import ths_http


def test_history_quotation_includes_functionpara(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ths_http,
        "post",
        lambda endpoint, payload, **kwargs: seen.update(
            endpoint=endpoint, payload=payload
        ) or {"tables": []},
    )

    ths_http.history_quotation(
        ["000001.SZ"],
        ["open", "close"],
        "2026-01-01",
        "2026-01-31",
        functionpara={"CPS": "3", "Fill": "Omit"},
        access_token="token",
    )

    assert seen == {
        "endpoint": "cmd_history_quotation",
        "payload": {
            "codes": "000001.SZ",
            "indicators": "open,close",
            "startdate": "2026-01-01",
            "enddate": "2026-01-31",
            "functionpara": {"CPS": "3", "Fill": "Omit"},
        },
    }


def test_high_frequency_posts_and_flattens(monkeypatch):
    seen = {}

    def fake_post(endpoint, payload, **kwargs):
        seen.update(endpoint=endpoint, payload=payload)
        return {
            "tables": [{
                "thscode": "000001.SZ",
                "time": ["2026-01-12 09:30"],
                "table": {
                    "close": [10.0],
                    "volume": [100.0],
                    "amount": [1000.0],
                },
            }]
        }

    monkeypatch.setattr(ths_http, "post", fake_post)
    result = ths_http.high_frequency(
        ["000001.SZ"],
        ["close", "volume", "amount"],
        "2026-01-12 09:30:00",
        "2026-01-12 15:00:00",
        functionpara={"Fill": "Original", "Timeformat": "LocalTime"},
    )

    assert seen["endpoint"] == "high_frequency"
    assert seen["payload"]["functionpara"]["Fill"] == "Original"
    assert result.loc[0, "amount"] == 1000.0


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
