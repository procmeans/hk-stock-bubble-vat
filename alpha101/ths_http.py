"""Thin iFinD HTTP API client used by the Alpha101 tools."""

import math
import os
import re
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://quantapi.51ifind.com/api/v1"
REFRESH_TOKEN_ENV = "THS_HTTP_REFRESH_TOKEN"
_MAX_ERROR_CODE_CHARS = 32
_MAX_ERROR_CODE_INT_BITS = 107
_NUMERIC_ERROR_CODE_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
)


def get_access_token(refresh_token: Optional[str] = None, timeout: int = 15) -> str:
    """Exchange an iFinD refresh token for an access token."""
    refresh_token = refresh_token or os.getenv(REFRESH_TOKEN_ENV)
    if not refresh_token:
        raise ValueError(
            f"Missing refresh token. Pass refresh_token or set {REFRESH_TOKEN_ENV}."
        )

    response = requests.post(
        f"{BASE_URL}/get_access_token",
        headers={"Content-Type": "application/json", "refresh_token": refresh_token},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raise_for_api_error(payload)

    access_token = (payload.get("data") or {}).get("access_token")
    if not access_token:
        raise RuntimeError("iFinD HTTP API did not return access_token.")
    return access_token


def post(
    endpoint: str,
    payload: dict,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 30,
) -> dict:
    """POST to an iFinD HTTP API endpoint and return the decoded JSON."""
    access_token = access_token or get_access_token(
        refresh_token=refresh_token,
        timeout=timeout,
    )
    response = requests.post(
        f"{BASE_URL}/{endpoint.strip('/')}",
        json=payload,
        headers={"Content-Type": "application/json", "access_token": access_token},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    raise_for_api_error(data)
    return data


def real_time_quotation(
    codes,
    indicators,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Call iFinD real_time_quotation and return a flat DataFrame."""
    data = post(
        "real_time_quotation",
        {"codes": join_if_sequence(codes), "indicators": join_if_sequence(indicators)},
        access_token=access_token,
        refresh_token=refresh_token,
        timeout=timeout,
    )
    return tables_to_dataframe(data)


def history_quotation(
    codes,
    indicators,
    startdate,
    enddate,
    functionpara=None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Call iFinD cmd_history_quotation and return a flat DataFrame."""
    payload = {
        "codes": join_if_sequence(codes),
        "indicators": join_if_sequence(indicators),
        "startdate": startdate,
        "enddate": enddate,
    }
    if functionpara is not None:
        payload["functionpara"] = functionpara
    data = post(
        "cmd_history_quotation",
        payload,
        access_token=access_token,
        refresh_token=refresh_token,
        timeout=timeout,
    )
    return tables_to_dataframe(data)


def high_frequency(
    codes,
    indicators,
    starttime,
    endtime,
    functionpara=None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Call iFinD high_frequency and return a flat DataFrame."""
    payload = {
        "codes": join_if_sequence(codes),
        "indicators": join_if_sequence(indicators),
        "starttime": starttime,
        "endtime": endtime,
    }
    if functionpara is not None:
        payload["functionpara"] = functionpara
    data = post(
        "high_frequency",
        payload,
        access_token=access_token,
        refresh_token=refresh_token,
        timeout=timeout,
    )
    return tables_to_dataframe(data)


def smart_stock_picking(
    searchstring: str,
    searchtype: str = "stock",
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Run an iFinD semantic stock query and return a flat DataFrame."""
    data = post(
        "smart_stock_picking",
        {"searchstring": searchstring, "searchtype": searchtype},
        access_token=access_token,
        refresh_token=refresh_token,
        timeout=timeout,
    )
    return tables_to_dataframe(data)


def join_if_sequence(value):
    if isinstance(value, (list, tuple)):
        return ",".join(value)
    return value


def _safe_error_code(value) -> str:
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        if value.bit_length() > _MAX_ERROR_CODE_INT_BITS:
            return "unknown"
        rendered = str(value)
        return rendered if len(rendered) <= _MAX_ERROR_CODE_CHARS else "unknown"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "unknown"
        rendered = str(value)
        return rendered if len(rendered) <= _MAX_ERROR_CODE_CHARS else "unknown"
    if isinstance(value, str):
        value = value.strip()
        if (len(value) <= _MAX_ERROR_CODE_CHARS
                and _NUMERIC_ERROR_CODE_RE.fullmatch(value)):
            return value
    return "unknown"


def raise_for_api_error(payload: dict) -> None:
    errorcode = payload.get("errorcode", payload.get("errcode", 0))
    if errorcode not in (0, "0", None):
        message = payload.get("errmsg") or payload.get("message")
        if not isinstance(message, str) or not message.strip():
            message = f"iFinD HTTP API error {_safe_error_code(errorcode)}"
        raise RuntimeError(message)


def tables_to_dataframe(payload: dict) -> pd.DataFrame:
    rows = []
    for table_item in payload.get("tables", []):
        thscode = table_item.get("thscode")
        times = table_item.get("time") or []
        table = {
            name: values if isinstance(values, list) else []
            for name, values in (table_item.get("table") or {}).items()
        }
        max_len = max([len(times), *[len(values) for values in table.values()]], default=0)
        for index in range(max_len):
            row = {"thscode": thscode}
            if times:
                row["time"] = times[index] if index < len(times) else None
            for name, values in table.items():
                row[name] = values[index] if index < len(values) else None
            rows.append(row)
    return pd.DataFrame(rows)
