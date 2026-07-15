"""同花顺个股热度与热度排名上升单因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101 import ths_http

STRATEGIES = {"ths_heat", "ths_heat_rise"}
SIGNAL_COLUMNS = ["date", "strategy", "rank", "ticker", "name", "factor_value"]
VALUE_PREFIX = {
    "ths_heat": "个股热度",
    "ths_heat_rise": "个股热度排名环比增长率",
}


def _day(signal_date) -> pd.Timestamp:
    return pd.Timestamp(signal_date).normalize()


def build_query(signal_date, strategy: str, top_n: int) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported THS heat strategy: {strategy}")
    day = _day(signal_date)
    return f"{day.year}年{day.month}月{day.day}日{VALUE_PREFIX[strategy]}排名前{int(top_n)}"


def _value_column(data: pd.DataFrame, signal_date, strategy: str) -> str:
    stamp = _day(signal_date).strftime("%Y%m%d")
    expected = f"{VALUE_PREFIX[strategy]}[{stamp}]"
    if expected not in data.columns:
        raise ValueError(f"missing {VALUE_PREFIX[strategy]} column for {stamp}")
    return expected


def normalize_signal(data, signal_date, strategy: str, top_n: int = 20):
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported THS heat strategy: {strategy}")
    if "股票代码" not in data.columns:
        raise ValueError("missing 股票代码 column")
    value_column = _value_column(data, signal_date, strategy)
    result = pd.DataFrame({
        "ticker": data["股票代码"].astype(str).str.extract(r"(\d{6})", expand=False),
        "name": data["股票简称"].astype(str)
        if "股票简称" in data.columns else data["股票代码"].astype(str),
        "factor_value": pd.to_numeric(data[value_column], errors="coerce"),
    }).dropna(subset=["ticker", "factor_value"])
    result = result[np.isfinite(result["factor_value"])]
    result = result.drop_duplicates("ticker").sort_values(
        ["factor_value", "ticker"], ascending=[False, True]
    ).head(int(top_n)).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"empty {strategy} signal for {_day(signal_date).date()}")
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "strategy", strategy)
    result.insert(0, "date", _day(signal_date).strftime("%Y-%m-%d"))
    return result[SIGNAL_COLUMNS]


def fetch_signal(signal_date, strategy: str, top_n: int = 20, access_token=None):
    raw = ths_http.smart_stock_picking(
        build_query(signal_date, strategy, top_n), access_token=access_token
    )
    return normalize_signal(raw, signal_date, strategy, top_n=top_n)


def target_weights(signal: pd.DataFrame, prices: pd.Series) -> dict[str, float]:
    tickers = signal["ticker"].astype(str).tolist()
    quoted = pd.to_numeric(prices.reindex(tickers), errors="coerce")
    valid = [ticker for ticker, price in quoted.items()
             if pd.notna(price) and np.isfinite(price) and price > 0]
    if not valid:
        return {}
    return {ticker: 1.0 / len(valid) for ticker in valid}
