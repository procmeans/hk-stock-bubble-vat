"""同花顺注意力上升、7日动量与低流通比例组合策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101 import ths_http

STRATEGIES = {"ths_attention_weighted", "ths_attention_funnel"}
CANDIDATE_COLUMNS = [
    "date", "ticker", "name", "attention_rise", "float_a", "total_shares",
]


def _day(signal_date) -> pd.Timestamp:
    return pd.Timestamp(signal_date).normalize()


def build_query(signal_date, candidate_n: int = 100) -> str:
    day = _day(signal_date)
    prefix = f"{day.year}年{day.month}月{day.day}日"
    return (
        f"{prefix}个股热度排名环比增长率排名前{int(candidate_n)}，"
        f"{prefix}流通A股，{prefix}总股本"
    )


def _dated_column(data: pd.DataFrame, prefix: str, signal_date) -> str:
    expected = f"{prefix}[{_day(signal_date).strftime('%Y%m%d')}]"
    matches = [column for column in data.columns
               if str(column).lower() == expected.lower()]
    if not matches:
        raise ValueError(f"missing {expected} column")
    return matches[0]


def normalize_candidates(data, signal_date, candidate_n: int = 100):
    for column in ("股票代码", "股票简称"):
        if column not in data.columns:
            raise ValueError(f"missing {column} column")
    attention = _dated_column(data, "个股热度排名环比增长率", signal_date)
    float_a = _dated_column(data, "流通a股", signal_date)
    total = _dated_column(data, "总股本", signal_date)
    result = pd.DataFrame({
        "ticker": data["股票代码"].astype(str).str.extract(
            r"(\d{6})", expand=False
        ),
        "name": data["股票简称"].astype(str),
        "attention_rise": pd.to_numeric(data[attention], errors="coerce"),
        "float_a": pd.to_numeric(data[float_a], errors="coerce"),
        "total_shares": pd.to_numeric(data[total], errors="coerce"),
    }).dropna(subset=["ticker"])
    result = result.drop_duplicates("ticker").sort_values(
        ["attention_rise", "ticker"], ascending=[False, True], na_position="last"
    ).head(int(candidate_n)).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"empty attention candidates for {_day(signal_date).date()}")
    result.insert(0, "date", _day(signal_date).strftime("%Y-%m-%d"))
    return result[CANDIDATE_COLUMNS]


def fetch_candidates(signal_date, candidate_n: int = 100, access_token=None):
    raw = ths_http.smart_stock_picking(
        build_query(signal_date, candidate_n), access_token=access_token
    )
    return normalize_candidates(raw, signal_date, candidate_n=candidate_n)
