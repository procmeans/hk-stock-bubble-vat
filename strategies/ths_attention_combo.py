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


FACTOR_COLUMNS = CANDIDATE_COLUMNS + [
    "momentum_7d", "float_ratio", "attention_pct",
    "momentum_pct", "low_float_pct",
]


def factor_frame(candidates, close: pd.DataFrame, min_history: int = 60):
    if len(close.index) < 8:
        return pd.DataFrame(columns=FACTOR_COLUMNS)
    base = candidates.drop_duplicates("ticker").copy()
    base = base[~base["name"].astype(str).str.match(
        r"^\*?ST", case=False, na=False
    )]
    tickers = base["ticker"].astype(str).tolist()
    history = close.reindex(columns=tickers).apply(
        pd.to_numeric, errors="coerce"
    )
    current = pd.to_numeric(history.iloc[-1], errors="coerce")
    past = pd.to_numeric(history.iloc[-8], errors="coerce")
    valid_history = history.where(np.isfinite(history) & (history > 0))
    valid_count = valid_history.notna().sum()
    base = base.set_index("ticker")
    base["momentum_7d"] = current / past - 1.0
    base["float_ratio"] = base["float_a"] / base["total_shares"]
    finite = np.isfinite(base[[
        "attention_rise", "float_a", "total_shares",
        "momentum_7d", "float_ratio",
    ]]).all(axis=1)
    valid = (
        finite
        & (valid_count.reindex(base.index).fillna(0) >= int(min_history))
        & (current.reindex(base.index) > 0)
        & (past.reindex(base.index) > 0)
        & (base["float_a"] > 0)
        & (base["total_shares"] > 0)
        & (base["float_ratio"] > 0)
        & (base["float_ratio"] <= 1)
    )
    result = base.loc[valid].reset_index()
    if result.empty:
        return pd.DataFrame(columns=FACTOR_COLUMNS)
    result["attention_pct"] = result["attention_rise"].rank(
        method="average", pct=True, ascending=True
    )
    result["momentum_pct"] = result["momentum_7d"].rank(
        method="average", pct=True, ascending=True
    )
    result["low_float_pct"] = result["float_ratio"].rank(
        method="average", pct=True, ascending=False
    )
    return result.sort_values("ticker").reset_index(drop=True)[FACTOR_COLUMNS]


SELECTED_COLUMNS = ["strategy", "rank", *FACTOR_COLUMNS, "score"]


def _ranked(frame, strategy: str, top_n: int, score) -> pd.DataFrame:
    limit = min(max(int(top_n), 0), 20)
    selected = frame.copy()
    selected["score"] = score.reindex(selected.index)
    selected = selected.sort_values(
        ["score", "attention_rise", "ticker"],
        ascending=[False, False, True],
    ).head(limit).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected.insert(0, "strategy", strategy)
    return selected[SELECTED_COLUMNS]


def select_weighted(factors, top_n: int = 20):
    score = (
        0.50 * factors["attention_pct"]
        + 0.30 * factors["momentum_pct"]
        + 0.20 * factors["low_float_pct"]
    )
    return _ranked(factors, "ths_attention_weighted", top_n, score)


def select_funnel(factors, top_n: int = 20):
    import math

    positive = factors[factors["momentum_7d"] > 0].sort_values(
        ["float_ratio", "ticker"], ascending=[True, True]
    )
    kept = positive.head(math.ceil(len(positive) / 2)).copy()
    return _ranked(
        kept, "ths_attention_funnel", top_n, kept["attention_rise"]
    )


def target_weights(selected, prices: pd.Series) -> dict[str, float]:
    tickers = selected["ticker"].astype(str).tolist()
    quoted = pd.to_numeric(prices.reindex(tickers), errors="coerce")
    valid = [ticker for ticker, price in quoted.items()
             if pd.notna(price) and np.isfinite(price) and price > 0]
    if not valid:
        return {}
    return {ticker: 1.0 / len(valid) for ticker in valid}
