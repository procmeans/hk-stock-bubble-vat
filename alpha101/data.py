"""A股前复权日线数据:拉取、缓存、构建字段面板。"""
import os
import time
import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
CACHE = "alpha101/cache/panel.parquet"


def build_panel(raw):
    """长表(code,date,open,high,low,close,volume,amount) -> 字段面板 dict。"""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    panel = {}
    for f in FIELDS:
        panel[f] = raw.pivot(index="date", columns="code", values=f).sort_index()
    panel["vwap"] = panel["amount"] / panel["volume"].replace(0, np.nan)
    panel["returns"] = panel["close"].pct_change()
    return panel


def adv(panel, d):
    return panel["amount"].rolling(int(d)).mean()


def _fetch_one(code, start, end):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close", "成交量": "volume",
                            "成交额": "amount"})
    df["code"] = code
    return df[["code", "date"] + FIELDS]


def fetch_all(years=5, cache=CACHE, sleep=0.05):
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    codes = ak.stock_zh_a_spot_em()["代码"].astype(str).tolist()
    frames = []
    for i, code in enumerate(codes):
        try:
            one = _fetch_one(code, start, end)
            if one is not None:
                frames.append(one)
        except Exception as e:
            print(f"skip {code}: {e}", flush=True)
        if i % 200 == 0:
            print(f"{i}/{len(codes)}", flush=True)
        time.sleep(sleep)
    raw = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    raw.to_parquet(cache)
    return build_panel(raw)


def load_panel(cache=CACHE):
    raw = pd.read_parquet(cache)
    return build_panel(raw)
