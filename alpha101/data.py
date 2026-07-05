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
    # 成交量(volume)单位为手(1 手 = 100 股),成交额(amount)单位为元,
    # 故每股 vwap = amount / (volume * 100),否则会放大约 100 倍。
    panel["vwap"] = panel["amount"] / (panel["volume"].replace(0, np.nan) * 100)
    panel["returns"] = panel["close"].pct_change()
    return panel


def adv(panel, d):
    return panel["amount"].rolling(int(d)).mean()


def _fetch_one(code, start, end, retries=3, pause=0.5):
    """抓单只前复权日线;失败重试 retries 次。空数据(退市/无行情)不重试。"""
    import akshare as ak
    for k in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty:
                return None
            df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                    "最低": "low", "收盘": "close", "成交量": "volume",
                                    "成交额": "amount"})
            df["code"] = code
            return df[["code", "date"] + FIELDS]
        except Exception:
            time.sleep(pause * (k + 1))
    return None


def _all_codes(retries=4, pause=1.0):
    """全市场代码列表(东财快照),失败重试。"""
    import akshare as ak
    for k in range(retries):
        try:
            return ak.stock_zh_a_spot_em()["代码"].astype(str).tolist()
        except Exception:
            time.sleep(pause * (k + 1))
    raise RuntimeError("获取全市场代码失败(东财快照接口异常/被限流)")


def _pending_codes(all_codes, cached_codes):
    """尚未抓取的代码,保序,断点续传用。"""
    cached = set(cached_codes)
    return [c for c in all_codes if c not in cached]


def fetch_all(years=5, cache=CACHE, sleep=0.12, checkpoint_every=200):
    """全 A 股前复权日线 -> parquet。带重试、每 checkpoint_every 只存盘、断点续传:
    中途中断后重跑会读取已有 cache、跳过已抓代码、继续。若要全量重抓,先删除 cache 文件。"""
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    codes = _all_codes()
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    frames, done = [], []
    if os.path.exists(cache):
        prev = pd.read_parquet(cache)
        frames.append(prev)
        done = prev["code"].astype(str).unique().tolist()
        print(f"断点续传:已有 {len(done)} 只,跳过", flush=True)

    todo = _pending_codes(codes, done)
    print(f"全市场 {len(codes)} 只,待抓 {len(todo)} 只,区间 {start}~{end}", flush=True)
    t0, got = time.time(), 0
    for i, code in enumerate(todo):
        one = _fetch_one(code, start, end)
        if one is not None:
            frames.append(one)
            got += 1
        if (i + 1) % checkpoint_every == 0:
            pd.concat(frames, ignore_index=True).to_parquet(cache)
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"{i+1}/{len(todo)} 抓到 {got},已存盘,用时 {el:.0f}s ETA {eta:.0f}s", flush=True)
        time.sleep(sleep)

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    raw.to_parquet(cache)
    print(f"完成:{raw['code'].nunique()} 只 × {raw['date'].nunique()} 交易日", flush=True)
    return build_panel(raw)


def load_panel(cache=CACHE):
    raw = pd.read_parquet(cache)
    return build_panel(raw)
