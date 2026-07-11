"""业绩预告事件驱动(PEAD):事件抓取、事件研究(CAR)、持有信号。

数据源:东方财富数据中心 RPT_PUBLIC_OP_NEWPREDICT(免费,无鉴权)。
口径:信号日 = 首个交易日 ≥ 公告日(业绩预告多为盘后发布),回测层再
shift(1) 次日成交;未剔除一字板无法成交的情形,纸面结果偏乐观。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("alpha101/cache/events_yjyg.csv")
API = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
       "?reportName=RPT_PUBLIC_OP_NEWPREDICT"
       "&columns=SECURITY_CODE,NOTICE_DATE,REPORT_DATE,PREDICT_TYPE,"
       "ADD_AMP_LOWER,ADD_AMP_UPPER"
       "&pageSize=500&pageNumber={page}&sortColumns=NOTICE_DATE&sortTypes=-1"
       "&filter=(REPORT_DATE%3D%27{report}%27)")
POSITIVE = ("预增", "扭亏")


def parse_forecast_page(payload: dict) -> list:
    result = payload.get("result") or {}
    rows = []
    for item in result.get("data") or []:
        rows.append({
            "code": str(item["SECURITY_CODE"]).zfill(6),
            "notice_date": str(item["NOTICE_DATE"])[:10],
            "report_date": str(item["REPORT_DATE"])[:10],
            "type": item["PREDICT_TYPE"],
            "amp_lower": item.get("ADD_AMP_LOWER"),
            "amp_upper": item.get("ADD_AMP_UPPER"),
        })
    return rows


def quarter_ends(start: str) -> list:
    ends = pd.date_range(start, pd.Timestamp.today(), freq="QE")
    return [d.strftime("%Y-%m-%d") for d in ends]


def _get_json(url: str, retries: int = 4, timeout: int = 30) -> dict:
    import requests

    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout).json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_events(start: str = "2022-06-30", pause: float = 0.5) -> pd.DataFrame:
    rows = []
    for report in quarter_ends(start):
        page = 1
        while True:
            payload = _get_json(API.format(page=page, report=report))
            rows.extend(parse_forecast_page(payload))
            pages = (payload.get("result") or {}).get("pages") or 1
            if page >= pages:
                break
            page += 1
            time.sleep(pause)
        print(f"{report}: 累计 {len(rows)} 条", flush=True)
    frame = pd.DataFrame(rows).drop_duplicates(["code", "report_date", "notice_date"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    return frame


def load_events() -> pd.DataFrame:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"事件缓存不存在 {CACHE},先运行: python -m strategies.events fetch")
    return pd.read_csv(CACHE, dtype={"code": str})


def signal(panel, events=None, hold=20, positive=POSITIVE):
    close = panel["close"]
    events = load_events() if events is None else events
    chosen = events[events["type"].isin(positive)]
    idx = close.index
    active = pd.DataFrame(0.0, index=idx, columns=close.columns)
    for code, notice in zip(chosen["code"], pd.to_datetime(chosen["notice_date"])):
        if code not in active.columns:
            continue
        pos = idx.searchsorted(notice)
        if pos >= len(idx):
            continue
        column = active.columns.get_loc(code)
        active.iloc[pos:pos + hold, column] = 1.0
    counts = active.sum(axis=1)
    return active.div(counts.replace(0, np.nan), axis=0).fillna(0.0)


def car(panel, events=None, pre=5, post=20, min_events=5):
    """事件研究:各预告类型的平均累计异常收益与事件后 CAR t 值。"""
    rets = panel["close"].pct_change()
    abnormal = rets.sub(rets.mean(axis=1), axis=0)
    idx = rets.index
    events = load_events() if events is None else events
    rows = {}
    for kind, group in events.groupby("type"):
        windows = []
        for code, notice in zip(group["code"], pd.to_datetime(group["notice_date"])):
            if code not in abnormal.columns:
                continue
            pos = idx.searchsorted(notice)
            if pos - pre < 1 or pos + post >= len(idx):
                continue
            window = abnormal[code].iloc[pos - pre: pos + post + 1].to_numpy()
            if np.isnan(window).any():
                continue
            windows.append(window)
        if len(windows) < min_events:
            continue
        arr = np.array(windows)
        car_each = arr[:, pre + 1:].sum(axis=1)          # 事件后 [1, post] 累计
        sem = car_each.std(ddof=0) / np.sqrt(len(car_each))
        rows[kind] = {
            "n_events": len(windows),
            "car_pre": float(arr[:, :pre].sum(axis=1).mean()),
            "car_day0": float(arr[:, pre].mean()),
            "car_post": float(car_each.mean()),
            "t_post": float(car_each.mean() / sem) if sem > 0 else np.nan,
        }
    if not rows:
        return pd.DataFrame(
            columns=["n_events", "car_pre", "car_day0", "car_post", "t_post"])
    return pd.DataFrame(rows).T.sort_values("car_post", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["fetch", "study"])
    parser.add_argument("--start", default="2022-06-30")
    parser.add_argument("--top", type=int, default=None,
                        help="study 用面板的流动性截断(默认全市场)")
    parser.add_argument("--post", type=int, default=20)
    args = parser.parse_args()
    if args.cmd == "fetch":
        frame = fetch_events(args.start)
        print(f"共 {len(frame)} 条,类型分布:")
        print(frame["type"].value_counts().to_string())
    else:
        from strategies import data
        panel = data.load_panel("a", top=args.top)
        table = car(panel, post=args.post)
        print(f"事件研究(异常收益=个股−等权基准,事件后 {args.post} 日):")
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()
