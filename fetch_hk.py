#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天在 GitHub Actions 里运行:并发抓取全部港股 PE / 市值 / 营收 / 净利润,
写入 data/<港时日期>.json 并更新 data/manifest.json。

效率:用线程池并发(默认 12 并发),把原来串行 ~100 分钟压到 ~10-15 分钟。
正确性保护:
  - 每次请求带重试 + 退避;
  - 单只缺市值或 PE 则跳过(不写半条);
  - 全量有效记录 < MIN_OK 视为被限流,直接报错退出、不覆盖任何文件。
并发数可用环境变量 WORKERS 调整。
"""
import json
import os
import time
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import pandas as pd
import akshare as ak

DATA_DIR = "data"
MIN_OK = 1500
WORKERS = int(os.environ.get("WORKERS", "12"))


def with_retry(func, *args, retries=4, pause=0.8, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(pause * (i + 1))
    return None


def code_list():
    df = with_retry(ak.stock_hk_spot, retries=6, pause=3.0)
    if df is None or df.empty:
        raise RuntimeError("stock_hk_spot 拉取失败")
    df = df.rename(columns={"中文名称": "名称"})[["代码", "名称"]]
    df = df.drop_duplicates("代码")
    df = df[~df["名称"].astype(str).str.contains("－Ｒ")].reset_index(drop=True)
    return df


def revenue_profit(symbol):
    df = with_retry(ak.stock_financial_hk_analysis_indicator_em,
                    symbol=symbol, indicator="年度")
    if df is None or df.empty:
        return {}
    row = df.iloc[0]
    return {
        "rev": pd.to_numeric(row.get("OPERATE_INCOME"), errors="coerce"),
        "profit": pd.to_numeric(row.get("HOLDER_PROFIT"), errors="coerce"),
        "cur": row.get("CURRENCY", ""),
    }


def last_val(symbol, indicator):
    df = with_retry(ak.stock_hk_valuation_baidu,
                    symbol=symbol, indicator=indicator, period="近一年")
    if df is None or df.empty:
        return None
    return float(df["value"].iloc[-1])


def industry_map():
    """港股代码 -> 行业(东财 push2 列表的 f100 字段,批量翻页)。失败返回空表,不影响主流程。"""
    import requests
    url = "https://72.push2.eastmoney.com/api/qt/clist/get"
    out, page = {}, 1
    while True:
        params = {
            "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f12", "fs": "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2",
            "fields": "f12,f100",
        }
        d = None
        for i in range(3):
            try:
                d = requests.get(url, params=params, timeout=15).json().get("data")
                break
            except Exception:
                time.sleep(1 + i)
        if not d or not d.get("diff"):
            break
        for row in d["diff"]:
            ind = row.get("f100")
            out[str(row.get("f12", ""))] = ind if isinstance(ind, str) and ind != "-" else ""
        if page * 100 >= d.get("total", 0):
            break
        page += 1
        time.sleep(0.2)
    return out


def fetch_one(code, name):
    """抓单只;缺市值或 PE 返回 None(跳过)。"""
    mc = last_val(code, "总市值")
    pe = last_val(code, "市盈率(TTM)")
    if mc is None or pe is None:
        return None
    rp = revenue_profit(code)
    rev, prof = rp.get("rev"), rp.get("profit")
    return {
        "code": code, "name": name,
        "pe": round(pe, 2), "mc": round(mc, 2),
        "rev": None if rev is None or pd.isna(rev) else float(rev),
        "profit": None if prof is None or pd.isna(prof) else float(prof),
        "cur": "" if pd.isna(rp.get("cur", "")) else str(rp.get("cur", "")),
    }


def main():
    today = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    codes = code_list()
    total = len(codes)
    print(f"[{today}] 待抓取 {total} 家,并发 {WORKERS}", flush=True)

    recs = []
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, str(r["代码"]), str(r["名称"])): r["代码"]
                for _, r in codes.iterrows()}
        for fut in as_completed(futs):
            rec = None
            try:
                rec = fut.result()
            except Exception:
                rec = None
            with lock:
                done += 1
                if rec:
                    recs.append(rec)
                if done % 300 == 0:
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"  {done}/{total}  有效 {len(recs)}  "
                          f"{rate:.1f}/s  已用 {int(time.time()-t0)}s", flush=True)

    print(f"有效记录 {len(recs)} 家,用时 {int(time.time()-t0)}s", flush=True)
    if len(recs) < MIN_OK:
        raise SystemExit(f"有效记录过少({len(recs)} < {MIN_OK}),疑似被限流,放弃写入")

    from gics_map import sec_g_for_hk
    inds = industry_map()
    print(f"行业映射 {len(inds)} 条", flush=True)
    for r in recs:
        r["ind"] = inds.get(r["code"], "")
        r["sec"], r["g"] = sec_g_for_hk(r["ind"], r["code"])

    recs.sort(key=lambda x: -x["mc"])
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"{today}.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)

    mpath = os.path.join(DATA_DIR, "manifest.json")
    dates = []
    if os.path.exists(mpath):
        dates = json.load(open(mpath, encoding="utf-8")).get("dates", [])
    if today not in dates:
        dates.append(today)
    dates = sorted(set(dates))
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"dates": dates}, f, ensure_ascii=False)
    print(f"已写入 {today}.json,manifest 共 {len(dates)} 天", flush=True)


if __name__ == "__main__":
    main()
