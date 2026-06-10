#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天在 GitHub Actions 里运行:抓取全部港股 PE / 市值 / 营收 / 净利润,
写入 data/<港时日期>.json,并更新 data/manifest.json。

为安全起见:若抓到的有效记录 < MIN_OK,直接报错退出、不覆盖任何文件,
避免某天接口被限流导致网页数据被一份空数据冲掉。
"""
import json
import os
import time
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import akshare as ak

DATA_DIR = "data"
MIN_OK = 1500  # 低于这个数量视为抓取失败


def with_retry(func, *args, retries=3, pause=1.0, **kwargs):
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
    # 去掉「－Ｒ」人民币柜台(与港币柜台同公司)
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


def main():
    today = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    codes = code_list()
    print(f"[{today}] 待抓取 {len(codes)} 家", flush=True)

    recs = []
    for i, (_, r) in enumerate(codes.iterrows(), 1):
        code, name = str(r["代码"]), str(r["名称"])
        rp = revenue_profit(code)
        mc = last_val(code, "总市值")
        pe = last_val(code, "市盈率(TTM)")
        if mc is None or pe is None:
            continue
        rev = rp.get("rev")
        prof = rp.get("profit")
        recs.append({
            "code": code, "name": name,
            "pe": round(pe, 2), "mc": round(mc, 2),
            "rev": None if rev is None or pd.isna(rev) else float(rev),
            "profit": None if prof is None or pd.isna(prof) else float(prof),
            "cur": "" if pd.isna(rp.get("cur", "")) else str(rp.get("cur", "")),
        })
        if i % 200 == 0:
            print(f"  {i}/{len(codes)}  已收集 {len(recs)}", flush=True)
        time.sleep(0.3)

    print(f"有效记录 {len(recs)} 家", flush=True)
    if len(recs) < MIN_OK:
        raise SystemExit(f"有效记录过少({len(recs)} < {MIN_OK}),疑似被限流,放弃写入")

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
