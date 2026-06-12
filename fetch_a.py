#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天在 GitHub Actions 里运行:抓取全部 A 股的 PE(TTM) / 总市值 / 营收 / 净利润,
写入 data/a-<日期>.json 并更新 data/manifest_a.json。

数据源均为东财数据中心(datacenter-web.eastmoney.com),全市场批量返回,
不用逐只抓,1~2 分钟即可跑完:
  - 估值(PE_TTM / 总市值): RPT_VALUEANALYSIS_DET 按最新交易日过滤翻页
  - 营收 / 净利润:          stock_yjbb_em(业绩报表,最新一期年报)

正确性保护:有效记录 < MIN_OK 视为接口异常,直接报错退出、不写任何文件。
"""
import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import akshare as ak

DATA_DIR = "data"
MIN_OK = 3000
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
ANNUAL_REPORT_DATE = "20251231"  # 最新一期年报


def dc_get(params, retries=4, pause=1.0):
    for i in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=20)
            j = r.json()
            if j.get("result"):
                return j["result"]
        except Exception:
            pass
        time.sleep(pause * (i + 1))
    return None


def latest_trade_date():
    res = dc_get({
        "sortColumns": "TRADE_DATE", "sortTypes": "-1",
        "pageSize": "1", "pageNumber": "1",
        "reportName": "RPT_VALUEANALYSIS_DET", "columns": "TRADE_DATE",
        "source": "WEB", "client": "WEB",
        "filter": '(SECURITY_CODE="000001")',
    })
    if not res:
        raise RuntimeError("拉取最新交易日失败")
    return res["data"][0]["TRADE_DATE"][:10]


def fetch_valuation(trade_date):
    rows, page = [], 1
    while True:
        res = dc_get({
            "sortColumns": "TOTAL_MARKET_CAP", "sortTypes": "-1",
            "pageSize": "500", "pageNumber": str(page),
            "reportName": "RPT_VALUEANALYSIS_DET", "columns": "ALL",
            "source": "WEB", "client": "WEB",
            "filter": f"(TRADE_DATE='{trade_date}')",
        })
        if not res or not res.get("data"):
            break
        rows.extend(res["data"])
        if page >= int(res.get("pages") or 1):
            break
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "SECURITY_CODE": "code", "SECURITY_NAME_ABBR": "name",
    })[["code", "name", "PE_TTM", "TOTAL_MARKET_CAP"]]
    df["code"] = df["code"].astype(str)
    df["mc"] = pd.to_numeric(df["TOTAL_MARKET_CAP"], errors="coerce") / 1e8  # 亿元
    df["pe"] = pd.to_numeric(df["PE_TTM"], errors="coerce")
    return df[["code", "name", "pe", "mc"]].drop_duplicates("code")


def fetch_earnings(report_date):
    df = None
    for i in range(3):
        try:
            df = ak.stock_yjbb_em(date=report_date)
            break
        except Exception:
            time.sleep(2 * (i + 1))
    if df is None or df.empty:
        print("业绩报表拉取失败,营收/净利润留空", flush=True)
        return pd.DataFrame(columns=["code", "rev", "profit"])
    df = df.rename(columns={
        "股票代码": "code",
        "营业总收入-营业总收入": "rev",
        "净利润-净利润": "profit",
        "所处行业": "ind",
    })[["code", "rev", "profit", "ind"]]
    df["code"] = df["code"].astype(str)
    # 同一代码多行时优先保留行业非空的
    df = df.sort_values("ind", na_position="last")
    return df.drop_duplicates("code")


def main():
    today = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    td = latest_trade_date()
    print(f"[{today}] A股最新交易日:{td}", flush=True)
    val = fetch_valuation(td)
    earn = fetch_earnings(ANNUAL_REPORT_DATE)
    out = val.merge(earn, on="code", how="left")
    out = out.dropna(subset=["pe", "mc"])
    print(f"估值 {len(val)} 只,合并业绩后有效 {len(out)} 只", flush=True)
    if len(out) < MIN_OK:
        raise SystemExit(f"有效记录过少({len(out)} < {MIN_OK}),疑似接口异常,放弃写入")

    recs = []
    for _, r in out.sort_values("mc", ascending=False).iterrows():
        recs.append({
            "code": r["code"], "name": str(r["name"]),
            "pe": round(float(r["pe"]), 2), "mc": round(float(r["mc"]), 2),
            "rev": None if pd.isna(r["rev"]) else float(r["rev"]),
            "profit": None if pd.isna(r["profit"]) else float(r["profit"]),
            "cur": "CNY",
            "ind": "" if pd.isna(r.get("ind")) else str(r.get("ind")),
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"a-{today}.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)

    mpath = os.path.join(DATA_DIR, "manifest_a.json")
    dates = []
    if os.path.exists(mpath):
        dates = json.load(open(mpath, encoding="utf-8")).get("dates", [])
    if today not in dates:
        dates.append(today)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"dates": sorted(set(dates))}, f, ensure_ascii=False)
    print(f"已写入 a-{today}.json,manifest_a 共 {len(dates)} 天", flush=True)


if __name__ == "__main__":
    main()
