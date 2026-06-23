#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天在 GitHub Actions 里运行:抓取全部美股的 PE(TTM) / 总市值(美元),
写入 data/us-<日期>.json 并更新 data/manifest_us.json。

数据源:东财 push2 行情列表(NASDAQ/NYSE/AMEX 全量翻页,每页 100,约 135 页)。
字段:f12=代码 f14=名称(中文) f20=总市值(USD) f115=PE(TTM)
营收/净利润 push2 无可靠字段,留空。

过滤:总市值 < MIN_MC 美元的微型股丢弃(否则 1.3 万只气泡渲染太重)。
正确性保护:有效记录 < MIN_OK 视为接口异常,直接报错退出、不写任何文件。
"""
import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

import requests

from gics_map import sec_g_for_us

DATA_DIR = "data"
MIN_OK = 3000
MIN_MC = 1e8  # 1 亿美元以下的微型股不要
URL = "https://72.push2.eastmoney.com/api/qt/clist/get"


UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def fetch_page(page, retries=6, pause=1.5):
    params = {
        "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f20", "fs": "m:105,m:106,m:107", "fields": "f12,f14,f20,f115,f100",
    }
    for i in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=20,
                             headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})
            d = r.json().get("data")
            if d is not None:
                return d
        except Exception:
            pass
        time.sleep(pause * (i + 1))
    return None


def nasdaq_industry_map():
    """纳斯达克官方 screener:全美股 symbol -> 细分行业(英文),一次请求。失败返回空表。"""
    for i in range(4):
        try:
            r = requests.get(
                "https://api.nasdaq.com/api/screener/stocks",
                params={"tableonly": "true", "limit": 25, "download": "true"},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                         "Accept": "application/json, text/plain, */*",
                         "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"},
                timeout=30)
            rows = r.json()["data"]["rows"]
            return {row["symbol"]: (row.get("industry") or "").strip() for row in rows}
        except Exception:
            time.sleep(2 * (i + 1))
    print("纳斯达克行业表拉取失败,二级行业按大类缺省值回退", flush=True)
    return {}


def main():
    today = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    nas = nasdaq_industry_map()
    print(f"纳斯达克行业表 {len(nas)} 条", flush=True)
    recs, seen = [], set()
    page, total = 1, None
    fails = 0
    while True:
        d = fetch_page(page)
        if d is None:
            # 单页拉取失败(限流/网络)不要直接放弃整轮翻页:跳过本页继续,
            # 累计失败过多才停。列表按市值降序,跳过中段页只损失少量中小盘。
            fails += 1
            print(f"  第 {page} 页拉取失败(累计 {fails}),跳过继续", flush=True)
            if fails > 12:
                print("  累计失败过多,提前结束翻页", flush=True)
                break
            if total and page * 100 >= total:
                break
            page += 1
            time.sleep(1.5)
            continue
        if not d.get("diff"):
            break
        total = total or d.get("total", 0)
        for row in d["diff"]:
            code, name = str(row.get("f12", "")), str(row.get("f14", ""))
            mc, pe = row.get("f20"), row.get("f115")
            if not code or code in seen:
                continue
            if not isinstance(mc, (int, float)) or mc < MIN_MC:
                continue
            if not isinstance(pe, (int, float)):
                continue
            seen.add(code)
            em_sec = row.get("f100")
            em_sec = em_sec if isinstance(em_sec, str) and em_sec != "-" else ""
            # 纳斯达克的 A/B 类股代码用 . 或 /(东财用 _),做变体匹配
            nind = nas.get(code) or nas.get(code.replace("_", ".")) or nas.get(code.replace("_", "/")) or ""
            sec, g = sec_g_for_us(em_sec, nind, code)
            recs.append({
                "code": code, "name": name,
                "pe": round(float(pe), 2), "mc": round(mc / 1e8, 2),  # 亿美元
                "rev": None, "profit": None, "cur": "USD",
                "ind": nind, "sec": sec, "g": g,
            })
        if page % 20 == 0:
            print(f"  第 {page} 页,累计有效 {len(recs)}", flush=True)
        if page * 100 >= (total or 0):
            break
        page += 1
        time.sleep(0.2)

    print(f"[{today}] 美股总数 {total},市值≥{MIN_MC/1e8:.0f}亿美元且有 PE 的 {len(recs)} 只", flush=True)
    if len(recs) < MIN_OK:
        raise SystemExit(f"有效记录过少({len(recs)} < {MIN_OK}),疑似接口异常,放弃写入")

    recs.sort(key=lambda x: -x["mc"])
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"us-{today}.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)

    mpath = os.path.join(DATA_DIR, "manifest_us.json")
    dates = []
    if os.path.exists(mpath):
        dates = json.load(open(mpath, encoding="utf-8")).get("dates", [])
    if today not in dates:
        dates.append(today)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"dates": sorted(set(dates))}, f, ensure_ascii=False)
    print(f"已写入 us-{today}.json,manifest_us 共 {len(dates)} 天", flush=True)


if __name__ == "__main__":
    # 单个市场抓取失败(限流/网络/接口异常)只跳过本市场、保留上次快照,
    # 以 exit 0 退出,绝不阻断同一 Actions 作业里其它市场的写入与提交推送。
    import sys
    try:
        main()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        print(f"⚠ 美股抓取未完成,跳过、保留上次快照:{e}", flush=True)
        sys.exit(0)
