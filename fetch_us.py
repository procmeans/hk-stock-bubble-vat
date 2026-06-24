#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每天在 GitHub Actions 里运行:抓取全部美股的 PE / 总市值(美元),
写入 data/us-<日期>.json 并更新 data/manifest_us.json。

数据源(主):stockanalysis.com 筛选器的 SSR 数据(SvelteKit devalue 格式),
  一次请求 __data.json 拿全美股 symbol/name/marketCap(USD)/peRatio。
  无需 key、无分页,GitHub runner 上不被限流——取代了会限流的东财 push2。
数据源(兜底):东财 push2 行情列表翻页(f20=市值 f115=PE),仅当主源异常时启用。
行业:用纳斯达克官方 screener 的 symbol→细分行业表,经 gics_map 映射到 GICS 二级。

过滤:总市值 < MIN_MC 美元的微型股丢弃。无 PE / 负 PE(SPAC、亏损股)照收,
  pe 存为 null/负值——前端会把它们沉到缸底沉淀区(尤其大市值的不能丢)。
正确性保护:有效记录 < MIN_OK 视为接口异常,跳过写入、保留上次快照(入口兜底 exit 0)。
"""
import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

import requests

from gics_map import sec_g_for_us

DATA_DIR = "data"
MIN_OK = 1500       # 主源正常给 ~2700 只;低于此视为异常
MIN_MC = 1e8        # 1 亿美元以下的微型股不要
SA_MIN_RAW = 2000   # stockanalysis 解析出的总行数低于此 → 判定主源异常,回退 push2

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SA_URL = "https://stockanalysis.com/stocks/screener/__data.json"
PUSH2_URL = "https://72.push2.eastmoney.com/api/qt/clist/get"


def fetch_stockanalysis(retries=4, pause=2.0):
    """主源:一次请求拿全美股。返回 [{code,name,mc(USD),pe,em_sec}],失败返回 []。

    __data.json 是 SvelteKit 的 devalue 扁平格式:一个池子 list,每只股票是
    {"s":i,"n":j,"marketCap":k,...},字段值都是指向池子的下标。
    """
    headers = {"User-Agent": UA, "Accept": "application/json",
               "Referer": "https://stockanalysis.com/stocks/screener/"}
    for i in range(retries):
        try:
            r = requests.get(SA_URL, headers=headers, timeout=30)
            payload = r.json()
            for node in payload.get("nodes", []):
                if not isinstance(node, dict) or node.get("type") != "data":
                    continue
                pool = node.get("data")
                if not isinstance(pool, list):
                    continue

                def deref(idx):
                    return pool[idx] if isinstance(idx, int) and 0 <= idx < len(pool) else None

                rows = []
                for el in pool:
                    if isinstance(el, dict) and "s" in el and "marketCap" in el and "peRatio" in el:
                        sym, mc = deref(el["s"]), deref(el["marketCap"])
                        pe, name = deref(el["peRatio"]), deref(el.get("n", -1))
                        if isinstance(sym, str) and isinstance(mc, (int, float)):
                            rows.append({
                                "code": sym, "name": name if isinstance(name, str) else sym,
                                "mc": float(mc),
                                "pe": float(pe) if isinstance(pe, (int, float)) else None,
                                "em_sec": "",
                            })
                if rows:
                    return rows
        except Exception:
            pass
        time.sleep(pause * (i + 1))
    return []


def _push2_page(page, retries=6, pause=1.5):
    params = {
        "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f20", "fs": "m:105,m:106,m:107", "fields": "f12,f14,f20,f115,f100",
    }
    for i in range(retries):
        try:
            r = requests.get(PUSH2_URL, params=params, timeout=20,
                             headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})
            d = r.json().get("data")
            if d is not None:
                return d
        except Exception:
            pass
        time.sleep(pause * (i + 1))
    return None


def fetch_push2():
    """兜底源:东财 push2 翻页。返回 [{code,name,mc(USD),pe,em_sec}]。"""
    rows, seen = [], set()
    page, total, fails = 1, None, 0
    while True:
        d = _push2_page(page)
        if d is None:
            fails += 1
            print(f"  push2 第 {page} 页失败(累计 {fails}),跳过继续", flush=True)
            if fails > 12:
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
            code = str(row.get("f12", ""))
            if not code or code in seen:
                continue
            seen.add(code)
            em_sec = row.get("f100")
            rows.append({
                "code": code, "name": str(row.get("f14", "")),
                "mc": row.get("f20"), "pe": row.get("f115"),
                "em_sec": em_sec if isinstance(em_sec, str) and em_sec != "-" else "",
            })
        if page * 100 >= (total or 0):
            break
        page += 1
        time.sleep(0.2)
    print(f"  push2 总数 {total},取回 {len(rows)} 行", flush=True)
    return rows


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


def build_recs(raw_rows, nas):
    recs, seen = [], set()
    for row in raw_rows:
        code, mc, pe = row.get("code", ""), row.get("mc"), row.get("pe")
        if not code or code in seen:
            continue
        if not isinstance(mc, (int, float)) or mc < MIN_MC:
            continue
        seen.add(code)
        # 代码在纳斯达克表里可能用 . / 或去分隔符,做变体匹配取行业
        nind = (nas.get(code) or nas.get(code.replace(".", "/")) or nas.get(code.replace("/", "."))
                or nas.get(code.replace(".", "")) or nas.get(code.split(".")[0]) or "")
        sec, g = sec_g_for_us(row.get("em_sec", ""), nind, code)
        recs.append({
            "code": code, "name": row.get("name", code),
            "pe": round(float(pe), 2) if isinstance(pe, (int, float)) else None,
            "mc": round(mc / 1e8, 2),  # 亿美元
            "rev": None, "profit": None, "cur": "USD",
            "ind": nind, "sec": sec, "g": g,
        })
    recs.sort(key=lambda x: -x["mc"])
    return recs


def main():
    today = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    nas = nasdaq_industry_map()
    print(f"纳斯达克行业表 {len(nas)} 条", flush=True)

    raw = fetch_stockanalysis()
    src = "stockanalysis"
    if len(raw) < SA_MIN_RAW:
        print(f"主源 stockanalysis 返回过少({len(raw)} < {SA_MIN_RAW}),回退 push2", flush=True)
        raw = fetch_push2()
        src = "push2"
    print(f"主数据源:{src},原始 {len(raw)} 行", flush=True)

    recs = build_recs(raw, nas)
    npe = sum(1 for r in recs if isinstance(r["pe"], (int, float)) and r["pe"] > 0)
    print(f"[{today}] 市值≥{MIN_MC/1e8:.0f}亿美元 {len(recs)} 只(有正PE {npe},无PE/亏损沉底 {len(recs)-npe};源:{src})", flush=True)
    if len(recs) < MIN_OK:
        raise SystemExit(f"有效记录过少({len(recs)} < {MIN_OK}),疑似接口异常,放弃写入")

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
