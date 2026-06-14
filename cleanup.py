#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保留每个市场最近 KEEP_DAYS 个交易日的快照,删除更早的数据文件并同步 manifest。

前端只读最近 90 天(index.html: dates.slice(-90)),这里多留一点做缓冲。
每天在抓取之后、提交之前运行,使仓库体积长期稳定(约 KEEP_DAYS × 2.3MB),
不会随时间无限增长撑爆 GitHub Pages / 仓库容量。

git rm 由外层 workflow 的 `git add data/`(含已删除文件)自动捕获,这里只管删文件 + 改 manifest。
"""
import json
import os

DATA_DIR = "data"
KEEP_DAYS = 100  # 前端用 90,留 10 天缓冲

# (manifest 文件, 该市场某日数据文件名的生成函数)
MARKETS = [
    ("manifest.json", lambda d: f"{d}.json"),       # 港股
    ("manifest_a.json", lambda d: f"a-{d}.json"),   # A股
    ("manifest_us.json", lambda d: f"us-{d}.json"), # 美股
]


def cleanup_market(manifest_name, fname):
    mpath = os.path.join(DATA_DIR, manifest_name)
    if not os.path.exists(mpath):
        return
    dates = sorted(set(json.load(open(mpath, encoding="utf-8")).get("dates", [])))
    if len(dates) <= KEEP_DAYS:
        print(f"{manifest_name}: {len(dates)} 天,未超过 {KEEP_DAYS},无需清理", flush=True)
        return

    keep = dates[-KEEP_DAYS:]
    drop = dates[:-KEEP_DAYS]
    removed = 0
    for d in drop:
        fp = os.path.join(DATA_DIR, fname(d))
        if os.path.exists(fp):
            os.remove(fp)
            removed += 1
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"dates": keep}, f, ensure_ascii=False)
    print(f"{manifest_name}: 删除 {len(drop)} 天(实删 {removed} 个文件),保留 {len(keep)} 天", flush=True)


def main():
    for manifest_name, fname in MARKETS:
        cleanup_market(manifest_name, fname)


if __name__ == "__main__":
    main()
