"""yfinance 港股/美股历史日线:抓取、缓存、面板构建、Alpha101 全流程。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from alpha101 import alphas, compose, select, universe
from alpha101.ths_history import read_raw_cache, write_raw_cache
from alpha101.ths_today import chunks

FIELDS = ["open", "high", "low", "close", "volume"]
MANIFESTS = {"hk": ("manifest.json", ""), "us": ("manifest_us.json", "us-")}


def to_yf_ticker(code: str, market: str) -> str:
    if market == "hk":
        return f"{str(code).lstrip('0').zfill(4)}.HK"
    return str(code).replace(".", "-")


def default_universe(market: str, data_dir: Path = Path("data")) -> Path:
    manifest_name, prefix = MANIFESTS[market]
    dates = json.loads((data_dir / manifest_name).read_text())["dates"]
    return data_dir / f"{prefix}{max(dates)}.json"


def default_cache(market: str) -> Path:
    return Path(f"alpha101/cache/yf_panel_{market}.pkl")


def load_universe(path: Path) -> pd.DataFrame:
    """读快照 JSON,保留原始 code(不做 A 股过滤),附带 GICS 行业组 g。"""
    if not path.exists():
        raise FileNotFoundError(f"missing universe: {path}")
    data = pd.DataFrame(json.loads(path.read_text()))
    if "name" not in data.columns:
        data["name"] = data["code"]
    if "g" not in data.columns:
        data["g"] = None
    data["code"] = data["code"].astype(str)
    return data[["code", "name", "g"]].drop_duplicates("code")


def normalize_download(data: pd.DataFrame, code_map: dict) -> pd.DataFrame:
    """yf.download 宽表 -> 长表(code,date,open,high,low,close,volume)。

    code_map: yf ticker -> 原始快照 code;全 NaN 的 ticker(退市/无数据)丢弃。
    """
    frames = []
    for ticker, code in code_map.items():
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(1):
                continue
            sub = data.xs(ticker, axis=1, level=1)
        else:
            sub = data
        if sub.empty or not set(["Open", "High", "Low", "Close", "Volume"]) <= set(sub.columns):
            continue
        frame = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
        frame.columns = FIELDS
        frame = frame.dropna(how="all")
        if frame.empty:
            continue
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["code"] = code
        frames.append(frame[["code", "date"] + FIELDS])
    if not frames:
        return pd.DataFrame(columns=["code", "date"] + FIELDS)
    return pd.concat(frames, ignore_index=True).dropna(subset=["close"])


def build_panel(raw: pd.DataFrame, industries: pd.Series | None = None) -> dict:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    panel = {}
    for field in FIELDS:
        panel[field] = raw.pivot(index="date", columns="code", values=field).sort_index()
    # yfinance 无成交额字段,用典型价近似 vwap,amount = vwap * volume。
    panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3
    panel["amount"] = panel["vwap"] * panel["volume"]
    panel["returns"] = panel["close"].pct_change()
    if industries is not None:
        panel["ind"] = industries.reindex(panel["close"].columns)
    return panel


def fetch_history(
    universe_path: Path,
    market: str,
    start: str,
    end: str,
    cache: Path | None = None,
    batch_size: int = 100,
    pause: float = 1.0,
) -> pd.DataFrame:
    import yfinance as yf

    cache = cache or default_cache(market)
    pool = load_universe(universe_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if cache.exists():
        cached = read_raw_cache(cache)
        frames.append(cached)
        done_codes = set(cached["code"].astype(str).unique())
        print(f"resume: {len(done_codes)} cached codes", flush=True)
        pool = pool[~pool["code"].isin(done_codes)]

    code_map = {to_yf_ticker(code, market): code for code in pool["code"]}
    tickers = list(code_map)
    for index, batch in enumerate(chunks(tickers, batch_size), start=1):
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
        )
        frame = normalize_download(data, {t: code_map[t] for t in batch})
        if not frame.empty:
            frames.append(frame)
            write_raw_cache(
                pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"]),
                cache,
            )
        print(f"history batch {index}: {len(batch)} tickers, {len(frame)} rows", flush=True)
        time.sleep(pause)
    if not frames:
        raise RuntimeError("yfinance returned empty data")

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    write_raw_cache(raw, cache)
    return raw


def run_full(
    universe_path: Path,
    market: str,
    cache: Path | None = None,
    output: Path | None = None,
    top_n: int = 101,
) -> pd.DataFrame:
    cache = cache or default_cache(market)
    output = output or Path(f"output/yf_{market}_alpha101_picks.csv")
    pool = load_universe(universe_path)
    panel = build_panel(read_raw_cache(cache), pool.set_index("code")["g"])
    mask = universe.liquidity_mask(panel)
    factors = alphas.compute_all(panel)
    score = compose.composite(factors, mask=mask)
    last = score.dropna(how="all").index[-1]
    names = pool.set_index("code")["name"].to_dict()
    picks = select.pick(score, last, names=names, top_n=top_n)
    picks["date"] = last.date().isoformat()
    picks["factor_count"] = len(factors)
    output.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"computed {len(factors)} factors")
    print(f"wrote {len(picks)} rows to {output}")
    print(picks.head(20).to_string(index=False))
    return picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["fetch", "run", "all"])
    parser.add_argument("--market", choices=["hk", "us"], required=True)
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=101)
    args = parser.parse_args()

    universe_path = args.universe or default_universe(args.market)
    if args.cmd in {"fetch", "all"}:
        fetch_history(
            universe_path,
            args.market,
            args.start,
            args.end,
            cache=args.cache,
            batch_size=args.batch_size,
        )
    if args.cmd in {"run", "all"}:
        run_full(
            universe_path,
            args.market,
            cache=args.cache,
            output=args.output,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
