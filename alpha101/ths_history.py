"""Build iFinD historical panels and run the full Alpha101 stack."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alpha101 import alphas, compose, select, ths_http, universe
from alpha101.ths_today import chunks, load_code_pool, to_thscode

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
DEFAULT_CACHE = Path("alpha101/cache/ths_panel.pkl")
DEFAULT_OUTPUT = Path("output/ths_full_alpha101_picks.csv")


def read_raw_cache(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def write_raw_cache(raw: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        raw.to_parquet(path)
    else:
        raw.to_pickle(path)


def normalize_history_frame(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "thscode" in result.columns and "code" not in result.columns:
        result["code"] = result["thscode"].astype(str).str.slice(0, 6)
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["date"] = pd.to_datetime(result["time"])
    for column in FIELDS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[["code", "date"] + FIELDS].dropna(subset=["date", "close"])


def build_panel(raw: pd.DataFrame, industries: pd.Series | None = None) -> dict:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    panel = {}
    for field in FIELDS:
        panel[field] = raw.pivot(index="date", columns="code", values=field).sort_index()
    # iFinD returns A-share volume in shares, so per-share VWAP is amount / volume.
    panel["vwap"] = panel["amount"] / panel["volume"].replace(0, np.nan)
    panel["returns"] = panel["close"].pct_change()
    if industries is not None:
        panel["ind"] = industries.reindex(panel["close"].columns)
    return panel


def load_industries(path: Path) -> pd.Series:
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, dtype={"code": str, "代码": str})
    else:
        data = pd.read_json(path)
    data = data.rename(columns={"代码": "code", "名称": "name"})
    industry_column = next(
        (column for column in ["ind", "sec", "g"] if column in data.columns),
        None,
    )
    if industry_column is None:
        return pd.Series(dtype=object)
    data["code"] = data["code"].astype(str).str.zfill(6)
    return data.drop_duplicates("code").set_index("code")[industry_column]


def fetch_history(
    universe_path: Path,
    start: str,
    end: str,
    cache: Path = DEFAULT_CACHE,
    batch_size: int = 80,
    pause: float = 0.2,
) -> pd.DataFrame:
    code_pool = load_code_pool(universe_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    done_codes = set()
    if cache.exists():
        cached = read_raw_cache(cache)
        frames.append(cached)
        done_codes = set(cached["code"].astype(str).str.zfill(6).unique())
        print(f"resume: {len(done_codes)} cached codes", flush=True)
        code_pool = code_pool[~code_pool["code"].isin(done_codes)]

    codes = [to_thscode(code) for code in code_pool["code"].tolist()]
    for index, batch in enumerate(chunks(codes, batch_size), start=1):
        frame = ths_http.history_quotation(
            batch,
            "open,high,low,close,volume,amount",
            start,
            end,
            functionpara={"CPS": "1", "Fill": "Omit"},
        )
        if not frame.empty:
            frames.append(normalize_history_frame(frame))
            write_raw_cache(
                pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"]),
                cache,
            )
        print(f"history batch {index}: {len(batch)} codes", flush=True)
        time.sleep(pause)
    if not frames:
        raise RuntimeError("iFinD history quotation returned empty data")

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    write_raw_cache(raw, cache)
    return raw


def load_panel(cache: Path, universe_path: Path) -> dict:
    raw = read_raw_cache(cache)
    return build_panel(raw, load_industries(universe_path))


def run_full(
    universe_path: Path,
    cache: Path = DEFAULT_CACHE,
    output: Path = DEFAULT_OUTPUT,
    top_n: int = 101,
) -> pd.DataFrame:
    panel = load_panel(cache, universe_path)
    mask = universe.liquidity_mask(panel)
    factors = alphas.compute_all(panel)
    score = compose.composite(factors, mask=mask)
    last = score.dropna(how="all").index[-1]
    names = load_code_pool(universe_path).set_index("code")["name"].to_dict()
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
    parser.add_argument("--universe", type=Path, default=Path("data/a-2026-07-07.json"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2024-07-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=101)
    args = parser.parse_args()

    if args.cmd in {"fetch", "all"}:
        fetch_history(
            args.universe,
            args.start,
            args.end,
            args.cache,
            batch_size=args.batch_size,
        )
    if args.cmd in {"run", "all"}:
        run_full(args.universe, args.cache, args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
