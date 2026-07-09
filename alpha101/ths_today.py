"""Run today's iFinD quotation through WorldQuant Alpha101 factor #101."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha101 import ths_http

DEFAULT_OUTPUT = Path("output/ths_alpha101_today.csv")


def to_thscode(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def is_supported_a_share_code(code: str) -> bool:
    code = str(code).zfill(6)
    return code.startswith(("00", "30", "60", "68", "90", "92"))


def chunks(values, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_code_pool(path: Path) -> pd.DataFrame:
    """Load a code/name universe from CSV or from data/a-*.json snapshots."""
    if not path.exists():
        raise FileNotFoundError(f"missing code pool: {path}")
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, dtype={"code": str, "代码": str})
    else:
        data = pd.read_json(path)
    rename_map = {"代码": "code", "名称": "name", "name_cn": "name"}
    data = data.rename(columns=rename_map)
    if "code" not in data.columns:
        raise ValueError(f"{path} must contain code or 代码 column")
    if "name" not in data.columns:
        data["name"] = data["code"]
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data[data["code"].map(is_supported_a_share_code)]
    return data[["code", "name"]].drop_duplicates("code")


def normalize_ths_spot(
    data: pd.DataFrame,
    names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = data.copy()
    result = result.rename(columns={"latest": "close"})
    if "thscode" in result.columns and "code" not in result.columns:
        result["code"] = result["thscode"].astype(str).str.slice(0, 6)
    if "code" in result.columns:
        result["code"] = result["code"].astype(str).str.zfill(6)
    if names is not None and "name" not in result.columns:
        name_map = names.copy()
        name_map["code"] = name_map["code"].astype(str).str.zfill(6)
        result = result.merge(name_map[["code", "name"]], on="code", how="left")
    return result


def fetch_today_spot(code_pool: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
    frames = []
    codes = [to_thscode(code) for code in code_pool["code"].tolist()]
    for batch in chunks(codes, batch_size):
        frame = ths_http.real_time_quotation(batch, "open,high,low,latest")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("iFinD real_time_quotation returned empty data")
    return normalize_ths_spot(pd.concat(frames, ignore_index=True), code_pool)


def score_today_alpha101(data: pd.DataFrame, top_n: int = 101) -> pd.DataFrame:
    required = ["code", "name", "open", "high", "low", "close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    result = data.copy()
    for column in ["open", "high", "low", "close"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["open", "high", "low", "close"])
    result = result[result["high"] > result["low"]]
    result["alpha101"] = (result["close"] - result["open"]) / (
        result["high"] - result["low"] + 0.001
    )
    result["score"] = (result["alpha101"].rank(pct=True, method="average") * 100).round(2)
    result = result.sort_values(
        ["score", "alpha101", "code"], ascending=[False, False, True]
    )
    result = result.head(top_n).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def run(
    universe: Path,
    output: Path = DEFAULT_OUTPUT,
    top_n: int = 101,
    batch_size: int = 200,
) -> pd.DataFrame:
    code_pool = load_code_pool(universe)
    spot = fetch_today_spot(code_pool, batch_size=batch_size)
    result = score_today_alpha101(spot, top_n=top_n)
    result["factor"] = "WQAlpha101"
    result["data_source"] = "iFinD HTTP real_time_quotation"

    columns = [
        "rank",
        "code",
        "name",
        "score",
        "alpha101",
        "close",
        "open",
        "high",
        "low",
        "factor",
        "data_source",
    ]
    result = result[[column for column in columns if column in result.columns]]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, default=Path("data/a-2026-07-07.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    result = run(
        universe=args.universe,
        output=args.output,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
    print(f"wrote {len(result)} rows to {args.output}")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
