"""Daily-data preparation for the dynamic intraday research universe."""

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd


MINUTE_COLUMNS = ["code", "time", "close", "volume", "amount"]
COVERAGE_COLUMNS = [
    "date",
    "code",
    "minute_count",
    "amount_relative_error",
    "reason",
]


def load_daily_raw(path: Path) -> pd.DataFrame:
    """Load the cached daily long table from pickle or Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def prepare_universe(
    raw: pd.DataFrame,
    start,
    end,
    top: int = 500,
    adv_window: int = 20,
    min_age: int = 60,
) -> dict:
    """Build the T-1 ADV-ranked and post-ranking eligible daily pools."""
    data = raw.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    amount = data.pivot(index="date", columns="code", values="amount").sort_index()
    volume = data.pivot(index="date", columns="code", values="volume").sort_index()
    adv = (
        amount.rolling(adv_window, min_periods=adv_window)
        .mean()
        .shift(1)
    )
    age = volume.fillna(0).gt(0).cumsum()

    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    eval_dates = amount.loc[start_day:end_day].index
    if eval_dates.empty:
        empty_pool = pd.DataFrame(
            columns=["date", "code", "adv20", "liquidity_rank"]
        )
        return {
            "eval_dates": eval_dates,
            "fetch_dates": amount.index[:0],
            "ranked_pool": empty_pool,
            "eligible_pool": empty_pool.copy(),
            "candidates": [],
            "estimated_rows": 0,
            "estimated_cells": 0,
        }

    ranked_rows = []
    eligible_rows = []
    for day in eval_dates:
        ranked = adv.loc[day].dropna().rename_axis("code").reset_index(name="adv20")
        ranked = ranked.sort_values(
            ["adv20", "code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top)
        ranked.insert(0, "date", day)
        ranked["liquidity_rank"] = np.arange(1, len(ranked) + 1)
        ranked_rows.append(ranked)

        age_on_day = ranked["code"].map(age.loc[day]).fillna(0)
        eligible = ranked[age_on_day.ge(min_age)]
        volume_on_day = eligible["code"].map(volume.loc[day]).fillna(0)
        eligible_rows.append(eligible[volume_on_day.gt(0)])

    ranked_pool = pd.concat(ranked_rows, ignore_index=True)
    eligible_pool = pd.concat(eligible_rows, ignore_index=True)
    candidates = sorted(ranked_pool["code"].unique())

    first_position = amount.index.get_loc(eval_dates[0])
    last_position = amount.index.get_loc(eval_dates[-1])
    fetch_dates = amount.index[max(0, first_position - adv_window):last_position + 1]
    estimated_rows = len(candidates) * len(fetch_dates) * 241
    return {
        "eval_dates": eval_dates,
        "fetch_dates": fetch_dates,
        "ranked_pool": ranked_pool,
        "eligible_pool": eligible_pool,
        "candidates": candidates,
        "estimated_rows": estimated_rows,
        "estimated_cells": estimated_rows * 3,
    }


def normalize_minute_day(
    frame: pd.DataFrame,
    day,
    daily_amount,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean one raw minute day and record per-code quality decisions."""
    if frame.empty:
        return pd.DataFrame(columns=MINUTE_COLUMNS), pd.DataFrame(
            columns=COVERAGE_COLUMNS
        )

    data = frame.copy()
    data["code"] = data["thscode"].astype(str).str.slice(0, 6).str.zfill(6)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.drop_duplicates(["code", "time"], keep="last")
    valid_day = data["time"].dt.normalize().eq(pd.Timestamp(day).normalize())
    valid_time = valid_day & (
        (
            (data["time"].dt.time >= time(9, 30))
            & (data["time"].dt.time <= time(11, 30))
        )
        | (
            (data["time"].dt.time >= time(13, 0))
            & (data["time"].dt.time <= time(15, 0))
        )
    )
    valid_values = (
        data["close"].gt(0)
        & data["volume"].ge(0)
        & data["amount"].ge(0)
    )
    data = (
        data.loc[valid_time & valid_values, MINUTE_COLUMNS]
        .sort_values(["code", "time"])
    )

    kept = []
    coverage_rows = []
    for code, group in data.groupby("code", sort=True):
        expected = float(daily_amount.get(code, np.nan))
        relative_error = (
            abs(group["amount"].sum() - expected) / expected
            if expected > 0
            else np.inf
        )
        reason = "ok"
        if len(group) < 200:
            reason = "too_few_minutes"
        elif group["volume"].gt(0).sum() < 30:
            reason = "too_few_trades"
        elif relative_error > 0.02:
            reason = "amount_mismatch"

        if reason == "ok":
            kept.append(group)
        coverage_rows.append({
            "date": pd.Timestamp(day),
            "code": code,
            "minute_count": len(group),
            "amount_relative_error": relative_error,
            "reason": reason,
        })

    clean = (
        pd.concat(kept, ignore_index=True)
        if kept
        else data.iloc[0:0].copy()
    )
    coverage = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    return clean, coverage


def _day_paths(day, root) -> tuple[Path, Path]:
    stem = pd.Timestamp(day).strftime("%Y-%m-%d")
    directory = Path(root) / "minute"
    return directory / f"{stem}.parquet", directory / f"{stem}.json"


def write_day_partition(
    frame: pd.DataFrame,
    statuses,
    day,
    root,
) -> tuple[Path, Path]:
    """Stage and atomically replace one day's data and completion manifest."""
    parquet, manifest = _day_paths(day, root)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    temp_parquet = parquet.with_suffix(".parquet.tmp")
    temp_manifest = manifest.with_suffix(".json.tmp")

    frame.to_parquet(temp_parquet, index=False)
    temp_manifest.write_text(
        json.dumps(
            {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "statuses": dict(sorted(statuses.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temp_parquet.replace(parquet)
    temp_manifest.replace(manifest)
    return parquet, manifest


def day_complete(day, codes, root) -> bool:
    """Return whether the day exactly covers the requested code collection."""
    parquet, manifest = _day_paths(day, root)
    if not parquet.exists() or not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    statuses = payload.get("statuses", {})
    if not isinstance(statuses, dict):
        return False
    return set(statuses) == set(codes) and set(statuses.values()) <= {
        "returned",
        "no_data",
    }
