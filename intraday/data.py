"""Daily-data preparation for the dynamic intraday research universe."""

from pathlib import Path

import numpy as np
import pandas as pd


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
