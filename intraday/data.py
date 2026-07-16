"""Daily-data preparation for the dynamic intraday research universe."""

import json
import time
from datetime import time as clock_time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from alpha101 import ths_http
from alpha101.ths_today import chunks


MINUTE_DTYPES = {
    "code": "string",
    "time": "datetime64[ns]",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
}
COVERAGE_DTYPES = {
    "date": "datetime64[ns]",
    "code": "string",
    "minute_count": "int64",
    "amount_relative_error": "float64",
    "reason": "string",
}
MINUTE_COLUMNS = list(MINUTE_DTYPES)
COVERAGE_COLUMNS = list(COVERAGE_DTYPES)


def _to_thscode(code) -> str:
    code = str(code).zfill(6)
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _empty_typed_frame(dtypes) -> pd.DataFrame:
    return pd.DataFrame({
        column: pd.Series(dtype=dtype)
        for column, dtype in dtypes.items()
    })


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
        return (
            _empty_typed_frame(MINUTE_DTYPES),
            _empty_typed_frame(COVERAGE_DTYPES),
        )

    data = frame.copy()
    data["code"] = (
        data["thscode"].astype(str).str.slice(0, 6).str.zfill(6).astype("string")
    )
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(
            "float64"
        )
    raw_codes = sorted(data["code"].unique())

    data = data.drop_duplicates(["code", "time"], keep="last")
    valid_day = data["time"].dt.normalize().eq(pd.Timestamp(day).normalize())
    valid_time = valid_day & (
        (
            (data["time"].dt.time >= clock_time(9, 30))
            & (data["time"].dt.time <= clock_time(11, 30))
        )
        | (
            (data["time"].dt.time >= clock_time(13, 0))
            & (data["time"].dt.time <= clock_time(15, 0))
        )
    )
    valid_values = (
        np.isfinite(data[["close", "volume", "amount"]]).all(axis=1)
        & data["close"].gt(0)
        & data["volume"].ge(0)
        & data["amount"].ge(0)
    )
    data = (
        data.loc[valid_time & valid_values, MINUTE_COLUMNS]
        .sort_values(["code", "time"])
    )

    kept = []
    coverage_rows = []
    groups = dict(tuple(data.groupby("code", sort=True)))
    empty_group = data.iloc[0:0]
    for code in raw_codes:
        group = groups.get(code, empty_group)
        try:
            expected = float(daily_amount.get(code, np.nan))
        except (TypeError, ValueError):
            expected = np.nan
        relative_error = (
            abs(group["amount"].sum() - expected) / expected
            if np.isfinite(expected) and expected > 0
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
    coverage = pd.DataFrame(
        coverage_rows,
        columns=COVERAGE_COLUMNS,
    ).astype(COVERAGE_DTYPES)
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
    previous_manifest = manifest.with_suffix(".json.previous")

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
    if manifest.exists():
        manifest.replace(previous_manifest)
    temp_parquet.replace(parquet)
    temp_manifest.replace(manifest)
    try:
        previous_manifest.unlink()
    except FileNotFoundError:
        pass
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
    expected_date = pd.Timestamp(day).strftime("%Y-%m-%d")
    if payload.get("date") != expected_date or "statuses" not in payload:
        return False
    statuses = payload["statuses"]
    if not isinstance(statuses, dict):
        return False
    if not all(
        isinstance(code, str)
        and isinstance(status, str)
        and status in {"returned", "no_data"}
        for code, status in statuses.items()
    ):
        return False
    if isinstance(codes, (str, bytes)):
        return False
    try:
        requested_codes = list(codes)
    except TypeError:
        return False
    if not all(isinstance(code, str) for code in requested_codes):
        return False
    return set(statuses) == set(requested_codes)


def _retry(call, waits=(1, 2, 4), sleeper=None):
    """Retry temporary request failures with the fixed iFinD backoff."""
    sleeper = time.sleep if sleeper is None else sleeper
    waits = tuple(waits)
    for attempt in range(len(waits) + 1):
        try:
            return call()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == len(waits):
                raise
            sleeper(waits[attempt])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            temporary = status == 429 or (
                status is not None and 500 <= status <= 599
            )
            if not temporary or attempt == len(waits):
                raise
            sleeper(waits[attempt])


def fetch_minute_partitions(
    plan,
    raw_daily,
    root,
    access_token,
    batch_size=200,
) -> pd.DataFrame:
    """Download, validate, and cache each unfinished minute partition."""
    candidates = list(plan["candidates"])
    daily = raw_daily.copy()
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    coverage_frames = []

    for value in plan["fetch_dates"]:
        day = pd.Timestamp(value).normalize()
        if day_complete(day, candidates, root):
            continue

        frames = []
        for batch in chunks(candidates, batch_size):
            thscodes = [_to_thscode(code) for code in batch]

            def fetch_batch(thscodes=thscodes, day=day):
                return ths_http.high_frequency(
                    thscodes,
                    "close,volume,amount",
                    f"{day:%Y-%m-%d} 09:30:00",
                    f"{day:%Y-%m-%d} 15:00:00",
                    functionpara={
                        "CPS": "no",
                        "Fill": "Original",
                        "Timeformat": "LocalTime",
                        "Limitstart": "09:30:00",
                        "Limitend": "15:00:00",
                    },
                    access_token=access_token,
                )

            frame = _retry(fetch_batch)
            if not frame.empty:
                frames.append(frame)

        joined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
        returned = set(
            joined.get("thscode", pd.Series(dtype="string"))
            .astype(str)
            .str.extract(r"(\d{6})", expand=False)
            .dropna()
        )
        statuses = {
            code: "returned" if code in returned else "no_data"
            for code in candidates
        }
        amounts = (
            daily.loc[daily["date"].eq(day), ["code", "amount"]]
            .set_index("code")["amount"]
        )
        clean, coverage = normalize_minute_day(joined, day, amounts)
        write_day_partition(clean, statuses, day, root)
        if not coverage.empty:
            coverage_frames.append(coverage)

    if not coverage_frames:
        return _empty_typed_frame(COVERAGE_DTYPES)
    return pd.concat(coverage_frames, ignore_index=True).astype(COVERAGE_DTYPES)


def fetch_adjusted_daily(
    codes,
    start,
    end,
    access_token,
    batch_size=200,
) -> pd.DataFrame:
    """Download CPS3 adjusted open and close history in bounded batches."""
    frames = []
    for batch in chunks(list(codes), batch_size):
        thscodes = [_to_thscode(code) for code in batch]

        def fetch_batch(thscodes=thscodes):
            return ths_http.history_quotation(
                thscodes,
                "open,close",
                start,
                end,
                functionpara={"CPS": "3", "Fill": "Omit"},
                access_token=access_token,
            )

        frame = _retry(fetch_batch)
        if frame.empty:
            continue
        normalized = frame.copy()
        normalized["code"] = normalized["thscode"].astype(str).str.extract(
            r"(\d{6})", expand=False
        )
        normalized["date"] = pd.to_datetime(
            normalized["time"], errors="coerce"
        ).dt.normalize()
        for column in ["open", "close"]:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            )
        frames.append(normalized[["code", "date", "open", "close"]])

    if not frames:
        raise RuntimeError("iFinD adjusted history returned no rows")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["code", "date"])
        .reset_index(drop=True)
    )


def build_attribute_query(day) -> str:
    """Build an explicitly dated iFinD point-in-time attribute query."""
    day = pd.Timestamp(day).normalize()
    prefix = f"{day.year}年{day.month}月{day.day}日"
    return f"{prefix}A股，{prefix}流通市值，所属同花顺行业"


def _empty_attributes() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "code": pd.Series(dtype="string"),
        "name": pd.Series(dtype="string"),
        "float_cap": pd.Series(dtype="float64"),
        "industry": pd.Series(dtype="string"),
    })


def normalize_attributes(frame, day) -> pd.DataFrame:
    """Normalize one dated smart-picking response without look-ahead."""
    day = pd.Timestamp(day).normalize()
    stamp = day.strftime("%Y%m%d")
    cap_column = next(
        (
            column
            for column in frame.columns
            if stamp in str(column)
            and "市值" in str(column)
            and "限售" in str(column)
        ),
        None,
    )
    if cap_column is None:
        raise ValueError(f"missing dated float cap for {stamp}")

    result = pd.DataFrame({
        "date": day,
        "code": frame["股票代码"].astype(str).str.extract(
            r"(\d{6})", expand=False
        ),
        "name": frame["股票简称"].astype(str),
        "float_cap": pd.to_numeric(frame[cap_column], errors="coerce"),
        "industry": frame["所属同花顺行业"].astype(str),
    })
    return (
        result.dropna(subset=["code"])
        .drop_duplicates("code", keep="first")
        .reset_index(drop=True)
    )


def fetch_attributes(anchor_dates, access_token) -> pd.DataFrame:
    """Fetch each unique point-in-time attribute anchor once."""
    anchors = sorted({pd.Timestamp(day).normalize() for day in anchor_dates})
    frames = []
    for day in anchors:

        def fetch_anchor(day=day):
            return ths_http.smart_stock_picking(
                build_attribute_query(day),
                access_token=access_token,
                timeout=90,
            )

        raw = _retry(fetch_anchor)
        frames.append(normalize_attributes(raw, day))
    if not frames:
        return _empty_attributes()
    return pd.concat(frames, ignore_index=True)


def apply_attribute_filters(
    eligible_pool,
    attributes,
    eval_dates,
) -> pd.DataFrame:
    """Apply fresh point-in-time ST and float-cap filters after ranking."""
    if eligible_pool.empty or attributes.empty:
        return eligible_pool.iloc[0:0].copy()

    pool = eligible_pool.copy()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    attrs = attributes.copy()
    attrs["date"] = pd.to_datetime(attrs["date"]).dt.normalize()
    attrs["float_cap"] = pd.to_numeric(attrs["float_cap"], errors="coerce")

    eval_index = pd.DatetimeIndex(eval_dates).normalize().drop_duplicates()
    date_positions = {day: position for position, day in enumerate(eval_index)}
    anchors = pd.DatetimeIndex(attrs["date"].dropna().unique()).sort_values()
    rows = []

    for day, members in pool.groupby("date", sort=True):
        if day not in date_positions:
            continue
        prior = anchors[anchors <= day]
        if prior.empty:
            continue
        anchor = prior[-1]
        anchor_position = eval_index.searchsorted(anchor, side="left")
        if date_positions[day] - anchor_position > 4:
            continue

        dated = (
            attrs.loc[attrs["date"].eq(anchor)]
            .drop_duplicates("code", keep="first")
        )
        names = dated["name"].astype("string").str.strip()
        valid_name = ~names.str.match(r"^\*?ST", case=False, na=False)
        valid_cap = np.isfinite(dated["float_cap"]) & dated["float_cap"].gt(0)
        allowed = set(dated.loc[valid_name & valid_cap, "code"])
        rows.append(members[members["code"].isin(allowed)])

    if not rows:
        return eligible_pool.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)
