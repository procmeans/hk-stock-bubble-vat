"""Daily-data preparation for the dynamic intraday research universe."""

import json
import re
import time
from datetime import time as clock_time
from numbers import Integral, Real
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
_THSCODE_PATTERN = re.compile(r"^([0-9]{6})\.(SZ|SH|BJ)$")
_BASE_CODE_PATTERN = re.compile(r"^([0-9]{6})(?:\.(SZ|SH|BJ))?$")


def _market_suffix(code: str) -> str:
    if code.startswith(("4", "8", "92")):
        return "BJ"
    if code.startswith(("6", "9")):
        return "SH"
    return "SZ"


def _normalize_code(value) -> str:
    """Return one unambiguous six-digit base code or raise ValueError."""
    suffix = None
    if isinstance(value, bool):
        raise ValueError(f"invalid stock code: {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"invalid stock code: {value!r}")
        number = int(numeric)
    elif isinstance(value, str):
        match = _BASE_CODE_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid stock code: {value!r}")
        code, suffix = match.groups()
        if suffix is not None and suffix != _market_suffix(code):
            raise ValueError(f"invalid stock code: {value!r}")
        return code
    else:
        raise ValueError(f"invalid stock code: {value!r}")

    if number < 0 or number > 999999:
        raise ValueError(f"invalid stock code: {value!r}")
    return f"{number:06d}"


def _to_thscode(code) -> str:
    code = _normalize_code(code)
    return f"{code}.{_market_suffix(code)}"


def _normalize_thscode(value):
    if not isinstance(value, str) or _THSCODE_PATTERN.fullmatch(value) is None:
        return None
    try:
        return _normalize_code(value)
    except ValueError:
        return None


def _normalize_code_series(series: pd.Series, context: str) -> pd.Series:
    codes = []
    for index, value in series.items():
        try:
            codes.append(_normalize_code(value))
        except ValueError as exc:
            raise ValueError(
                f"{context} contains invalid stock code at index {index}: {value!r}"
            ) from exc
    return pd.Series(codes, index=series.index, dtype="string")


def _normalize_code_list(values, context: str) -> list[str]:
    normalized = []
    seen = set()
    for value in values:
        try:
            code = _normalize_code(value)
        except ValueError as exc:
            raise ValueError(
                f"{context} contains invalid stock code: {value!r}"
            ) from exc
        if code not in seen:
            normalized.append(code)
            seen.add(code)
    return normalized


def _require_columns(frame: pd.DataFrame, required, context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def _normalize_dates(values, context: str):
    normalized = pd.to_datetime(values, errors="coerce", format="mixed")
    if pd.isna(normalized).any():
        raise ValueError(f"{context} contains invalid date")
    if isinstance(normalized, pd.Series):
        return normalized.dt.normalize()
    return pd.DatetimeIndex(normalized).normalize()


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
    data["code"] = _normalize_code_series(data["code"], "raw daily")
    data["date"] = _normalize_dates(data["date"], "raw daily")

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
    _require_columns(
        data,
        ["thscode", "time", "close", "volume", "amount"],
        "minute response",
    )
    response_codes = data["thscode"].map(_normalize_thscode)
    data = data.loc[response_codes.notna()].copy()
    if data.empty:
        return (
            _empty_typed_frame(MINUTE_DTYPES),
            _empty_typed_frame(COVERAGE_DTYPES),
        )
    data["code"] = response_codes.loc[data.index].astype("string")
    data["time"] = pd.to_datetime(
        data["time"], errors="coerce", format="mixed"
    )
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
    normalized_statuses = {}
    for value, status in statuses.items():
        code = _normalize_code(value)
        if code in normalized_statuses:
            raise ValueError(f"duplicate manifest stock code: {code}")
        normalized_statuses[code] = status

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
                    "statuses": dict(sorted(normalized_statuses.items())),
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
    candidates = _normalize_code_list(plan["candidates"], "plan candidates")
    _require_columns(raw_daily, ["code", "date", "amount"], "raw_daily")
    daily = raw_daily.copy()
    daily["code"] = _normalize_code_series(daily["code"], "raw_daily")
    daily["date"] = _normalize_dates(daily["date"], "raw_daily")
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
    coverage_frames = []

    for value in plan["fetch_dates"]:
        day = pd.Timestamp(value).normalize()
        if day_complete(day, candidates, root):
            continue

        frames = []
        returned = set()
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
                _require_columns(frame, ["thscode"], "minute response")
                response_codes = frame["thscode"].map(_normalize_thscode)
                accepted = response_codes.isin(set(batch))
                if accepted.any():
                    frames.append(frame.loc[accepted].copy())
                    returned.update(response_codes.loc[accepted])

        joined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
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
    try:
        start_day = pd.Timestamp(start)
        end_day = pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("adjusted history date bounds are invalid") from exc
    if pd.isna(start_day) or pd.isna(end_day):
        raise ValueError("adjusted history date bounds are invalid")
    start_day = start_day.date()
    end_day = end_day.date()
    if start_day > end_day:
        raise ValueError("adjusted history date range requires start <= end")
    request_start = start_day.strftime("%Y-%m-%d")
    request_end = end_day.strftime("%Y-%m-%d")

    frames = []
    requested = _normalize_code_list(codes, "adjusted daily codes")
    for batch in chunks(requested, batch_size):
        thscodes = [_to_thscode(code) for code in batch]

        def fetch_batch(thscodes=thscodes):
            return ths_http.history_quotation(
                thscodes,
                "open,close",
                request_start,
                request_end,
                functionpara={"CPS": "3", "Fill": "Omit"},
                access_token=access_token,
            )

        frame = _retry(fetch_batch)
        if frame.empty:
            continue
        _require_columns(
            frame,
            ["thscode", "time", "open", "close"],
            "adjusted history",
        )
        normalized = frame.copy()
        normalized["code"] = normalized["thscode"].map(_normalize_thscode)
        normalized["date"] = pd.to_datetime(
            normalized["time"], errors="coerce", format="mixed"
        ).dt.normalize()
        for column in ["open", "close"]:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            )
        valid_values = (
            np.isfinite(normalized[["open", "close"]]).all(axis=1)
            & normalized["open"].gt(0)
            & normalized["close"].gt(0)
        )
        valid = (
            normalized["code"].isin(set(batch))
            & normalized["date"].notna()
            & valid_values
        )
        if valid.any():
            frames.append(
                normalized.loc[valid, ["code", "date", "open", "close"]]
            )

    if not frames:
        raise RuntimeError("iFinD adjusted history returned no rows")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["code", "date"], keep="first")
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
    _require_columns(
        frame,
        ["股票代码", "股票简称", "所属同花顺行业"],
        "attributes",
    )
    cap_pattern = re.compile(
        rf"^a股市值\(不含限售股\)\[{stamp}\]$",
        re.IGNORECASE,
    )
    cap_columns = [
        column
        for column in frame.columns
        if cap_pattern.fullmatch(str(column)) is not None
    ]
    if len(cap_columns) != 1:
        raise ValueError(
            f"expected exactly one dated A-share float cap for {stamp}; "
            f"found {len(cap_columns)}"
        )
    cap_column = cap_columns[0]

    result = pd.DataFrame({
        "date": day,
        "code": _normalize_code_series(frame["股票代码"], "attributes"),
        "name": frame["股票简称"].astype("string"),
        "float_cap": pd.to_numeric(frame[cap_column], errors="coerce"),
        "industry": frame["所属同花顺行业"].astype("string"),
    })
    return (
        result.dropna(subset=["code"])
        .drop_duplicates("code", keep="first")
        .reset_index(drop=True)
    )


def fetch_attributes(anchor_dates, access_token) -> pd.DataFrame:
    """Fetch each unique point-in-time attribute anchor once."""
    anchors = sorted(
        set(_normalize_dates(pd.Index(anchor_dates), "attribute anchors"))
    )
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

    _require_columns(eligible_pool, ["date", "code"], "eligible_pool")
    _require_columns(
        attributes,
        ["date", "code", "name", "float_cap"],
        "attributes",
    )
    pool = eligible_pool.copy()
    pool["date"] = _normalize_dates(pool["date"], "eligible_pool")
    pool["code"] = _normalize_code_series(pool["code"], "eligible_pool")
    attrs = attributes.copy()
    attrs["date"] = _normalize_dates(attrs["date"], "attributes")
    attrs["code"] = _normalize_code_series(attrs["code"], "attributes")
    attrs["float_cap"] = pd.to_numeric(attrs["float_cap"], errors="coerce")

    eval_index = (
        _normalize_dates(pd.Index(eval_dates), "eval_dates")
        .drop_duplicates()
        .sort_values()
    )
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
        valid_name = names.notna() & ~names.fillna("").str.match(
            r"^\*?ST", case=False
        )
        valid_cap = np.isfinite(dated["float_cap"]) & dated["float_cap"].gt(0)
        allowed = set(dated.loc[valid_name & valid_cap, "code"])
        rows.append(members[members["code"].isin(allowed)])

    if not rows:
        return eligible_pool.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)
