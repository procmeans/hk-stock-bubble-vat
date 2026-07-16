"""Compress minute partitions into rolling daily factor panels."""

import re

import numpy as np
import pandas as pd


_DAY_FACTOR_KEYS = ("rskew_day", "cpv_day", "smart_q_day")
_MINUTE_COLUMNS = ("time", "close", "volume", "amount")
_CODE_PATTERN = re.compile(r"^[0-9]{6}$")


def _nan_day_factors() -> dict[str, float]:
    return {key: np.nan for key in _DAY_FACTOR_KEYS}


def minute_day_factors(frame: pd.DataFrame) -> dict[str, float]:
    """Compute one stock-day's intraday RSkew, CPV, and SmartQ."""
    if frame.empty:
        return _nan_day_factors()
    missing = [column for column in _MINUTE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            f"minute frame missing required columns: {', '.join(missing)}"
        )
    data = frame.copy()
    data["time"] = pd.to_datetime(
        data["time"], errors="coerce", format="mixed"
    )
    if data["time"].isna().any():
        raise ValueError("minute frame contains invalid time")
    if data["time"].duplicated().any():
        raise ValueError("minute frame contains duplicate time")

    numeric = data[["close", "volume", "amount"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("minute frame contains non-finite values")
    if (data["close"] <= 0).any():
        raise ValueError("minute frame contains non-positive close values")

    data = data.sort_values("time", kind="mergesort")
    data["r"] = np.log(data["close"] / data["close"].shift(1))

    returns = data["r"].dropna()
    squared_sum = float((returns ** 2).sum())
    if squared_sum > 0:
        rskew = (
            np.sqrt(len(returns))
            * float((returns ** 3).sum())
            / squared_sum ** 1.5
        )
    else:
        rskew = np.nan

    traded = data[data["volume"] > 0].copy()
    if (
        len(traded) >= 2
        and traded["close"].nunique() > 1
        and traded["volume"].nunique() > 1
    ):
        cpv = traded["close"].corr(traded["volume"])
    else:
        cpv = np.nan
    traded["smartness"] = traded["r"].abs() / np.sqrt(traded["volume"])
    ranked = traded.dropna(subset=["smartness"]).sort_values(
        ["smartness", "time"],
        ascending=[False, True],
        kind="mergesort",
    )
    if ranked.empty or ranked["volume"].sum() <= 0:
        smart_q = np.nan
    else:
        target = 0.20 * traded["volume"].sum()
        prior_volume = ranked["volume"].cumsum().shift(fill_value=0)
        smart = ranked[prior_volume < target]
        all_amount = traded["amount"].sum()
        if all_amount > 0:
            smart_vwap = smart["amount"].sum() / smart["volume"].sum()
            all_vwap = all_amount / traded["volume"].sum()
            smart_q = smart_vwap / all_vwap
        else:
            smart_q = np.nan

    return {
        "rskew_day": rskew,
        "cpv_day": cpv,
        "smart_q_day": smart_q,
    }


def factor_panels(
    partitions,
    codes,
    dates,
    window: int = 20,
    min_periods: int = 15,
) -> dict[str, pd.DataFrame]:
    """Aggregate stock-day factors and apply the requested rolling windows."""
    parsed_dates = pd.to_datetime(
        list(dates), errors="coerce", format="mixed"
    )
    index = pd.DatetimeIndex(parsed_dates)
    if index.isna().any():
        raise ValueError("dates contains invalid date")
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("dates must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("dates must be increasing")

    code_values = list(codes)
    if not all(
        isinstance(code, str) and _CODE_PATTERN.fullmatch(code)
        for code in code_values
    ):
        raise ValueError("codes must contain six-digit strings")
    columns = pd.Index(code_values)
    if columns.has_duplicates:
        raise ValueError("codes must be unique")

    rows = []
    seen_keys = set()
    for day, frame in partitions:
        parsed_day = pd.to_datetime(day, errors="coerce", format="mixed")
        if pd.isna(parsed_day):
            raise ValueError("partition contains invalid date")
        partition_day = pd.Timestamp(parsed_day).normalize()
        if "code" not in frame:
            raise ValueError("partition missing required column: code")
        for code, group in frame.groupby("code", sort=True):
            key = (partition_day, code)
            if key in seen_keys:
                raise ValueError("partitions contain duplicate date/code")
            seen_keys.add(key)
            rows.append({
                "date": partition_day,
                "code": code,
                **minute_day_factors(group),
            })

    if rows:
        daily = pd.DataFrame(rows)

        def panel(column):
            return daily.pivot(
                index="date", columns="code", values=column
            ).reindex(index=index, columns=columns)

        skew_day = panel("rskew_day")
        cpv_day = panel("cpv_day")
        smart_day = panel("smart_q_day")
    else:
        skew_day = pd.DataFrame(index=index, columns=columns, dtype=float)
        cpv_day = skew_day.copy()
        smart_day = skew_day.copy()

    return {
        "rskew": skew_day.rolling(
            window, min_periods=min_periods
        ).mean(),
        "cpv_mean": cpv_day.rolling(
            window, min_periods=min_periods
        ).mean(),
        "cpv_std": cpv_day.rolling(
            window, min_periods=min_periods
        ).std(ddof=1),
        "smart": smart_day.rolling(
            window, min_periods=min_periods
        ).mean(),
    }
