"""Cross-sectional preprocessing and factor-block composition."""

import numpy as np
import pandas as pd


DIRECTIONS = {
    "rskew": -1.0,
    "cpv_mean": -1.0,
    "cpv_std": -1.0,
    "smart": -1.0,
}


def winsorize_mad(values: pd.Series, n: float = 5.0) -> pd.Series:
    """Clip finite observations to scaled median-absolute-deviation bounds."""
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return values.copy()
    width = n * 1.4826 * mad
    return values.clip(median - width, median + width)


def _zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)
    if standard_deviation > 0:
        return (values - values.mean()) / standard_deviation
    return values * np.nan


def neutralize_day(
    values: pd.Series,
    float_cap: pd.Series,
    industry: pd.Series,
    min_count: int = 400,
) -> pd.Series:
    """Winsorize, standardize, neutralize, and restandardize one day."""
    for name, series in [
        ("values", values),
        ("float_cap", float_cap),
        ("industry", industry),
    ]:
        if not series.index.is_unique:
            raise ValueError(f"{name} index must be unique")
    frame = pd.concat(
        {"y": values, "cap": float_cap, "industry": industry},
        axis=1,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["cap"] > 0]
    if len(frame) < min_count:
        return pd.Series(np.nan, index=values.index, dtype=float)

    result = pd.Series(np.nan, index=values.index, dtype=float)
    y = _zscore(winsorize_mad(frame["y"]))
    if not np.isfinite(y).all():
        return result
    dummies = pd.get_dummies(
        frame["industry"],
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=frame.index, name="const"),
            np.log(frame["cap"]).rename("log_cap"),
            dummies,
        ],
        axis=1,
    )
    design_values = design.to_numpy()
    y_values = y.to_numpy()
    beta = np.linalg.lstsq(design_values, y_values, rcond=None)[0]
    if not np.isfinite(beta).all():
        return result
    residual_values = y_values - design_values @ beta
    residual = pd.Series(residual_values, index=frame.index)
    residual_norm = np.linalg.norm(residual_values, ord=2)
    error_scale = (
        np.linalg.norm(design_values, ord=2)
        * np.linalg.norm(beta, ord=2)
        + np.linalg.norm(y_values, ord=2)
    )
    backward_error = (
        np.finfo(float).eps * max(design_values.shape) * error_scale
    )
    residual_std = residual.std(ddof=0)
    if (
        not np.isfinite(residual_norm)
        or residual_norm <= backward_error
        or not np.isfinite(residual_std)
        or residual_std == 0
    ):
        return result
    result.loc[residual.index] = _zscore(residual)
    return result


def _require_columns(frame: pd.DataFrame, required, context: str) -> None:
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            f"{context} missing required columns: {', '.join(missing)}"
        )


def _normalize_dates(values, context: str):
    normalized = pd.to_datetime(values, errors="coerce", format="mixed")
    if pd.isna(normalized).any():
        raise ValueError(f"{context} contains invalid date")
    if isinstance(normalized, pd.Series):
        return normalized.dt.normalize()
    return pd.DatetimeIndex(normalized).normalize()


def preprocess_panels(
    factors: dict[str, pd.DataFrame],
    pools: pd.DataFrame,
    attributes: pd.DataFrame,
    min_count: int = 400,
) -> dict[str, pd.DataFrame]:
    """Neutralize fixed-direction factors inside each dated pool."""
    missing = [name for name in DIRECTIONS if name not in factors]
    if missing:
        raise ValueError(f"missing required factors: {', '.join(missing)}")
    unexpected = [name for name in factors if name not in DIRECTIONS]
    if unexpected:
        raise ValueError(f"unexpected factors: {', '.join(unexpected)}")

    normalized_factors = {}
    date_values = set()
    code_values = set()
    for name in DIRECTIONS:
        panel = factors[name].copy()
        panel.index = _normalize_dates(panel.index, f"factor {name}")
        if panel.index.has_duplicates:
            raise ValueError(f"factor {name} dates must be unique")
        if panel.columns.has_duplicates:
            raise ValueError(f"factor {name} codes must be unique")
        normalized_factors[name] = panel
        date_values.update(panel.index)
        code_values.update(panel.columns)

    dates = pd.DatetimeIndex(sorted(date_values))
    codes = pd.Index(sorted(code_values))
    aligned = {
        name: panel.reindex(index=dates, columns=codes)
        for name, panel in normalized_factors.items()
    }
    results = {
        name: pd.DataFrame(np.nan, index=dates, columns=codes, dtype=float)
        for name in DIRECTIONS
    }

    _require_columns(pools, ["date", "code"], "pools")
    pool = pools.copy()
    pool["date"] = _normalize_dates(pool["date"], "pools")
    if pool["code"].isna().any():
        raise ValueError("pools contains missing code")
    if pool.duplicated(["date", "code"]).any():
        raise ValueError("pools contains duplicate date/code")

    _require_columns(
        attributes,
        ["date", "code", "float_cap", "industry"],
        "attributes",
    )
    attrs = attributes.copy()
    attrs["date"] = _normalize_dates(attrs["date"], "attributes")
    if attrs["code"].isna().any():
        raise ValueError("attributes contains missing code")
    if attrs.duplicated(["date", "code"]).any():
        raise ValueError("attributes contains duplicate date/code")
    attrs["float_cap"] = pd.to_numeric(attrs["float_cap"], errors="coerce")
    anchor_dates = pd.DatetimeIndex(attrs["date"].unique()).sort_values()
    indexed_attrs = attrs.set_index(["date", "code"]).sort_index()

    date_positions = {day: position for position, day in enumerate(dates)}
    factor_codes = set(codes)
    for day in dates:
        members = [
            code
            for code in pool.loc[pool["date"].eq(day), "code"]
            if code in factor_codes
        ]
        if len(members) < min_count:
            continue

        prior = anchor_dates[anchor_dates <= day]
        if prior.empty:
            continue
        anchor = prior[-1]
        anchor_position = dates.searchsorted(anchor, side="left")
        if date_positions[day] - anchor_position > 4:
            continue
        latest = indexed_attrs.xs(anchor, level="date")

        for name, panel in aligned.items():
            values = panel.loc[day].reindex(members) * DIRECTIONS[name]
            processed = neutralize_day(
                values,
                latest["float_cap"],
                latest["industry"],
                min_count=min_count,
            )
            results[name].loc[day, members] = processed.reindex(
                members
            ).to_numpy()
    return results


def compose(processed: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compose two CPV inputs and three equally weighted logic blocks."""
    missing = [name for name in DIRECTIONS if name not in processed]
    if missing:
        raise ValueError(f"missing required factors: {', '.join(missing)}")

    for name in DIRECTIONS:
        if not processed[name].index.is_unique:
            raise ValueError(f"{name} index must be unique")
        if not processed[name].columns.is_unique:
            raise ValueError(f"{name} columns must be unique")

    index = processed["rskew"].index
    columns = processed["rskew"].columns
    for name in list(DIRECTIONS)[1:]:
        index = index.union(processed[name].index, sort=False)
        columns = columns.union(processed[name].columns, sort=False)
    aligned = {
        name: processed[name].reindex(index=index, columns=columns)
        for name in DIRECTIONS
    }

    cpv = (aligned["cpv_mean"] + aligned["cpv_std"]) / 2
    cpv = cpv.apply(_zscore, axis=1)
    score = (aligned["rskew"] + cpv + aligned["smart"]) / 3
    return {**processed, "cpv_block": cpv, "score": score}
