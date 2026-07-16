"""Five-day factor IC and overlapping quantile-cohort evaluation."""

from numbers import Integral

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "factor",
    "ic_mean",
    "ic_std",
    "icir",
    "positive_ic_rate",
    "ic_nw_t",
    "q5_q1_mean",
    "monotonicity",
]
QUANTILE_COLUMNS = ["factor", "date", "group", "return"]


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_panel(panel: pd.DataFrame, name: str) -> None:
    if not panel.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be increasing")
    if not panel.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    if not panel.columns.is_monotonic_increasing:
        raise ValueError(f"{name} columns must be increasing")


def _valid_open_prices(open_prices: pd.DataFrame) -> pd.DataFrame:
    return open_prices.where(
        np.isfinite(open_prices) & open_prices.gt(0)
    )


def forward_open_return(
    open_prices: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Return T+1-open to T+horizon+1-open forward returns."""
    horizon = _positive_integer(horizon, "horizon")
    _validate_panel(open_prices, "open_prices")
    prices = _valid_open_prices(open_prices)
    return prices.shift(-(horizon + 1)) / prices.shift(-1) - 1.0


def newey_west_t(values, lags: int = 4) -> float:
    """Return the Bartlett-kernel Newey-West t-statistic of a mean."""
    if isinstance(lags, bool) or not isinstance(lags, Integral) or lags < 0:
        raise ValueError("lags must be a nonnegative integer")
    lags = int(lags)
    array = (
        pd.Series(values, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    sample_size = len(array)
    if sample_size < 2:
        return np.nan
    demeaned = array - array.mean()
    long_variance = float(demeaned @ demeaned / sample_size)
    for lag in range(1, min(lags, sample_size - 1) + 1):
        covariance = float(
            demeaned[lag:] @ demeaned[:-lag] / sample_size
        )
        weight = 1 - lag / (lags + 1)
        long_variance += 2 * weight * covariance
    standard_error = np.sqrt(max(long_variance, 0) / sample_size)
    if standard_error <= 0:
        return np.nan
    return float(array.mean() / standard_error)


def rank_ic(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    min_count: int = 400,
) -> pd.Series:
    """Compute daily cross-sectional Spearman factor IC."""
    min_count = _positive_integer(min_count, "min_count")
    _validate_panel(factor, "factor")
    _validate_panel(forward, "forward")
    common_codes = factor.columns.intersection(
        forward.columns,
        sort=False,
    )
    rows = {}
    for day in factor.index.intersection(forward.index):
        joined = pd.DataFrame({
            "factor": factor.loc[day].reindex(common_codes),
            "forward": forward.loc[day].reindex(common_codes),
        })
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if len(joined) >= min_count:
            if joined.nunique().lt(2).any():
                rows[day] = np.nan
                continue
            ranks = joined.rank(method="average")
            correlation = ranks["factor"].corr(ranks["forward"])
            if np.isclose(abs(correlation), 1.0, rtol=0, atol=1e-12):
                correlation = float(np.copysign(1.0, correlation))
            rows[day] = correlation
    return pd.Series(rows, dtype=float).sort_index()


def quantile_cohorts(
    factor: pd.DataFrame,
    open_prices: pd.DataFrame,
    q: int = 5,
    horizon: int = 5,
    min_count: int = 400,
) -> pd.DataFrame:
    """Aggregate equal-weighted daily returns of overlapping cohorts."""
    q = _positive_integer(q, "q")
    horizon = _positive_integer(horizon, "horizon")
    min_count = _positive_integer(min_count, "min_count")
    _validate_panel(factor, "factor")
    _validate_panel(open_prices, "open_prices")
    prices = _valid_open_prices(open_prices)
    daily_returns = prices / prices.shift(1) - 1.0
    common_codes = factor.columns.intersection(
        open_prices.columns,
        sort=False,
    )
    records = []
    for day in factor.index.intersection(open_prices.index):
        entry_position = open_prices.index.get_loc(day) + 1
        exit_position = entry_position + horizon
        if exit_position >= len(open_prices.index):
            continue
        values = (
            factor.loc[day]
            .reindex(common_codes)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_index()
        )
        entry_prices = prices.iloc[entry_position].reindex(values.index)
        values = values[entry_prices.notna()]
        if len(values) < max(min_count, q):
            continue
        labels = pd.qcut(
            values.rank(method="first"),
            q,
            labels=False,
        ).astype(int)
        for return_position in range(entry_position + 1, exit_position + 1):
            returns = daily_returns.iloc[return_position]
            for group in range(q):
                members = labels[labels == group].index
                records.append({
                    "date": daily_returns.index[return_position],
                    "group": group,
                    "return": returns.reindex(members).mean(),
                })
    if not records:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=pd.Index(range(q), name="group"),
            dtype=float,
        )
    rows = pd.DataFrame(records)
    result = (
        rows.groupby(["date", "group"], sort=True)["return"]
        .mean()
        .unstack("group")
        .reindex(columns=range(q))
        .sort_index()
    )
    result.index.name = "date"
    result.columns.name = "group"
    return result


def _monotonicity(group_means: pd.Series) -> float:
    valid = group_means.dropna()
    if len(valid) < 2 or valid.nunique() < 2:
        return np.nan
    groups = pd.Series(valid.index.to_numpy(dtype=float), index=valid.index)
    ranks = pd.DataFrame({"group": groups, "return": valid}).rank()
    correlation = ranks["group"].corr(ranks["return"])
    if np.isclose(abs(correlation), 1.0, rtol=0, atol=1e-12):
        return float(np.copysign(1.0, correlation))
    return float(correlation)


def evaluate_factors(
    factors: dict[str, pd.DataFrame],
    open_prices: pd.DataFrame,
    min_count: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate IC statistics and five overlapping quantile cohorts."""
    min_count = _positive_integer(min_count, "min_count")
    forward = forward_open_return(open_prices, horizon=5)
    summaries = []
    daily_columns = {}
    quantile_rows = []

    for name, factor in factors.items():
        ic = rank_ic(factor, forward, min_count=min_count)
        quantile_returns = quantile_cohorts(
            factor,
            open_prices,
            q=5,
            horizon=5,
            min_count=min_count,
        )
        daily_columns[name] = ic

        valid_ic = ic.dropna()
        ic_mean = valid_ic.mean() if not valid_ic.empty else np.nan
        ic_std = valid_ic.std(ddof=1) if not valid_ic.empty else np.nan
        icir = (
            ic_mean / ic_std
            if np.isfinite(ic_std) and ic_std > 0
            else np.nan
        )
        positive_rate = (
            valid_ic.gt(0).mean() if not valid_ic.empty else np.nan
        )
        group_means = quantile_returns.mean()
        q5_q1 = (
            group_means.loc[4] - group_means.loc[0]
            if group_means.loc[[0, 4]].notna().all()
            else np.nan
        )
        summaries.append({
            "factor": name,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "positive_ic_rate": positive_rate,
            "ic_nw_t": newey_west_t(valid_ic, lags=4),
            "q5_q1_mean": q5_q1,
            "monotonicity": _monotonicity(group_means),
        })

        if not quantile_returns.empty:
            long = (
                quantile_returns.reset_index()
                .melt(
                    id_vars="date",
                    var_name="group",
                    value_name="return",
                )
                .dropna(subset=["return"])
            )
            long.insert(0, "factor", name)
            quantile_rows.append(long[QUANTILE_COLUMNS])

    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    if daily_columns:
        daily_ic = pd.DataFrame(daily_columns).sort_index()
    else:
        daily_ic = pd.DataFrame(index=pd.DatetimeIndex([]))
    daily_ic.index.name = "date"
    quantiles = (
        pd.concat(quantile_rows, ignore_index=True)
        if quantile_rows
        else pd.DataFrame(columns=QUANTILE_COLUMNS)
    )
    return summary, daily_ic, quantiles
