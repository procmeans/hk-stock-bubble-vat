"""Next-open event-driven portfolio simulation for intraday factors."""

from numbers import Integral

import numpy as np
import pandas as pd

from intraday.evaluate import newey_west_t


TRADE_COLUMNS = [
    "signal_date",
    "date",
    "code",
    "side",
    "shares",
    "price",
    "notional",
    "cost",
    "status",
]
RAW_COLUMNS = ["open", "high", "low", "close"]


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalized_pools(pools: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ["date", "code"] if column not in pools]
    if missing:
        raise ValueError(
            f"pools missing required columns: {', '.join(missing)}"
        )
    result = pools.copy()
    parsed = pd.to_datetime(result["date"], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("pools contains invalid date")
    result["date"] = parsed.dt.normalize()
    if result["code"].isna().any():
        raise ValueError("pools contains missing code")
    if result.duplicated(["date", "code"]).any():
        raise ValueError("pools contains duplicate date/code")
    return result


def build_targets(
    score: pd.DataFrame,
    pools: pd.DataFrame,
    every: int = 5,
    top_n: int = 50,
    min_count: int = 400,
) -> dict[pd.Timestamp, pd.Series]:
    """Build fixed-cadence top-score targets inside the final daily pool."""
    every = _positive_integer(every, "every")
    top_n = _positive_integer(top_n, "top_n")
    min_count = _positive_integer(min_count, "min_count")
    if not score.index.is_unique:
        raise ValueError("score index must be unique")
    if not score.index.is_monotonic_increasing:
        raise ValueError("score index must be increasing")
    if not score.columns.is_unique:
        raise ValueError("score columns must be unique")
    pool = _normalized_pools(pools)

    def valid_scores(day):
        members = sorted(pool.loc[pool["date"].eq(day), "code"])
        return (
            score.loc[day]
            .reindex(members)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

    anchor_position = None
    for position, day in enumerate(score.index):
        if len(valid_scores(day)) >= min_count:
            anchor_position = position
            break
    if anchor_position is None:
        return {}

    targets = {}
    for position in range(anchor_position, len(score.index), every):
        day = score.index[position]
        valid = valid_scores(day)
        if len(valid) < min_count:
            continue
        ranked = (
            valid.sort_index()
            .sort_values(ascending=False, kind="mergesort")
            .head(top_n)
        )
        if not ranked.empty:
            targets[day] = pd.Series(
                1.0 / len(ranked),
                index=ranked.index,
                dtype=float,
            )
    return targets


def build_benchmark_targets(
    pools: pd.DataFrame,
    signal_dates,
) -> dict[pd.Timestamp, pd.Series]:
    """Build equal weights for each signal day's final eligible pool."""
    pool = _normalized_pools(pools)
    parsed = pd.DatetimeIndex(
        pd.to_datetime(list(signal_dates), errors="coerce", format="mixed")
    )
    if parsed.isna().any():
        raise ValueError("signal_dates contains invalid date")
    parsed = parsed.normalize()
    if parsed.has_duplicates:
        raise ValueError("signal_dates must be unique")
    targets = {}
    for day in parsed:
        members = sorted(pool.loc[pool["date"].eq(day), "code"])
        if members:
            targets[day] = pd.Series(
                1.0 / len(members),
                index=members,
                dtype=float,
            )
    return targets


def _validate_simulation_inputs(
    targets,
    adjusted_open,
    raw_daily,
    cost_bps,
) -> tuple[float, dict]:
    try:
        cost_value = float(cost_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("cost_bps must be finite and nonnegative") from exc
    if not np.isfinite(cost_value) or cost_value < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if not adjusted_open.index.is_unique:
        raise ValueError("adjusted_open index must be unique")
    if not adjusted_open.index.is_monotonic_increasing:
        raise ValueError("adjusted_open index must be increasing")
    if not adjusted_open.columns.is_unique:
        raise ValueError("adjusted_open columns must be unique")
    if not adjusted_open.columns.is_monotonic_increasing:
        raise ValueError("adjusted_open columns must be increasing")
    if not isinstance(raw_daily.index, pd.MultiIndex) or raw_daily.index.nlevels != 2:
        raise ValueError("raw_daily index must be date/code MultiIndex")
    if not raw_daily.index.is_unique:
        raise ValueError("raw_daily index must be unique")
    if list(raw_daily.index.names) != ["date", "code"]:
        raise ValueError("raw_daily index levels must be date,code")
    raw_dates = pd.to_datetime(
        raw_daily.index.get_level_values("date"),
        errors="coerce",
        format="mixed",
    )
    if raw_dates.isna().any():
        raise ValueError("raw_daily date level contains invalid date")
    if not raw_daily.index.is_monotonic_increasing:
        raise ValueError("raw_daily index must be increasing")
    missing_raw = [column for column in RAW_COLUMNS if column not in raw_daily]
    if missing_raw:
        raise ValueError(
            f"raw_daily missing required columns: {', '.join(missing_raw)}"
        )
    calendar = set(adjusted_open.index)
    normalized_targets = {}
    for signal_date, target in targets.items():
        if signal_date not in calendar:
            raise ValueError("target signal date must be in adjusted_open index")
        if not target.index.is_unique:
            raise ValueError("target index must be unique")
        weights = pd.to_numeric(target, errors="coerce")
        if not np.isfinite(weights).all() or weights.lt(0).any():
            raise ValueError("target weights must be finite and nonnegative")
        normalized_targets[signal_date] = pd.Series(
            weights.to_numpy(dtype=float),
            index=target.index,
            dtype=float,
        )
    return cost_value, normalized_targets


def is_one_price_limit(
    day_row: pd.Series,
    previous_close,
) -> tuple[bool, bool]:
    """Return buy/sell blocks for a rounded one-price raw OHLC row."""
    try:
        previous_value = float(previous_close)
    except (TypeError, ValueError):
        return True, True
    if not np.isfinite(previous_value) or previous_value <= 0:
        return True, True
    try:
        prices = day_row.reindex(["open", "high", "low", "close"]).to_numpy(
            dtype=float
        )
    except (AttributeError, TypeError, ValueError):
        return True, True
    prices = np.round(prices, 2)
    if not np.isfinite(prices).all():
        return True, True
    if len(set(prices)) != 1:
        return False, False
    one_price = prices[0]
    price_scale = max(abs(one_price), abs(previous_value), 1.0)
    tolerance = 8 * np.finfo(float).eps * price_scale
    if one_price >= previous_value * 1.045 - tolerance:
        return True, False
    if one_price <= previous_value * 0.955 + tolerance:
        return False, True
    return True, True


def _blocked(raw_daily, dates, position, day, code) -> tuple[bool, bool]:
    if position == 0:
        return True, True
    current_key = (day, code)
    previous_key = (dates[position - 1], code)
    if current_key not in raw_daily.index or previous_key not in raw_daily.index:
        return True, True
    return is_one_price_limit(
        raw_daily.loc[current_key],
        raw_daily.loc[previous_key, "close"],
    )


def _trade_row(
    signal_date,
    day,
    code,
    side,
    shares,
    price,
    notional,
    cost,
):
    return {
        "signal_date": signal_date,
        "date": day,
        "code": code,
        "side": side,
        "shares": shares,
        "price": price,
        "notional": notional,
        "cost": cost,
        "status": "filled",
    }


def _portfolio_value(cash, shares, marks) -> float:
    value = float(cash)
    for code, quantity in shares.items():
        mark = marks.get(code, np.nan)
        if np.isfinite(mark) and mark > 0:
            value += quantity * mark
    return value


def simulate(
    targets,
    adjusted_open: pd.DataFrame,
    raw_daily: pd.DataFrame,
    cost_bps: float = 20,
) -> dict:
    """Execute signal-date targets at the next valid adjusted open."""
    cost_value, normalized_targets = _validate_simulation_inputs(
        targets,
        adjusted_open,
        raw_daily,
        cost_bps,
    )
    targets = normalized_targets
    cost_rate = cost_value / 1e4
    cash = 1.0
    shares = {}
    nav_rows = {}
    turnover_rows = {}
    cost_rows = {}
    trade_rows = []
    dates = adjusted_open.index
    execution_prices = adjusted_open.where(
        np.isfinite(adjusted_open) & adjusted_open.gt(0)
    )
    valuation_prices = execution_prices.ffill()

    for position, day in enumerate(dates):
        prices = execution_prices.loc[day]
        marks = valuation_prices.loc[day]
        nav_before = _portfolio_value(cash, shares, marks)
        traded_notional = 0.0
        day_cost = 0.0
        signal_date = dates[position - 1] if position > 0 else None
        target = targets.get(signal_date) if signal_date is not None else None

        if target is not None:
            desired = target * nav_before
            all_codes = sorted(set(shares) | set(target.index))
            for code in all_codes:
                price = prices.get(code, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                current = shares.get(code, 0.0) * price
                need = desired.get(code, 0.0) - current
                _, sell_blocked = _blocked(
                    raw_daily,
                    dates,
                    position,
                    day,
                    code,
                )
                if need < 0 and not sell_blocked:
                    notional = min(-need, current)
                    fee = notional * cost_rate
                    quantity = notional / price
                    shares[code] = shares.get(code, 0.0) - quantity
                    cash += notional - fee
                    traded_notional += notional
                    day_cost += fee
                    trade_rows.append(
                        _trade_row(
                            signal_date,
                            day,
                            code,
                            "sell",
                            quantity,
                            price,
                            notional,
                            fee,
                        )
                    )

            buy_needs = {}
            for code in target.index:
                price = prices.get(code, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                buy_blocked, _ = _blocked(
                    raw_daily,
                    dates,
                    position,
                    day,
                    code,
                )
                current = shares.get(code, 0.0) * price
                if not buy_blocked and desired[code] > current:
                    buy_needs[code] = desired[code] - current

            required_cash = sum(buy_needs.values()) * (1 + cost_rate)
            scale = (
                min(1.0, cash / required_cash)
                if required_cash > 0
                else 0.0
            )
            for code, need in sorted(buy_needs.items()):
                price = prices[code]
                notional = need * scale
                fee = notional * cost_rate
                quantity = notional / price
                shares[code] = shares.get(code, 0.0) + quantity
                cash -= notional + fee
                traded_notional += notional
                day_cost += fee
                trade_rows.append(
                    _trade_row(
                        signal_date,
                        day,
                        code,
                        "buy",
                        quantity,
                        price,
                        notional,
                        fee,
                    )
                )

        nav_rows[day] = _portfolio_value(cash, shares, marks)
        turnover_rows[day] = (
            traded_notional / nav_before if nav_before > 0 else np.nan
        )
        cost_rows[day] = day_cost

    nav = pd.Series(nav_rows, name="nav", dtype=float)
    returns = nav / nav.shift(1) - 1.0
    if not returns.empty:
        returns.iloc[0] = 0.0
    returns.name = "returns"
    return {
        "nav": nav,
        "returns": returns,
        "turnover": pd.Series(turnover_rows, name="turnover", dtype=float),
        "cost": pd.Series(cost_rows, name="cost", dtype=float),
        "trades": pd.DataFrame(trade_rows, columns=TRADE_COLUMNS),
    }


def portfolio_metrics(strategy: dict, benchmark: dict) -> dict[str, float]:
    """Summarize net strategy and benchmark return streams."""
    aligned = pd.concat(
        {
            "strategy": strategy["returns"],
            "benchmark": benchmark["returns"],
        },
        axis=1,
        join="inner",
    )
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {
            "strategy_total": np.nan,
            "benchmark_total": np.nan,
            "strategy_annual": np.nan,
            "benchmark_annual": np.nan,
            "annual_excess": np.nan,
            "sharpe": np.nan,
            "information_ratio": np.nan,
            "max_drawdown": np.nan,
            "monthly_win_rate": np.nan,
            "excess_nw_t": np.nan,
            "annual_turnover": np.nan,
        }

    strategy_returns = aligned["strategy"]
    benchmark_returns = aligned["benchmark"]
    excess = strategy_returns - benchmark_returns
    sample_size = len(aligned)
    strategy_total = float((1 + strategy_returns).prod() - 1)
    benchmark_total = float((1 + benchmark_returns).prod() - 1)

    def annualized(total):
        return float((1 + total) ** (252 / sample_size) - 1)

    strategy_annual = annualized(strategy_total)
    benchmark_annual = annualized(benchmark_total)
    strategy_std = strategy_returns.std(ddof=0)
    excess_std = excess.std(ddof=0)
    sharpe = (
        float(strategy_returns.mean() / strategy_std * np.sqrt(252))
        if np.isfinite(strategy_std) and strategy_std > 0
        else np.nan
    )
    information_ratio = (
        float(excess.mean() / excess_std * np.sqrt(252))
        if np.isfinite(excess_std) and excess_std > 0
        else np.nan
    )
    equity = (1 + strategy_returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1).min())
    monthly = (
        (1 + aligned)
        .groupby(aligned.index.to_period("M"))
        .prod()
        - 1
    )
    monthly_win_rate = float(
        monthly["strategy"].gt(monthly["benchmark"]).mean()
    )
    turnover = (
        pd.Series(strategy["turnover"], dtype=float)
        .reindex(aligned.index)
        .replace([np.inf, -np.inf], np.nan)
    )
    annual_turnover = (
        float(turnover.sum(min_count=1) * 252 / sample_size)
        if turnover.notna().any()
        else np.nan
    )
    return {
        "strategy_total": strategy_total,
        "benchmark_total": benchmark_total,
        "strategy_annual": strategy_annual,
        "benchmark_annual": benchmark_annual,
        "annual_excess": strategy_annual - benchmark_annual,
        "sharpe": sharpe,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
        "monthly_win_rate": monthly_win_rate,
        "excess_nw_t": newey_west_t(excess, lags=4),
        "annual_turnover": annual_turnover,
    }
