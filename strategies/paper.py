"""模拟交易(paper trading):美股 momentum 纸面实盘,每日步进。

时序与回测一致:T 日收盘出信号(pending_targets),T+1 日收盘成交。
状态文件化(paper/<account>/),同日重复运行幂等,GitHub Actions 每日执行。
纸面简化:允许碎股、收盘价成交、无冲击成本;退市(无价)持仓按 0 冲销。
"""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategies import momentum, ths_heat

PARAMS = {"top_n": 40, "lookback": 126, "skip": 21, "rebalance": 21}
A101_PARAMS = {"top_n": 50, "rebalance": 5}
EW_PARAMS = {"top_n": 500, "rebalance": 5}
THS_HEAT_PARAMS = {"top_n": 20, "rebalance": 2}
HEAT_STRATEGIES = {"ths_heat", "ths_heat_rise"}
DEFAULT_PARAMS = {"momentum": PARAMS, "alpha101": A101_PARAMS,
                  "equal_weight": EW_PARAMS, "ths_heat": THS_HEAT_PARAMS,
                  "ths_heat_rise": THS_HEAT_PARAMS}
COST_BPS = 20.0
UNIVERSE_SIZE = 500
FETCH_SIZE = 800
WINDOW_DAYS = 400
PAPER_DIR = Path("paper")
CURRENCIES = {"us": "$", "hk": "HK$", "a": "¥"}
HEAT_SIGNAL_COLUMNS = [
    "date", "strategy", "rank", "ticker", "name", "factor_value",
    "status", "error",
]
ERROR_SUMMARY_LIMIT = 300
_CREDENTIAL_FIELD = r"(?:refresh_token|access_token|authorization)"
_QUOTED_CREDENTIAL_RE = re.compile(
    rf"([\"']?{_CREDENTIAL_FIELD}[\"']?\s*[:=]\s*)([\"'])(.*?)\2",
    flags=re.IGNORECASE | re.DOTALL,
)
_PLAIN_CREDENTIAL_RE = re.compile(
    rf"(\b{_CREDENTIAL_FIELD}\b\s*[:=]\s*)(?:Bearer\s+)?[^\s,;}}\]]+",
    flags=re.IGNORECASE,
)
_BEARER_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+", flags=re.IGNORECASE
)
_TOKEN_LIKE_RE = re.compile(r"(?<![A-Za-z0-9._~+/=-])[A-Za-z0-9._~+/=-]{24,}")


def account_dir(account: str) -> Path:
    return PAPER_DIR / account


def load_state(account: str) -> dict:
    return json.loads((account_dir(account) / "state.json").read_text())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_state(account: str, state: dict) -> None:
    _atomic_write_text(
        account_dir(account) / "state.json",
        json.dumps(state, ensure_ascii=False, indent=1),
    )


def register_account(account: str, title: str = None, currency: str = "$") -> None:
    path = PAPER_DIR / "accounts.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    if not any(entry["account"] == account for entry in entries):
        entries.append({"account": account, "title": title or account,
                        "currency": currency})
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=1))


def init(account: str = "us_momentum", capital: float = 100000.0,
         strategy: str = "momentum", market: str = "us",
         params: dict = None, title: str = None) -> None:
    directory = account_dir(account)
    if (directory / "state.json").exists():
        raise SystemExit(f"账户已存在: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    save_state(account, {
        "account": account, "capital": capital, "cash": capital,
        "positions": {}, "pending_targets": None,
        "days_since_rebalance": None, "bench_nav": capital, "last_run": None,
        "strategy": strategy, "market": market,
        "params": params or DEFAULT_PARAMS.get(strategy, PARAMS),
    })
    register_account(account, title=title, currency=CURRENCIES.get(market, "$"))
    pd.DataFrame(columns=["date", "nav", "cash", "positions_value", "bench_nav"]) \
        .to_csv(directory / "nav.csv", index=False)
    pd.DataFrame(columns=["date", "ticker", "side", "shares", "price", "value", "cost"]) \
        .to_csv(directory / "orders.csv", index=False)
    print(f"账户 {account} 已创建,虚拟资金 {capital:,.0f}")


def target_weights(close, top_n, lookback, skip, **_) -> dict:
    latest = momentum.score(close, lookback, skip).iloc[-1].dropna()
    top = latest.nlargest(top_n)
    if top.empty:
        return {}
    return {ticker: 1.0 / len(top) for ticker in top.index}


def positions_value(positions: dict, prices) -> float:
    return float(sum(
        shares * prices[ticker]
        for ticker, shares in positions.items()
        if ticker in prices.index and not np.isnan(prices[ticker])
    ))


def compute_targets(strategy, panel, params, target_override=None) -> dict:
    if target_override is not None:
        return target_override
    if strategy in HEAT_STRATEGIES:
        return {}
    if strategy == "alpha101":
        from strategies.alpha101_composite import targets
        return targets(panel, **params)
    if strategy == "equal_weight":
        from strategies.equal_weight import targets
        return targets(panel, **params)
    return target_weights(panel["close"], **params)


def rebalance_due(state: dict, params=None) -> bool:
    params = params or state.get("params") or PARAMS
    days = state.get("days_since_rebalance")
    return days is None or days + 1 >= params["rebalance"]


def step(state, panel, params=None, cost_bps=COST_BPS, target_override=None):
    """处理 panel(至少含 close)的最新一天;返回 (state, nav_row|None, orders)。幂等。"""
    close = panel["close"]
    params = params or state.get("params") or PARAMS
    today = close.index[-1]
    today_str = today.strftime("%Y-%m-%d")
    if state["last_run"] == today_str:
        return state, None, []
    valuation_prices = close.ffill().iloc[-1]
    execution_prices = close.iloc[-1]

    orders = []
    if state["pending_targets"] is not None:
        nav_before = state["cash"] + positions_value(
            state["positions"], valuation_prices
        )
        targets = state["pending_targets"]
        for ticker in sorted(set(targets) | set(state["positions"])):
            price = execution_prices.get(ticker, np.nan)
            if np.isnan(price) or price <= 0:
                continue                                # 无价:跳过,持仓保留
            current_value = state["positions"].get(ticker, 0.0) * price
            delta_value = targets.get(ticker, 0.0) * nav_before - current_value
            if abs(delta_value) < 1e-9:
                continue
            shares = delta_value / price
            cost = abs(delta_value) * cost_bps / 1e4
            state["cash"] -= delta_value + cost
            new_shares = state["positions"].get(ticker, 0.0) + shares
            if abs(new_shares) < 1e-9:
                state["positions"].pop(ticker, None)
            else:
                state["positions"][ticker] = new_shares
            orders.append({
                "date": today_str, "ticker": ticker,
                "side": "buy" if shares > 0 else "sell",
                "shares": round(shares, 4), "price": round(float(price), 4),
                "value": round(delta_value, 2), "cost": round(cost, 2),
            })
        state["pending_targets"] = None

    value = positions_value(state["positions"], valuation_prices)
    nav = state["cash"] + value
    benchmark_close = panel.get("benchmark_close", close)
    daily = benchmark_close.ffill().pct_change().iloc[-1]
    bench_ret = float(np.nanmean(daily)) if not daily.isna().all() else 0.0
    state["bench_nav"] = float(state["bench_nav"] * (1.0 + bench_ret))
    nav_row = {
        "date": today_str, "nav": round(nav, 2), "cash": round(state["cash"], 2),
        "positions_value": round(value, 2),
        "bench_nav": round(state["bench_nav"], 2),
    }

    if state["days_since_rebalance"] is None:
        due = True
    else:
        state["days_since_rebalance"] += 1
        due = state["days_since_rebalance"] >= params["rebalance"]
    if due:
        new_targets = compute_targets(
            state.get("strategy", "momentum"), panel, params,
            target_override=target_override,
        )
        if new_targets:
            state["pending_targets"] = new_targets
            state["days_since_rebalance"] = 0

    state["last_run"] = today_str
    return state, nav_row, orders


def universe_tickers(snapshot_dir: Path = Path("data")) -> list:
    """当日美股快照(Actions 每日维护)按市值取前 FETCH_SIZE。"""
    from alpha101.yf_history import default_universe
    rows = json.loads(default_universe("us", snapshot_dir).read_text())
    top = sorted(rows, key=lambda row: row.get("mc") or 0, reverse=True)[:FETCH_SIZE]
    return [row["code"] for row in top]


def a_universe_tickers(snapshot_dir: Path = Path("data")) -> list:
    """当日 A 股快照按市值取前 FETCH_SIZE。"""
    dates = json.loads((snapshot_dir / "manifest_a.json").read_text())["dates"]
    rows = json.loads((snapshot_dir / f"a-{max(dates)}.json").read_text())
    top = sorted(rows, key=lambda row: row.get("mc") or 0, reverse=True)[:FETCH_SIZE]
    return [str(row["code"]).zfill(6) for row in top]


def fetch_a_panel(codes, window_days: int = WINDOW_DAYS) -> dict:
    """iFinD 抓 OHLCV+amount 并构建 alpha101 口径面板。"""
    from alpha101 import ths_http
    from alpha101.ths_history import build_panel, normalize_history_frame
    from alpha101.ths_today import chunks, to_thscode

    start = (pd.Timestamp.today() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    token = ths_http.get_access_token()
    frames = []
    for batch in chunks([to_thscode(code) for code in codes], 25):
        data = ths_http.history_quotation(
            batch, "open,high,low,close,volume,amount", start, end,
            access_token=token)
        if not data.empty:
            frames.append(normalize_history_frame(data))
    if not frames:
        raise RuntimeError("iFinD 未返回任何数据")
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    return build_panel(raw)


def fetch_close_volume(codes, window_days: int = WINDOW_DAYS):
    import yfinance as yf
    from alpha101.yf_history import normalize_download, to_yf_ticker
    from alpha101.ths_today import chunks

    start = (pd.Timestamp.today() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    code_map = {to_yf_ticker(code, "us"): code for code in codes}
    frames = []
    for batch in chunks(list(code_map), 100):
        data = yf.download(batch, start=start, auto_adjust=True,
                           group_by="column", progress=False, threads=True)
        frame = normalize_download(data, {t: code_map[t] for t in batch})
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("yfinance 未返回任何数据")
    raw = pd.concat(frames, ignore_index=True)
    close = raw.pivot(index="date", columns="code", values="close").sort_index()
    volume = raw.pivot(index="date", columns="code", values="volume").sort_index()
    return close, volume


def _market_panel(market: str, held: set, fetch=None) -> dict:
    if market == "a":
        snapshot_codes = set(a_universe_tickers())
        codes = sorted(snapshot_codes | held)
        panel = (fetch or fetch_a_panel)(codes, window_days=WINDOW_DAYS)
        benchmark_codes = panel["amount"].reindex(
            columns=sorted(snapshot_codes)
        ).tail(60).mean() \
            .nlargest(UNIVERSE_SIZE).index.tolist()
        keep = set(benchmark_codes) | held
        result = {
            key: value[[column for column in value.columns if column in keep]]
            if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
        result["benchmark_close"] = panel["close"].reindex(
            columns=benchmark_codes
        )
        return result
    codes = sorted(set(universe_tickers()) | held)
    close, volume = (fetch or fetch_close_volume)(codes, window_days=WINDOW_DAYS)
    adv = (close * volume).tail(60).mean().nlargest(UNIVERSE_SIZE).index
    keep = set(adv) | held
    return {"close": close[[code for code in close.columns if code in keep]]}


def _held(state: dict) -> set:
    return set(state["positions"]) | set(state["pending_targets"] or {})


def _merge_panel(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if (isinstance(value, pd.DataFrame)
                and isinstance(merged.get(key), pd.DataFrame)):
            merged[key] = merged[key].combine_first(value)
        else:
            merged.setdefault(key, value)
    return merged


def _fetch_supplemental(panel, tickers, fetch_panel):
    tickers = sorted(set(tickers))
    if not tickers:
        return panel, {}
    try:
        extra = fetch_panel(tickers, window_days=WINDOW_DAYS)
        return _merge_panel(panel, extra), {}
    except Exception as error:
        if len(tickers) == 1:
            return panel, {tickers[0]: error}
        middle = len(tickers) // 2
        panel, left_errors = _fetch_supplemental(
            panel, tickers[:middle], fetch_panel
        )
        panel, right_errors = _fetch_supplemental(
            panel, tickers[middle:], fetch_panel
        )
        return panel, {**left_errors, **right_errors}


def sanitize_error(error, limit=ERROR_SUMMARY_LIMIT):
    text = str(error)
    configured_token = os.getenv("THS_HTTP_REFRESH_TOKEN")
    if configured_token:
        text = text.replace(configured_token, "[REDACTED]")
    text = _QUOTED_CREDENTIAL_RE.sub(r"\1[REDACTED]", text)
    text = _PLAIN_CREDENTIAL_RE.sub(r"\1[REDACTED]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _TOKEN_LIKE_RE.sub("[REDACTED]", text)
    return " ".join(text.split())[:limit]


def _error_row(date, strategy, error):
    return {
        "date": date, "strategy": strategy, "rank": "", "ticker": "",
        "name": "", "factor_value": "", "status": "error",
        "error": sanitize_error(error),
    }


def append_heat_signals(rows, path=None):
    if not rows:
        return
    path = path or PAPER_DIR / "ths_heat_signals.csv"
    incoming = pd.DataFrame(rows, columns=HEAT_SIGNAL_COLUMNS)
    existing = pd.read_csv(path, dtype={"ticker": str}) \
        if path.exists() and path.stat().st_size else pd.DataFrame()
    combined = pd.concat([existing, incoming], ignore_index=True)
    dedupe_columns = ["date", "strategy", "rank", "ticker", "status"]
    combined[dedupe_columns] = combined[dedupe_columns].fillna("")
    combined = combined.drop_duplicates(dedupe_columns, keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def _upsert_csv(path, rows, key_columns, columns):
    if not rows:
        return
    incoming = pd.DataFrame(rows, columns=columns)
    key_dtypes = {column: str for column in key_columns}
    existing = pd.read_csv(path, dtype=key_dtypes) \
        if path.exists() and path.stat().st_size else pd.DataFrame(columns=columns)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined[key_columns] = combined[key_columns].fillna("").astype(str)
    combined = combined.drop_duplicates(key_columns, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        combined.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_heat_targets(states, panel, fetch_panel, fetch_signal=None):
    signal_date = panel["close"].index[-1]
    date_text = signal_date.strftime("%Y-%m-%d")
    due = {
        account: state for account, state in states.items()
        if state.get("strategy") in HEAT_STRATEGIES
        and state.get("last_run") != date_text and rebalance_due(state)
    }
    if not due:
        return panel, {}, []
    loader = fetch_signal or ths_heat.fetch_signal
    signals, audit = {}, []
    for strategy in sorted({state["strategy"] for state in due.values()}):
        top_n = next(
            state["params"]["top_n"] for state in due.values()
            if state["strategy"] == strategy
        )
        try:
            signals[strategy] = loader(signal_date, strategy, top_n=top_n)
        except Exception as error:
            audit.append(_error_row(date_text, strategy, error))
    quote_errors = {}
    for strategy in sorted(signals):
        missing = sorted({
            ticker for ticker in signals[strategy]["ticker"].astype(str)
            if ticker not in panel["close"].columns
        })
        panel, quote_errors[strategy] = _fetch_supplemental(
            panel, missing, fetch_panel
        )
    prices = panel["close"].iloc[-1]
    overrides = {}
    for account, state in due.items():
        strategy = state["strategy"]
        signal = signals.get(strategy)
        if signal is None:
            overrides[account] = {}
            continue
        weights = ths_heat.target_weights(signal, prices)
        overrides[account] = weights
        used = signal[signal["ticker"].isin(weights)]
        invalid = [
            ticker for ticker in signal["ticker"].astype(str)
            if ticker not in weights
        ]
        if invalid:
            details = []
            for ticker in invalid:
                error = quote_errors.get(strategy, {}).get(ticker)
                details.append(
                    f"{ticker}: {error}" if error is not None
                    else f"{ticker}: no valid latest close"
                )
            audit.append(_error_row(
                date_text, strategy, RuntimeError("; ".join(details))
            ))
        if not used.empty:
            audit.extend(
                {**row, "status": "ok", "error": ""}
                for row in used.to_dict("records")
            )
    return panel, overrides, audit


def _step_and_persist(account, state, panel, target_override=None):
    state, nav_row, orders = step(
        state, panel, target_override=target_override
    )
    if nav_row is None:
        print(f"{account}: {state['last_run']} 已运行过,跳过")
        return
    directory = account_dir(account)
    _upsert_csv(
        directory / "nav.csv", [nav_row], ["date"],
        ["date", "nav", "cash", "positions_value", "bench_nav"],
    )
    _upsert_csv(
        directory / "orders.csv", orders, ["date", "ticker", "side"],
        ["date", "ticker", "side", "shares", "price", "value", "cost"],
    )
    save_state(account, state)
    pending = len(state["pending_targets"] or {})
    print(f"{account} {nav_row['date']}: NAV {nav_row['nav']:,.2f} "
          f"(基准 {nav_row['bench_nav']:,.2f}),成交 {len(orders)} 笔,"
          f"挂单 {pending} 只")


def run(account="us_momentum", fetch=None, heat_fetch=None):
    state = load_state(account)
    market = state.get("market", "us")
    panel = _market_panel(market, _held(state), fetch=fetch)
    overrides, audit = {}, []
    if market == "a":
        panel, overrides, audit = prepare_heat_targets(
            {account: state}, panel, fetch or fetch_a_panel,
            fetch_signal=heat_fetch,
        )
        append_heat_signals(audit)
    _step_and_persist(account, state, panel, overrides.get(account))


def run_market(market, fetch=None, heat_fetch=None):
    """同市场账户共用基础抓数，热度策略仅在到期日预取信号。"""
    manifest = PAPER_DIR / "accounts.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    states = {
        entry["account"]: load_state(entry["account"])
        for entry in entries
        if load_state(entry["account"]).get("market", "us") == market
    }
    if not states:
        raise SystemExit(f"无 {market} 市场账户")
    held = set()
    for state in states.values():
        held |= _held(state)
    panel = _market_panel(market, held, fetch=fetch)
    overrides, audit = {}, []
    if market == "a":
        panel, overrides, audit = prepare_heat_targets(
            states, panel, fetch or fetch_a_panel, fetch_signal=heat_fetch
        )
        append_heat_signals(audit)
    failures = []
    for account, state in states.items():
        try:
            _step_and_persist(account, state, panel, overrides.get(account))
        except Exception as error:
            failures.append((account, error))
    if failures:
        summary = "; ".join(
            f"{account}: {sanitize_error(error)}"
            for account, error in failures
        )
        raise RuntimeError(f"paper account failures: {summary}")


def status(account: str = "us_momentum") -> None:
    state = load_state(account)
    nav = pd.read_csv(account_dir(account) / "nav.csv")
    print(f"账户 {account}:现金 {state['cash']:,.2f},"
          f"持仓 {len(state['positions'])} 只,"
          f"挂单 {len(state['pending_targets'] or {})} 只")
    if not nav.empty:
        print(nav.tail(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["init", "run", "run-market", "status"])
    parser.add_argument("--account", default="us_momentum")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--strategy", default="momentum",
                        choices=sorted(DEFAULT_PARAMS))
    parser.add_argument("--market", default="us", choices=["us", "hk", "a"])
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    if args.cmd == "init":
        init(args.account, args.capital, strategy=args.strategy,
             market=args.market, title=args.title)
    elif args.cmd == "run":
        run(args.account)
    elif args.cmd == "run-market":
        run_market(args.market)
    else:
        status(args.account)


if __name__ == "__main__":
    main()
