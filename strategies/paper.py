"""模拟交易(paper trading):美股 momentum 纸面实盘,每日步进。

时序与回测一致:T 日收盘出信号(pending_targets),T+1 日收盘成交。
状态文件化(paper/<account>/),同日重复运行幂等,GitHub Actions 每日执行。
纸面简化:允许碎股、收盘价成交、无冲击成本;退市(无价)持仓按 0 冲销。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies import momentum

PARAMS = {"top_n": 40, "lookback": 126, "skip": 21, "rebalance": 21}
A101_PARAMS = {"top_n": 50, "rebalance": 5}
EW_PARAMS = {"top_n": 500, "rebalance": 5}
DEFAULT_PARAMS = {"momentum": PARAMS, "alpha101": A101_PARAMS,
                  "equal_weight": EW_PARAMS}
COST_BPS = 20.0
UNIVERSE_SIZE = 500
FETCH_SIZE = 800
WINDOW_DAYS = 400
PAPER_DIR = Path("paper")
CURRENCIES = {"us": "$", "hk": "HK$", "a": "¥"}


def account_dir(account: str) -> Path:
    return PAPER_DIR / account


def load_state(account: str) -> dict:
    return json.loads((account_dir(account) / "state.json").read_text())


def save_state(account: str, state: dict) -> None:
    (account_dir(account) / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1))


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


def compute_targets(strategy: str, panel: dict, params: dict) -> dict:
    if strategy == "alpha101":
        from strategies.alpha101_composite import targets
        return targets(panel, **params)
    if strategy == "equal_weight":
        from strategies.equal_weight import targets
        return targets(panel, **params)
    return target_weights(panel["close"], **params)


def step(state: dict, panel: dict, params=None, cost_bps=COST_BPS):
    """处理 panel(至少含 close)的最新一天;返回 (state, nav_row|None, orders)。幂等。"""
    close = panel["close"]
    params = params or state.get("params") or PARAMS
    today = close.index[-1]
    today_str = today.strftime("%Y-%m-%d")
    if state["last_run"] == today_str:
        return state, None, []
    prices = close.ffill().iloc[-1]

    orders = []
    if state["pending_targets"] is not None:
        nav_before = state["cash"] + positions_value(state["positions"], prices)
        targets = state["pending_targets"]
        for ticker in sorted(set(targets) | set(state["positions"])):
            price = prices.get(ticker, np.nan)
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

    value = positions_value(state["positions"], prices)
    nav = state["cash"] + value
    daily = close.ffill().pct_change().iloc[-1]
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
        new_targets = compute_targets(state.get("strategy", "momentum"), panel, params)
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
        codes = sorted(set(a_universe_tickers()) | held)
        panel = (fetch or fetch_a_panel)(codes, window_days=WINDOW_DAYS)
        adv = panel["amount"].tail(60).mean().nlargest(UNIVERSE_SIZE).index
        keep = set(adv) | held
        return {key: value[[c for c in value.columns if c in keep]]
                if isinstance(value, pd.DataFrame) else value
                for key, value in panel.items()}
    codes = sorted(set(universe_tickers()) | held)
    close, volume = (fetch or fetch_close_volume)(codes, window_days=WINDOW_DAYS)
    adv = (close * volume).tail(60).mean().nlargest(UNIVERSE_SIZE).index
    keep = set(adv) | held
    return {"close": close[[code for code in close.columns if code in keep]]}


def _held(state: dict) -> set:
    return set(state["positions"]) | set(state["pending_targets"] or {})


def _step_and_persist(account: str, state: dict, panel: dict) -> None:
    state, nav_row, orders = step(state, panel)
    if nav_row is None:
        print(f"{account}: {state['last_run']} 已运行过,跳过")
        return
    directory = account_dir(account)
    pd.DataFrame([nav_row]).to_csv(directory / "nav.csv", mode="a",
                                   header=False, index=False)
    if orders:
        pd.DataFrame(orders).to_csv(directory / "orders.csv", mode="a",
                                    header=False, index=False)
    save_state(account, state)
    pending = len(state["pending_targets"] or {})
    print(f"{account} {nav_row['date']}: NAV {nav_row['nav']:,.2f} "
          f"(基准 {nav_row['bench_nav']:,.2f}),成交 {len(orders)} 笔,"
          f"挂单 {pending} 只")


def run(account: str = "us_momentum", fetch=None) -> None:
    state = load_state(account)
    panel = _market_panel(state.get("market", "us"), _held(state), fetch=fetch)
    _step_and_persist(account, state, panel)


def run_market(market: str, fetch=None) -> None:
    """同市场所有账户共用一次抓数(iFinD 配额不随账户数增长)。"""
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
    for account, state in states.items():
        _step_and_persist(account, state, panel)


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
