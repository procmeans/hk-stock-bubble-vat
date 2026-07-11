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
COST_BPS = 20.0
UNIVERSE_SIZE = 500
FETCH_SIZE = 800
WINDOW_DAYS = 400
PAPER_DIR = Path("paper")


def account_dir(account: str) -> Path:
    return PAPER_DIR / account


def load_state(account: str) -> dict:
    return json.loads((account_dir(account) / "state.json").read_text())


def save_state(account: str, state: dict) -> None:
    (account_dir(account) / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1))


def init(account: str = "us_momentum", capital: float = 100000.0) -> None:
    directory = account_dir(account)
    if (directory / "state.json").exists():
        raise SystemExit(f"账户已存在: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    save_state(account, {
        "account": account, "capital": capital, "cash": capital,
        "positions": {}, "pending_targets": None,
        "days_since_rebalance": None, "bench_nav": capital, "last_run": None,
    })
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


def step(state: dict, close, params=PARAMS, cost_bps=COST_BPS):
    """处理 close 的最新一天;返回 (state, nav_row|None, orders)。幂等。"""
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
        targets = target_weights(close, **params)
        if targets:
            state["pending_targets"] = targets
            state["days_since_rebalance"] = 0

    state["last_run"] = today_str
    return state, nav_row, orders


def universe_tickers(snapshot_dir: Path = Path("data")) -> list:
    """当日美股快照(Actions 每日维护)按市值取前 FETCH_SIZE。"""
    from alpha101.yf_history import default_universe
    rows = json.loads(default_universe("us", snapshot_dir).read_text())
    top = sorted(rows, key=lambda row: row.get("mc") or 0, reverse=True)[:FETCH_SIZE]
    return [row["code"] for row in top]


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


def run(account: str = "us_momentum", fetch=None) -> None:
    fetch = fetch or fetch_close_volume
    state = load_state(account)
    held = set(state["positions"]) | set(state["pending_targets"] or {})
    codes = sorted(set(universe_tickers()) | held)
    close, volume = fetch(codes, window_days=WINDOW_DAYS)
    adv = (close * volume).tail(60).mean().nlargest(UNIVERSE_SIZE).index
    keep = set(adv) | held
    pool = close[[code for code in close.columns if code in keep]]

    state, nav_row, orders = step(state, pool)
    if nav_row is None:
        print(f"{state['last_run']} 已运行过,跳过")
        return
    directory = account_dir(account)
    pd.DataFrame([nav_row]).to_csv(directory / "nav.csv", mode="a",
                                   header=False, index=False)
    if orders:
        pd.DataFrame(orders).to_csv(directory / "orders.csv", mode="a",
                                    header=False, index=False)
    save_state(account, state)
    pending = len(state["pending_targets"] or {})
    print(f"{nav_row['date']}: NAV {nav_row['nav']:,.2f} "
          f"(基准 {nav_row['bench_nav']:,.2f}),成交 {len(orders)} 笔,"
          f"挂单 {pending} 只")


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
    parser.add_argument("cmd", choices=["init", "run", "status"])
    parser.add_argument("--account", default="us_momentum")
    parser.add_argument("--capital", type=float, default=100000.0)
    args = parser.parse_args()
    if args.cmd == "init":
        init(args.account, args.capital)
    elif args.cmd == "run":
        run(args.account)
    else:
        status(args.account)


if __name__ == "__main__":
    main()
