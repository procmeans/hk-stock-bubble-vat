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
