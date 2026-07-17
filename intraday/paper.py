"""Publish the latest intraday validation output as a paper-account snapshot."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("output/intraday_6m")
DEFAULT_PAPER_DIR = Path("paper")
DEFAULT_ACCOUNT = "a_intraday_6m"
DEFAULT_TITLE = "A股 分钟线量价因子"
DEFAULT_CURRENCY = "¥"
ACCOUNT_COLUMNS = ["date", "nav", "cash", "positions_value", "bench_nav"]
ORDER_COLUMNS = ["date", "ticker", "side", "shares", "price", "value", "cost"]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_csv(path: Path, *, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {', '.join(missing)}"
        )
    return frame


def _load_nav(input_dir: Path) -> pd.DataFrame:
    nav = _load_csv(
        input_dir / "portfolio_nav.csv",
        required=["date", "strategy_net", "benchmark_net"],
    ).copy()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce", format="mixed")
    if nav["date"].isna().any():
        raise ValueError("portfolio_nav.csv contains invalid date")
    nav = nav.sort_values("date", kind="mergesort").reset_index(drop=True)
    nav["strategy_net"] = pd.to_numeric(nav["strategy_net"], errors="coerce")
    nav["benchmark_net"] = pd.to_numeric(nav["benchmark_net"], errors="coerce")
    if nav[["strategy_net", "benchmark_net"]].isna().any().any():
        raise ValueError("portfolio_nav.csv contains non-numeric nav values")
    return nav


def _load_trades(input_dir: Path) -> pd.DataFrame:
    trades = _load_csv(
        input_dir / "trades.csv",
        required=[
            "portfolio",
            "date",
            "code",
            "side",
            "shares",
            "price",
            "notional",
            "cost",
        ],
    ).copy()
    trades = trades.loc[trades["portfolio"].eq("strategy")].copy()
    if trades.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce", format="mixed")
    if trades["date"].isna().any():
        raise ValueError("trades.csv contains invalid date")
    trades["code"] = trades["code"].astype(str).str.zfill(6)
    for column in ["shares", "price", "notional", "cost"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    if trades[["shares", "price", "notional", "cost"]].isna().any().any():
        raise ValueError("trades.csv contains non-numeric trade values")
    trades = trades.sort_values(
        ["date", "code", "side"], kind="mergesort"
    ).reset_index(drop=True)
    orders = trades.rename(columns={"code": "ticker", "notional": "value"})
    orders["date"] = orders["date"].dt.strftime("%Y-%m-%d")
    return orders[ORDER_COLUMNS]


def _replay_state(
    nav: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    capital: float,
) -> tuple[pd.DataFrame, dict]:
    cash = float(capital)
    positions: dict[str, float] = {}
    order_frame = orders.copy()
    order_frame["date"] = pd.to_datetime(
        order_frame["date"], errors="coerce", format="mixed"
    )
    grouped = {
        day: frame.reset_index(drop=True)
        for day, frame in order_frame.groupby("date", sort=True)
    }
    rows = []
    for row in nav.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        for trade in grouped.get(day, pd.DataFrame()).itertuples(index=False):
            if trade.side == "buy":
                cash -= float(trade.value) + float(trade.cost)
                positions[trade.ticker] = positions.get(trade.ticker, 0.0) + float(
                    trade.shares
                )
            elif trade.side == "sell":
                cash += float(trade.value) - float(trade.cost)
                positions[trade.ticker] = positions.get(trade.ticker, 0.0) - float(
                    trade.shares
                )
            else:
                raise ValueError(f"unsupported trade side: {trade.side}")
            if abs(positions[trade.ticker]) < 1e-9:
                positions.pop(trade.ticker, None)
        strategy_nav = float(row.strategy_net)
        bench_nav = float(row.benchmark_net)
        positions_value = strategy_nav - cash
        rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "nav": round(strategy_nav, 2),
            "cash": round(cash, 2),
            "positions_value": round(positions_value, 2),
            "bench_nav": round(bench_nav, 2),
        })
    nav_frame = pd.DataFrame(rows, columns=ACCOUNT_COLUMNS)
    state = {
        "account": DEFAULT_ACCOUNT,
        "capital": float(capital),
        "cash": round(cash, 2),
        "positions": {ticker: round(quantity, 6) for ticker, quantity in sorted(positions.items())},
        "pending_targets": None,
        "days_since_rebalance": None,
        "bench_nav": round(float(nav.iloc[-1]["benchmark_net"]), 2),
        "last_run": pd.Timestamp(nav.iloc[-1]["date"]).strftime("%Y-%m-%d"),
        "strategy": "intraday_factor",
        "market": "a",
        "params": {
            "top": 500,
            "top_n": 50,
            "rebalance": 5,
            "cost_bps": 20.0,
            "min_count": 400,
        },
    }
    return nav_frame, state


def publish(
    input_dir: Path = DEFAULT_INPUT,
    paper_dir: Path = DEFAULT_PAPER_DIR,
    account: str = DEFAULT_ACCOUNT,
    title: str = DEFAULT_TITLE,
    currency: str = DEFAULT_CURRENCY,
    capital: float = 100000.0,
) -> Path:
    """Publish the latest intraday validation output as a paper account."""
    input_dir = Path(input_dir)
    paper_dir = Path(paper_dir)
    nav = _load_nav(input_dir)
    orders = _load_trades(input_dir)
    nav_frame, state = _replay_state(nav, orders, capital=capital)

    account_dir = paper_dir / account
    account_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(nav_frame, account_dir / "nav.csv")
    _atomic_write_csv(orders, account_dir / "orders.csv")
    _atomic_write_text(
        account_dir / "state.json",
        json.dumps(state, ensure_ascii=False, indent=1) + "\n",
    )

    manifest_path = paper_dir / "accounts.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []
    if not any(entry.get("account") == account for entry in manifest):
        manifest.append({
            "account": account,
            "title": title,
            "currency": currency,
        })
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        )
    return account_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["publish"])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--currency", default=DEFAULT_CURRENCY)
    parser.add_argument("--capital", type=float, default=100000.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "publish":
        publish(
            input_dir=args.input,
            paper_dir=args.paper_dir,
            account=args.account,
            title=args.title,
            currency=args.currency,
            capital=args.capital,
        )


if __name__ == "__main__":
    main()
