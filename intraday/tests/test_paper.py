import json
from pathlib import Path

import pandas as pd
import pytest

from intraday.paper import DEFAULT_ACCOUNT, publish


def _write_input(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2026-07-08", "strategy_net": 100000.0, "benchmark_net": 100000.0},
            {"date": "2026-07-09", "strategy_net": 101000.0, "benchmark_net": 100500.0},
        ]
    ).to_csv(output_dir / "portfolio_nav.csv", index=False)
    pd.DataFrame(
        [
            {
                "portfolio": "strategy",
                "signal_date": "2026-07-08",
                "date": "2026-07-09",
                "code": "000001",
                "side": "buy",
                "shares": 100.0,
                "price": 10.0,
                "notional": 1000.0,
                "cost": 2.0,
            },
            {
                "portfolio": "benchmark",
                "signal_date": "2026-07-08",
                "date": "2026-07-09",
                "code": "000002",
                "side": "buy",
                "shares": 100.0,
                "price": 10.0,
                "notional": 1000.0,
                "cost": 2.0,
            },
        ]
    ).to_csv(output_dir / "trades.csv", index=False)


def test_publish_writes_paper_account_files(tmp_path):
    input_dir = tmp_path / "output" / "intraday_6m"
    paper_dir = tmp_path / "paper"
    _write_input(input_dir)

    account_dir = publish(input_dir=input_dir, paper_dir=paper_dir)

    assert account_dir == paper_dir / DEFAULT_ACCOUNT
    nav = pd.read_csv(account_dir / "nav.csv")
    orders = pd.read_csv(account_dir / "orders.csv", dtype={"ticker": str})
    state = json.loads((account_dir / "state.json").read_text(encoding="utf-8"))
    assert nav.columns.tolist() == [
        "date", "nav", "cash", "positions_value", "bench_nav"
    ]
    assert orders.columns.tolist() == [
        "date", "ticker", "side", "shares", "price", "value", "cost"
    ]
    assert nav.to_dict("records") == [
        {
            "date": "2026-07-08",
            "nav": 100000.0,
            "cash": 100000.0,
            "positions_value": 0.0,
            "bench_nav": 100000.0,
        },
        {
            "date": "2026-07-09",
            "nav": 101000.0,
            "cash": 98998.0,
            "positions_value": 2002.0,
            "bench_nav": 100500.0,
        },
    ]
    assert orders.to_dict("records") == [{
        "date": "2026-07-09",
        "ticker": "000001",
        "side": "buy",
        "shares": 100.0,
        "price": 10.0,
        "value": 1000.0,
        "cost": 2.0,
    }]
    assert state["account"] == DEFAULT_ACCOUNT
    assert state["cash"] == 98998.0
    assert state["positions"] == {"000001": 100.0}
    assert state["last_run"] == "2026-07-09"


def test_publish_is_idempotent_for_existing_account(tmp_path):
    input_dir = tmp_path / "output" / "intraday_6m"
    paper_dir = tmp_path / "paper"
    _write_input(input_dir)

    first = publish(input_dir=input_dir, paper_dir=paper_dir)
    first_nav = (first / "nav.csv").read_text(encoding="utf-8")
    first_orders = (first / "orders.csv").read_text(encoding="utf-8")
    first_state = (first / "state.json").read_text(encoding="utf-8")

    second = publish(input_dir=input_dir, paper_dir=paper_dir)

    assert second == first
    assert (second / "nav.csv").read_text(encoding="utf-8") == first_nav
    assert (second / "orders.csv").read_text(encoding="utf-8") == first_orders
    assert (second / "state.json").read_text(encoding="utf-8") == first_state
    assert not list((paper_dir / DEFAULT_ACCOUNT).glob("*.tmp"))
