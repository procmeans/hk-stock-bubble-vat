import json

import pandas as pd
import pytest

from strategies import paper

SMALL = {"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5}


def _close(n=20, up=True):
    idx = pd.bdate_range("2024-01-01", periods=n)
    a = [100 * (1.02 if up else 1.0) ** i for i in range(n)]
    b = [100 + 0.1 * (i % 2) for i in range(n)]
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _state(cash=100000.0):
    return {"account": "t", "capital": cash, "cash": cash, "positions": {},
            "pending_targets": None, "days_since_rebalance": None,
            "bench_nav": cash, "last_run": None}


def test_first_step_creates_pending_without_trading():
    state, nav_row, orders = paper.step(_state(), {"close": _close()}, params=SMALL)
    assert orders == []
    assert state["pending_targets"] == {"A": 1.0}      # 动量最强者
    assert nav_row["nav"] == pytest.approx(100000.0)
    assert nav_row["cash"] == pytest.approx(100000.0)


def test_pending_fills_next_day_with_cost():
    close = _close()
    state, _, _ = paper.step(_state(), {"close": close.iloc[:-1]}, params=SMALL)
    state, nav_row, orders = paper.step(state, {"close": close}, params=SMALL)
    assert len(orders) == 1 and orders[0]["ticker"] == "A"
    price = close["A"].iloc[-1]
    assert orders[0]["price"] == pytest.approx(round(float(price), 4))
    fee = 100000.0 * 20 / 1e4 * 1.0                    # 全仓买入的单边成本
    assert nav_row["nav"] == pytest.approx(100000.0 - fee, rel=1e-6)
    assert nav_row["nav"] == pytest.approx(
        nav_row["cash"] + nav_row["positions_value"], rel=1e-9)  # 会计恒等


def test_idempotent_same_day():
    close = _close()
    state, _, _ = paper.step(_state(), {"close": close}, params=SMALL)
    before = json.dumps(state, sort_keys=True)
    state2, nav_row, orders = paper.step(state, {"close": close}, params=SMALL)
    assert nav_row is None and orders == []
    assert json.dumps(state2, sort_keys=True) == before


def test_rebalance_cadence():
    n = 22
    close = _close(n)
    state = _state()
    pending_days = []
    for k in range(12, n + 1):                          # 逐日步进
        state, _, _ = paper.step(state, {"close": close.iloc[:k]}, params=SMALL)
        pending_days.append(state["pending_targets"] is not None)
    # 首日出信号,次日成交后归 None,直到第 rebalance 个交易日再次出现
    assert pending_days[0] is True
    assert pending_days[1] is False
    assert any(pending_days[2:])                        # 计数到 5 再调仓


def test_target_weights_equal_weight_top_n():
    close = _close(15)
    w = paper.target_weights(close, top_n=2, lookback=10, skip=2)
    assert w == {"A": 0.5, "B": 0.5}


def test_init_creates_files_and_refuses_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 50000.0)
    state = paper.load_state("acct")
    assert state["cash"] == 50000.0
    assert (tmp_path / "acct" / "nav.csv").exists()
    with pytest.raises(SystemExit):
        paper.init("acct", 50000.0)


def test_run_appends_nav_and_saves_state(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 100000.0)
    close = _close()
    volume = close * 0 + 1000.0
    monkeypatch.setattr(paper, "universe_tickers", lambda: ["A", "B"])
    paper.run("acct", fetch=lambda codes, window_days=0: (close, volume))
    nav = pd.read_csv(tmp_path / "acct" / "nav.csv")
    assert len(nav) == 1
    state = paper.load_state("acct")
    assert state["last_run"] == close.index[-1].strftime("%Y-%m-%d")


def test_step_dispatches_alpha101_targets(monkeypatch):
    import strategies.alpha101_composite as ac
    seen = {}

    def fake_targets(panel, **kw):
        seen["kw"] = kw
        return {"A": 1.0}

    monkeypatch.setattr(ac, "targets", fake_targets)
    state = _state()
    state["strategy"] = "alpha101"
    state["params"] = {"top_n": 1, "rebalance": 5}

    state, nav_row, orders = paper.step(state, {"close": _close()})

    assert state["pending_targets"] == {"A": 1.0}
    assert seen["kw"] == {"top_n": 1, "rebalance": 5}


def test_init_registers_account_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("a_alpha101", 100000.0, strategy="alpha101", market="a",
               title="A股 Alpha101")
    entries = json.loads((tmp_path / "accounts.json").read_text())
    assert entries[0]["account"] == "a_alpha101"
    assert entries[0]["currency"] == "¥"
    state = paper.load_state("a_alpha101")
    assert state["strategy"] == "alpha101" and state["market"] == "a"
    assert state["params"] == paper.A101_PARAMS


def test_run_a_market_uses_amount_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct2", 100000.0, strategy="momentum", market="a",
               params={"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5})
    close = _close()
    panel = {"close": close, "volume": close * 0 + 1000.0,
             "amount": close * 1000.0, "returns": close.pct_change()}
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run("acct2", fetch=lambda codes, window_days=0: panel)
    assert paper.load_state("acct2")["last_run"] == \
        close.index[-1].strftime("%Y-%m-%d")
