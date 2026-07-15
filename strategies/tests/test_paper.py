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


def test_missing_execution_close_drops_order_but_values_existing_position():
    index = pd.bdate_range("2026-07-14", periods=2)
    close = pd.DataFrame({"A": [10.0, float("nan")]}, index=index)
    state = _state(900.0)
    state["positions"] = {"A": 10.0}
    state["pending_targets"] = {}
    state["days_since_rebalance"] = 0

    state, nav_row, orders = paper.step(state, {"close": close}, params=SMALL)

    assert orders == []
    assert state["positions"] == {"A": 10.0}
    assert state["pending_targets"] is None
    assert nav_row["positions_value"] == 100.0
    assert nav_row["nav"] == 1000.0


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


def test_save_state_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 50000.0)
    path = tmp_path / "acct" / "state.json"
    original = path.read_text()
    state = paper.load_state("acct")
    state["cash"] = 123.0

    def fail_replace(source, destination):
        assert source.parent == destination.parent
        raise OSError("injected replace failure")

    monkeypatch.setattr(paper.Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        paper.save_state("acct", state)

    assert path.read_text() == original
    assert list(path.parent.glob(".state.json.*.tmp")) == []


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


def test_step_dispatches_equal_weight(monkeypatch):
    import strategies.equal_weight as ew
    monkeypatch.setattr(ew, "targets", lambda panel, **kw: {"B": 1.0})
    state = _state()
    state["strategy"] = "equal_weight"
    state["params"] = {"top_n": 1, "rebalance": 5}

    state, _, _ = paper.step(state, {"close": _close()})

    assert state["pending_targets"] == {"B": 1.0}


def test_run_market_shares_one_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    small = {"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5}
    paper.init("a_one", 100000.0, strategy="momentum", market="a", params=small)
    paper.init("a_two", 100000.0, strategy="momentum", market="a", params=small)
    paper.init("us_x", 100000.0, strategy="momentum", market="us")
    close = _close()
    panel = {"close": close, "volume": close * 0 + 1000.0,
             "amount": close * 1000.0, "returns": close.pct_change()}
    calls = []

    def fake_fetch(codes, window_days=0):
        calls.append(list(codes))
        return panel

    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run_market("a", fetch=fake_fetch)

    assert len(calls) == 1                     # 一次抓数,共用面板
    today = close.index[-1].strftime("%Y-%m-%d")
    assert paper.load_state("a_one")["last_run"] == today
    assert paper.load_state("a_two")["last_run"] == today
    assert paper.load_state("us_x")["last_run"] is None   # 非本市场不动


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


@pytest.mark.parametrize("strategy", ["ths_heat", "ths_heat_rise"])
def test_step_uses_prefetched_heat_targets(strategy):
    state = _state()
    state["strategy"] = strategy
    state["params"] = {"top_n": 20, "rebalance": 2}
    state, _, _ = paper.step(
        state, {"close": _close()}, target_override={"B": 1.0}
    )
    assert state["pending_targets"] == {"B": 1.0}


def test_heat_rebalance_two_holds_for_two_closes():
    close = _close(15)
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 1, "rebalance": 2}
    state, _, _ = paper.step(
        state, {"close": close.iloc[:12]}, target_override={"A": 1.0}
    )
    state, _, _ = paper.step(state, {"close": close.iloc[:13]})
    assert state["pending_targets"] is None and "A" in state["positions"]
    state, _, _ = paper.step(
        state, {"close": close.iloc[:14]}, target_override={"B": 1.0}
    )
    assert state["pending_targets"] == {"B": 1.0}
    state, _, _ = paper.step(state, {"close": close.iloc[:15]})
    assert "B" in state["positions"] and "A" not in state["positions"]


def test_benchmark_close_excludes_supplemental_hot_stock():
    idx = pd.bdate_range("2026-07-14", periods=2)
    close = pd.DataFrame({"A": [10.0, 10.0], "HOT": [10.0, 20.0]}, index=idx)
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 1, "rebalance": 2}
    state, nav_row, _ = paper.step(
        state,
        {"close": close, "benchmark_close": close[["A"]]},
        target_override={},
    )
    assert nav_row["bench_nav"] == 100000.0


def test_market_panel_ranks_benchmark_from_snapshot_candidates_only(monkeypatch):
    monkeypatch.setattr(paper, "UNIVERSE_SIZE", 2)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    panel = _a_panel(("A", "B", "HOT"))
    panel["amount"]["HOT"] = 1_000_000_000.0

    result = paper._market_panel(
        "a", {"HOT"}, fetch=lambda codes, window_days=0: panel
    )

    assert set(result["benchmark_close"].columns) == {"A", "B"}
    assert "HOT" in result["close"].columns


def test_failed_due_heat_signal_stays_due_for_next_day():
    state = _state()
    state["strategy"] = "ths_heat"
    state["params"] = {"top_n": 20, "rebalance": 2}
    state["days_since_rebalance"] = 2
    state, _, _ = paper.step(state, {"close": _close()}, target_override={})
    assert state["pending_targets"] is None
    assert paper.rebalance_due(state) is True


def _a_panel(columns=("A", "B"), n=20):
    base = _close(n)
    close = pd.DataFrame(index=base.index)
    for column in columns:
        close[column] = base[column] if column in base else 50.0
    return {
        "close": close, "volume": close * 0 + 1000.0,
        "amount": close * 1000.0, "returns": close.pct_change(),
    }


def _signal(strategy, ticker, value):
    return pd.DataFrame([{
        "date": "2024-01-26", "strategy": strategy, "rank": 1,
        "ticker": ticker, "name": ticker, "factor_value": value,
    }])


def test_run_market_fetches_due_heat_once_and_supplements_prices(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    paper.init("rise", 100000.0, strategy="ths_heat_rise", market="a")
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    panel_calls, signal_calls = [], []

    def fake_panel(codes, window_days=0):
        panel_calls.append(set(codes))
        return _a_panel(tuple(sorted(codes))) if set(codes) <= {"C", "D"} else _a_panel()

    def fake_signal(day, strategy, top_n=20):
        signal_calls.append((strategy, top_n))
        return _signal(strategy, "C" if strategy == "ths_heat" else "D", 9.0)

    paper.run_market("a", fetch=fake_panel, heat_fetch=fake_signal)
    assert panel_calls == [{"A", "B"}, {"C"}, {"D"}]
    assert {item[0] for item in signal_calls} == {"ths_heat", "ths_heat_rise"}
    assert paper.load_state("heat")["pending_targets"] == {"C": 1.0}
    assert paper.load_state("rise")["pending_targets"] == {"D": 1.0}


def test_prepare_heat_targets_excludes_missing_signal_date_close():
    panel = _a_panel()
    panel["close"].loc[panel["close"].index[-1], "B"] = float("nan")
    state = _state()
    state.update({
        "strategy": "ths_heat",
        "params": {"top_n": 20, "rebalance": 2},
    })
    signal = pd.concat([
        _signal("ths_heat", "A", 9.0),
        _signal("ths_heat", "B", 8.0),
    ], ignore_index=True)

    _, overrides, _ = paper.prepare_heat_targets(
        {"heat": state}, panel,
        lambda *args, **kwargs: pytest.fail("supplemental fetch not expected"),
        fetch_signal=lambda *args, **kwargs: signal,
    )

    assert overrides["heat"] == {"A": 1.0}


def test_supplemental_quote_exception_isolated_by_strategy():
    states = {}
    for account, strategy in [
        ("heat", "ths_heat"), ("rise", "ths_heat_rise")
    ]:
        state = _state()
        state.update({
            "strategy": strategy,
            "params": {"top_n": 20, "rebalance": 2},
        })
        states[account] = state

    def fake_signal(day, strategy, top_n=20):
        ticker = "BAD" if strategy == "ths_heat" else "GOOD"
        return _signal(strategy, ticker, 9.0)

    calls = []

    def fake_panel(codes, window_days=0):
        calls.append(tuple(codes))
        if "BAD" in codes:
            raise RuntimeError("bad supplemental ticker")
        return _a_panel(tuple(codes))

    _, overrides, audit = paper.prepare_heat_targets(
        states, _a_panel(), fake_panel, fetch_signal=fake_signal
    )

    assert calls == [("BAD",), ("GOOD",)]
    assert overrides == {"heat": {}, "rise": {"GOOD": 1.0}}
    errors = [row for row in audit if row["status"] == "error"]
    assert {row["strategy"] for row in errors} == {"ths_heat"}


def test_supplemental_quote_split_keeps_valid_pick_and_audits_missing():
    state = _state()
    state.update({
        "strategy": "ths_heat",
        "params": {"top_n": 20, "rebalance": 2},
    })
    signal = pd.concat([
        _signal("ths_heat", "C", 9.0),
        _signal("ths_heat", "BAD", 8.0),
    ], ignore_index=True)
    calls = []

    def fake_panel(codes, window_days=0):
        calls.append(tuple(codes))
        if "BAD" in codes:
            raise RuntimeError("bad symbol")
        return _a_panel(tuple(codes))

    _, overrides, audit = paper.prepare_heat_targets(
        {"heat": state}, _a_panel(), fake_panel,
        fetch_signal=lambda *args, **kwargs: signal,
    )

    assert calls == [("BAD", "C"), ("BAD",), ("C",)]
    assert overrides["heat"] == {"C": 1.0}
    assert {row["ticker"] for row in audit if row["status"] == "ok"} == {"C"}
    errors = [row for row in audit if row["status"] == "error"]
    assert {row["strategy"] for row in errors} == {"ths_heat"}
    assert any("BAD" in row["error"] for row in errors)


def test_heat_query_failure_isolated_per_account(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    paper.init("rise", 100000.0, strategy="ths_heat_rise", market="a")
    paper.init(
        "plain", 100000.0, strategy="momentum", market="a",
        params={"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5},
    )
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])

    def fake_signal(day, strategy, top_n=20):
        if strategy == "ths_heat":
            raise RuntimeError("temporary API failure")
        return _signal(strategy, "A", 8.0)

    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(),
        heat_fetch=fake_signal,
    )
    assert paper.load_state("heat")["pending_targets"] is None
    assert paper.load_state("rise")["pending_targets"] == {"A": 1.0}
    assert paper.load_state("plain")["last_run"] == "2024-01-26"
    audit = pd.read_csv(tmp_path / "ths_heat_signals.csv")
    assert set(audit["status"]) == {"ok", "error"}


def test_run_market_attempts_later_accounts_after_persist_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("first", 100000.0, strategy="momentum", market="a", params=SMALL)
    paper.init("second", 100000.0, strategy="momentum", market="a", params=SMALL)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    attempted = []

    def fake_step_and_persist(account, state, panel, target_override=None):
        attempted.append(account)
        if account == "first":
            raise OSError("state replace failed")

    monkeypatch.setattr(paper, "_step_and_persist", fake_step_and_persist)

    with pytest.raises(RuntimeError, match="first"):
        paper.run_market(
            "a", fetch=lambda codes, window_days=0: _a_panel()
        )

    assert attempted == ["first", "second"]


def test_retry_after_state_save_failure_upserts_nav_and_orders(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 100000.0, strategy="momentum", market="a", params=SMALL)
    state = paper.load_state("acct")
    state["pending_targets"] = {"000001": 1.0}
    state["days_since_rebalance"] = 0
    paper.save_state("acct", state)
    close = _close().rename(columns={"A": "000001"})

    real_save_state = paper.save_state
    failures = 0

    def fail_first_save(account, next_state):
        nonlocal failures
        if failures == 0:
            failures += 1
            raise OSError("injected state replace failure")
        real_save_state(account, next_state)

    monkeypatch.setattr(paper, "save_state", fail_first_save)
    with pytest.raises(OSError, match="injected"):
        paper._step_and_persist("acct", paper.load_state("acct"), {"close": close})

    paper._step_and_persist("acct", paper.load_state("acct"), {"close": close})

    nav = pd.read_csv(tmp_path / "acct" / "nav.csv")
    orders = pd.read_csv(tmp_path / "acct" / "orders.csv", dtype={"ticker": str})
    assert len(nav) == 1
    assert len(orders) == 1
    assert orders.loc[0, ["ticker", "side"]].tolist() == ["000001", "buy"]


def test_non_due_heat_account_does_not_query(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("heat", 100000.0, strategy="ths_heat", market="a")
    state = paper.load_state("heat")
    state["days_since_rebalance"] = 0
    paper.save_state("heat", state)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run_market(
        "a", fetch=lambda codes, window_days=0: _a_panel(),
        heat_fetch=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("heat endpoint called on non-due day")
        ),
    )


def test_append_heat_signals_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    rows = [
        {
            "date": "2026-07-15", "strategy": "ths_heat", "rank": 1,
            "ticker": "000001", "name": "甲", "factor_value": 100.0,
            "status": "ok", "error": "",
        },
        {
            "date": "2026-07-15", "strategy": "ths_heat", "rank": "",
            "ticker": "", "name": "", "factor_value": "",
            "status": "error", "error": "temporary API failure",
        },
    ]
    paper.append_heat_signals(rows)
    paper.append_heat_signals(rows)
    assert len(pd.read_csv(tmp_path / "ths_heat_signals.csv")) == 2


def test_error_audit_redacts_credentials_and_token_like_strings(
    tmp_path, monkeypatch
):
    configured = "cfg-secret"
    secrets = [
        configured,
        "refresh-field-secret",
        "access-field-secret",
        "authorization-secret",
        "standalone-bearer-secret",
        "ZXCVBNMASDFGHJKLQWERTYUIOP1234567890",
    ]
    monkeypatch.setenv("THS_HTTP_REFRESH_TOKEN", configured)
    error = RuntimeError(
        " first line\n"
        "refresh_token=refresh-field-secret "
        "access_token: access-field-secret "
        "authorization='Bearer authorization-secret' "
        "Bearer standalone-bearer-secret "
        "ZXCVBNMASDFGHJKLQWERTYUIOP1234567890 "
        + "tail " * 100
    )

    row = paper._error_row("2026-07-15", "ths_heat", error)
    paper.append_heat_signals([row], path=tmp_path / "ths_heat_signals.csv")
    persisted = (tmp_path / "ths_heat_signals.csv").read_text()

    assert "\n" not in row["error"]
    assert len(row["error"]) <= 300
    assert "[REDACTED]" in row["error"]
    for secret in secrets:
        assert secret not in row["error"]
        assert secret not in persisted


def test_run_market_aggregate_redacts_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    monkeypatch.setenv("THS_HTTP_REFRESH_TOKEN", "configured-aggregate-secret")
    paper.init("acct", 100000.0, strategy="momentum", market="a", params=SMALL)
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])

    def fail_with_secret(*args, **kwargs):
        raise RuntimeError(
            "authorization: Bearer aggregate-bearer-secret "
            "refresh_token=configured-aggregate-secret"
        )

    monkeypatch.setattr(paper, "_step_and_persist", fail_with_secret)

    with pytest.raises(RuntimeError) as caught:
        paper.run_market(
            "a", fetch=lambda codes, window_days=0: _a_panel()
        )

    summary = str(caught.value)
    assert "acct" in summary and "[REDACTED]" in summary
    assert "aggregate-bearer-secret" not in summary
    assert "configured-aggregate-secret" not in summary


def test_repository_manifest_has_two_ths_heat_accounts():
    root = paper.Path(__file__).resolve().parents[2]
    entries = json.loads((root / "paper" / "accounts.json").read_text())
    by_account = {entry["account"]: entry for entry in entries}
    assert by_account["a_ths_heat"] == {
        "account": "a_ths_heat", "title": "A股 同花顺热度", "currency": "¥"
    }
    assert by_account["a_ths_heat_rise"] == {
        "account": "a_ths_heat_rise",
        "title": "A股 同花顺热度上升", "currency": "¥",
    }
    for account, strategy in [
        ("a_ths_heat", "ths_heat"),
        ("a_ths_heat_rise", "ths_heat_rise"),
    ]:
        state = json.loads((root / "paper" / account / "state.json").read_text())
        assert state["strategy"] == strategy and state["market"] == "a"
        assert state["params"] == {"top_n": 20, "rebalance": 2}


def test_paper_dashboard_formats_capital_and_cost_with_account_currency():
    root = paper.Path(__file__).resolve().parents[2]
    html = (root / "paper.html").read_text()

    assert "'起始 ' + fmt$(capital)" in html
    assert "<td>${fmt$(o.cost)}</td>" in html
    assert "'起始 $100,000'" not in html
    assert "<td>$${(+o.cost).toFixed(2)}</td>" not in html
