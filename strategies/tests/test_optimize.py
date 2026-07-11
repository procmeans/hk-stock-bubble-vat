import pandas as pd
import pytest

from strategies import optimize


def test_combos_respect_constraint(monkeypatch):
    monkeypatch.setitem(optimize.GRIDS, "ma_cross",
                        {"fast": [20, 50, 100], "slow": [60, 120, 200]})
    result = optimize.combos("ma_cross")
    assert len(result) == 8                       # (100,60) 被 fast<slow 滤掉
    assert all(p["fast"] < p["slow"] for p in result)


def test_split_dates_no_overlap():
    idx = pd.bdate_range("2024-01-01", periods=10)
    train, holdout = optimize.split_dates(idx, ratio=0.6)
    assert len(train) == 6 and len(holdout) == 4
    assert train.intersection(holdout).empty
    assert train.union(holdout).equals(idx)


def test_grid_search_picks_active_combo(make_panel, monkeypatch):
    n = 30
    panel = make_panel({"up": [100 * 1.02 ** i for i in range(n)],
                        "dn": [100 * 0.99 ** i for i in range(n)]})
    monkeypatch.setitem(optimize.GRIDS, "ma_cross", {"fast": [2], "slow": [4, 25]})
    r = optimize.grid_search("ma_cross", panel, ratio=0.6)
    assert r["best"] == {"fast": 2, "slow": 4}    # slow=25 在训练段几乎无信号
    assert r["train_sharpe"] > 0


def test_overfit_flag_on_reversal(make_panel, monkeypatch):
    # 训练段动量有效,留出段反转 -> 留出夏普为负 -> 红牌
    n = 50
    up_then_down = [100 * 1.03 ** i for i in range(30)] + \
                   [100 * 1.03 ** 30 * 0.95 ** i for i in range(1, 21)]
    panel = make_panel({"a": up_then_down,
                        "b": [100 + (i % 2) for i in range(n)],
                        "c": [100 - 0.1 * i for i in range(n)]})
    monkeypatch.setitem(optimize.GRIDS, "momentum",
                        {"top_n": [1], "lookback": [10], "skip": [2],
                         "rebalance": [5]})
    r = optimize.grid_search("momentum", panel, ratio=0.6)
    assert r["overfit_flag"] is True
