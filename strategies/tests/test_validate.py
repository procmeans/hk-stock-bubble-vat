import numpy as np
import pandas as pd
import pytest

from strategies import validate


def test_benchmark_is_cross_section_mean(make_panel):
    panel = make_panel({"a": [100, 110], "b": [100, 90]})
    bench = validate.benchmark_returns(panel)
    assert bench.iloc[1] == pytest.approx((0.10 + (-0.10)) / 2)


def test_yearly_table_splits_calendar_years():
    idx = pd.bdate_range("2024-12-30", periods=4)   # 两天 2024,两天 2025
    net = pd.Series([0.01, 0.02, 0.0, 0.01], index=idx)
    bench = pd.Series(0.0, index=idx)
    table = validate.yearly_table(net, bench)
    assert table.loc[2024, "strategy"] == pytest.approx(1.01 * 1.02 - 1)
    assert table.loc[2025, "excess"] == pytest.approx(1.01 - 1)


def test_validate_one_excludes_warmup(make_panel, monkeypatch):
    n = 30
    panel = make_panel({
        "w": [100 * 1.01 ** i for i in range(n)],
        "f": [100 + 0.5 * (-1) ** i for i in range(n)],
    })
    idx = panel["close"].index
    stub = pd.DataFrame(0.0, index=idx, columns=["w", "f"])
    stub.iloc[5:, 0] = 1.0                    # 第 5 日发信号,第 6 日起持仓
    monkeypatch.setitem(validate.REGISTRY, "stub", lambda panel: stub)

    r = validate.validate_one("stub", panel, "us", cost_bps=0.0)

    assert r["live_start"] == idx[6]          # 预热期剔除
    net = panel["close"]["w"].pct_change().loc[idx[6]:]
    bench = validate.benchmark_returns(panel).loc[idx[6]:]
    excess = net - bench
    expected_t = excess.mean() / excess.std(ddof=0) * np.sqrt(len(excess))
    assert r["t_stat"] == pytest.approx(expected_t)
    assert r["verdict"] in {"显著跑赢基准", "超额不显著"}


def test_validate_one_never_live(make_panel, monkeypatch):
    panel = make_panel({"a": [100.0] * 10, "b": [100.0] * 10})
    idx = panel["close"].index
    zero = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    monkeypatch.setitem(validate.REGISTRY, "zero", lambda panel: zero)

    r = validate.validate_one("zero", panel, "us")

    assert r["verdict"] == "从未建仓"
