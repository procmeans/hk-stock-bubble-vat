import pytest


def test_long_short_neutral(make_panel):
    n = 20
    panel = make_panel({
        "strong": [100 * 1.05 ** i for i in range(n)],
        "weak": [100 * 0.95 ** i for i in range(n)],
        "flat": [100.0] * n,
        "flat2": [100.0 + 0.01 * i for i in range(n)],
    })

    from strategies.market_neutral import signal
    w = signal(panel, top_n=1, lookback=10, skip=2, rebalance=5)

    last = w.iloc[-1]
    assert last["strong"] == pytest.approx(0.5)     # 多最强
    assert last["weak"] == pytest.approx(-0.5)      # 空最弱
    assert last.sum() == pytest.approx(0.0)         # 净敞口 0
    assert last.abs().sum() == pytest.approx(1.0)   # 总敞口 1
