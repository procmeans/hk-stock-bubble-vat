import pytest


def test_signal_equal_weights_top_adv(make_panel):
    n = 12
    panel = make_panel({"big": [100.0] * n, "mid": [50.0] * n, "small": [10.0] * n})
    # make_panel 的 amount = close*volume,big 的成交额最大

    from strategies.equal_weight import signal
    w = signal(panel, top_n=2, rebalance=5)

    last = w.iloc[-1]
    assert last["big"] == pytest.approx(0.5)
    assert last["mid"] == pytest.approx(0.5)
    assert last["small"] == 0.0
    assert (w.iloc[0] == w.iloc[4]).all()      # 调仓间隔内不变


def test_targets_latest_adv(make_panel):
    panel = make_panel({"big": [100.0] * 6, "small": [10.0] * 6})

    from strategies.equal_weight import targets

    assert targets(panel, top_n=1) == {"big": 1.0}
