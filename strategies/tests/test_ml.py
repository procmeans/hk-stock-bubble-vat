import numpy as np
import pytest


def test_walk_forward_picks_persistent_winner(make_panel):
    n = 60
    panel = make_panel({
        "winner": [100 * 1.02 ** i for i in range(n)],
        "loser": [100 * 0.99 ** i for i in range(n)],
        "flat": [100 + (i % 3) for i in range(n)],
        "flat2": [100 + ((i + 1) % 3) for i in range(n)],
    })

    from strategies.ml import signal
    w = signal(panel, top_n=1, train=30, retrain=5, horizon=3,
               feat_windows=(3, 5, 8))

    assert (w.iloc[:30].sum(axis=1) == 0).all()          # 训练期空仓
    assert w["winner"].iloc[-1] == pytest.approx(1.0)    # 学到持续上涨者
    row_sums = w.abs().sum(axis=1)
    assert ((row_sums == 0) | np.isclose(row_sums, 1.0)).all()
