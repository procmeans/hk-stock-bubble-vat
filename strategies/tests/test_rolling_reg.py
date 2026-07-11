def _panel(make_panel, n=60):
    return make_panel({
        "loser": [100 * 0.99 ** i for i in range(n)],
        "winner": [100 * 1.02 ** i for i in range(n)],
        "flat": [100 + (i % 3) for i in range(n)],
        "flat2": [100 + ((i + 1) % 3) for i in range(n)],
    })


def test_elastic_net_picks_winner(make_panel):
    from strategies.elastic_net import signal
    w = signal(_panel(make_panel), top_n=1, train=30, retrain=5, horizon=3,
               feat_windows=(3, 5, 8))

    assert (w.iloc[:30].sum(axis=1) == 0).all()   # 训练期空仓
    assert w["winner"].iloc[-1] == 1.0            # 学到持续上涨者
    assert w["loser"].iloc[-1] == 0.0


def test_lasso_picks_winner(make_panel):
    from strategies.lasso import signal
    w = signal(_panel(make_panel), top_n=1, train=30, retrain=5, horizon=3,
               feat_windows=(3, 5, 8), alpha=0.005)

    assert (w.iloc[:30].sum(axis=1) == 0).all()
    assert w["winner"].iloc[-1] == 1.0
    assert w["loser"].iloc[-1] == 0.0
