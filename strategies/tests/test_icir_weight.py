def test_icir_weighted_composite_picks_winner(make_panel):
    n = 60
    panel = make_panel({
        "loser": [100 * 0.99 ** i for i in range(n)],
        "winner": [100 * 1.02 ** i for i in range(n)],
        "flat": [100 + (i % 3) for i in range(n)],
        "flat2": [100 + ((i + 1) % 3) for i in range(n)],
    })

    from strategies.icir_weight import signal
    w = signal(panel, top_n=1, train=30, retrain=5, horizon=3,
               feat_windows=(3, 5, 8))

    assert (w.iloc[:30].sum(axis=1) == 0).all()   # 训练期空仓
    assert w["winner"].iloc[-1] == 1.0            # 动量因子 IC 为正,选中强势股
    assert w["loser"].iloc[-1] == 0.0
