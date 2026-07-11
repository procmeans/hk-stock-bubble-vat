def test_picks_strongest_and_rebalances(make_panel):
    n = 20
    strong = [100 * 1.05 ** i for i in range(n)]
    weak = [100 * 0.98 ** i for i in range(n)]
    flat = [100.0] * n
    panel = make_panel({"strong": strong, "weak": weak, "flat": flat})

    from strategies.momentum import signal
    w = signal(panel, top_n=1, lookback=10, skip=2, rebalance=5)

    assert w["strong"].iloc[-1] == 1.0
    assert w["weak"].iloc[-1] == 0.0
    assert (w.iloc[:10].sum(axis=1) == 0).all()       # lookback 未满前空仓
    assert (w.iloc[10] == w.iloc[12]).all().all()     # 调仓间隔内权重不变
