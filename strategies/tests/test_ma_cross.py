def test_holds_uptrend_not_downtrend(make_panel):
    up = [100 + i for i in range(10)]
    down = [100 - i for i in range(10)]
    panel = make_panel({"up": up, "down": down})

    from strategies.ma_cross import signal
    w = signal(panel, fast=2, slow=4)

    assert w["up"].iloc[-1] == 1.0      # 快线在慢线上方,独占权重
    assert w["down"].iloc[-1] == 0.0
    assert (w.iloc[:3] == 0).all().all()  # 慢线未形成前空仓
