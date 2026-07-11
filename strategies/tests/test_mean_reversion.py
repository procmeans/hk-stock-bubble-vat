def test_enters_on_dip_exits_on_recovery(make_panel):
    # 10 天横盘 -> 暴跌(z<entry 入场)-> 收复(z>=0 出场)。
    # 注意:窗口 n 内单日下跌的 z 下限为 -(n-1)/sqrt(n)(n=5 时 ≈ -1.79),
    # 故测试用 entry=-1.5;默认 window=20 时 -2 阈值可达(下限 ≈ -4.25)。
    prices = [100.0] * 10 + [90.0, 90.0, 101.0, 101.0]
    panel = make_panel({"a": prices, "flat": [100.0] * len(prices)})

    from strategies.mean_reversion import signal
    w = signal(panel, window=5, entry=-1.5, exit_=0.0)

    assert w["a"].iloc[10] == 1.0    # 暴跌日入场
    assert w["a"].iloc[11] == 1.0    # 未回归前继续持有
    assert w["a"].iloc[-1] == 0.0    # 收复后离场
    assert (w["flat"] == 0).all()    # 无偏离不交易
