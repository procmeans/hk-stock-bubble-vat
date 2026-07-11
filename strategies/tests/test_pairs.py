def _pair_prices(n=60, diverge_at=40):
    # a、b 高度相关(带确定性小噪声);diverge_at 后 a 跳涨脱离 b
    a, b = [], []
    for i in range(n):
        base = 100 + (i % 5)
        a.append(base * (1.25 if i >= diverge_at else 1.0))
        b.append(base + 0.5 * (-1) ** i)
    return a, b


def test_shorts_spread_after_divergence(make_panel):
    a, b = _pair_prices()
    noise = [100.0 + (i % 7) for i in range(len(a))]   # 第三只:相关性低
    panel = make_panel({"a": a, "b": b, "noise": noise})

    from strategies.pairs import signal, top_pairs
    assert top_pairs(panel["close"], train=40, n_pairs=1) == [("a", "b")]

    w = signal(panel, n_pairs=1, train=40, window=10, entry=2.0, exit_=0.5)

    assert (w.iloc[:40] == 0).all().all()       # 训练窗内不交易
    later = w.iloc[45]
    assert later["a"] < 0 and later["b"] > 0    # 价差过高:空 a 多 b
    assert abs(later["a"]) + abs(later["b"]) <= 1.0 + 1e-9
