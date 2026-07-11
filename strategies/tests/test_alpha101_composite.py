def _stub_alpha101(monkeypatch, panel):
    score = panel["close"].rank(axis=1, pct=True)      # 收盘价高者组合分高
    monkeypatch.setattr("alpha101.alphas.compute_all", lambda p: {"f": score})
    monkeypatch.setattr("alpha101.universe.liquidity_mask", lambda p: score.notna())
    monkeypatch.setattr("alpha101.compose.composite",
                        lambda factors, mask=None: factors["f"])


def test_signal_picks_top_composite_on_rebalance_grid(make_panel, monkeypatch):
    panel = make_panel({"hi": [200.0] * 12, "lo": [100.0] * 12})
    _stub_alpha101(monkeypatch, panel)
    from strategies.alpha101_composite import signal

    w = signal(panel, top_n=1, rebalance=5)

    assert w["hi"].iloc[-1] == 1.0 and w["lo"].iloc[-1] == 0.0
    assert (w.iloc[0] == w.iloc[4]).all()              # 调仓间隔内不变


def test_targets_latest_cross_section(make_panel, monkeypatch):
    panel = make_panel({"hi": [200.0] * 6, "lo": [100.0] * 6})
    _stub_alpha101(monkeypatch, panel)
    from strategies.alpha101_composite import targets

    assert targets(panel, top_n=1) == {"hi": 1.0}
