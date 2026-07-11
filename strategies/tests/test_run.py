import pandas as pd


def test_registry_has_all_strategies():
    from strategies.run import REGISTRY
    assert set(REGISTRY) == {
        "ma_cross", "mean_reversion", "momentum", "market_neutral",
        "pairs", "ml", "elastic_net", "icir_weight", "lasso",
        "alpha101_composite", "equal_weight", "pead",
    }


def test_run_one_writes_equity_and_stats(make_panel, tmp_path, monkeypatch):
    from strategies import run as run_mod
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", tmp_path)
    panel = make_panel({
        "a": [100 + i for i in range(30)],
        "b": [100 - i for i in range(30)],
    })

    stats = run_mod.run_one("ma_cross", panel, "us")

    assert "sharpe" in stats and "max_drawdown" in stats
    saved = pd.read_csv(tmp_path / "us_ma_cross_equity.csv")
    assert "equity" in saved.columns


def test_market_neutral_on_a_share_is_annotated(make_panel, tmp_path, monkeypatch):
    from strategies import run as run_mod
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", tmp_path)
    panel = make_panel({
        "s": [100 * 1.03 ** i for i in range(40)],
        "w": [100 * 0.97 ** i for i in range(40)],
        "f": [100 + (i % 2) for i in range(40)],
    })

    stats = run_mod.run_one("market_neutral", panel, "a")

    assert stats["note"] == "A股做空为纸面模拟"
