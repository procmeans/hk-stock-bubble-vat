import pandas as pd


def _scores():
    return pd.DataFrame({
        "date": ["2026-07-08"] * 4,
        "model": ["demo"] * 4,
        "code": ["000001", "000002", "000003", "000004"],
        "name": ["A", "B", "C", "D"],
        "score": [4.0, 3.0, 2.0, 1.0],
        "rank": [1, 2, 3, 4],
        "eligible": [True, True, True, True],
        "reason": [""] * 4,
    })


def test_generate_signals_buys_top_names_equal_weighted():
    from alpha101.trade import generate_signals

    result = generate_signals(_scores(), buy_top_n=2, max_weight=0.2)

    buys = result[result["action"] == "buy"]
    assert buys["code"].tolist() == ["000001", "000002"]
    assert buys["target_weight"].tolist() == [0.2, 0.2]


def test_generate_signals_uses_hold_buffer():
    from alpha101.trade import generate_signals

    result = generate_signals(
        _scores(),
        current_positions={"000003": 0.05},
        buy_top_n=2,
        sell_below_rank=3,
        max_weight=0.2,
    )

    hold = result[result["code"] == "000003"].iloc[0]
    assert hold["action"] == "hold"
    assert hold["target_weight"] == 0.2


def test_generate_signals_sells_names_outside_buffer():
    from alpha101.trade import generate_signals

    result = generate_signals(
        _scores(),
        current_positions={"000004": 0.05},
        buy_top_n=2,
        sell_below_rank=3,
        max_weight=0.2,
    )

    sell = result[result["code"] == "000004"].iloc[0]
    assert sell["action"] == "sell"
    assert sell["target_weight"] == 0.0
    assert sell["reason"] == "rank_below_sell_buffer"
