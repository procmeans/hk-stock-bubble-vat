import numpy as np
import pandas as pd
from alpha101 import universe


def _panel():
    idx = pd.date_range("2020-01-01", periods=5)
    cols = ["A", "B"]
    vol = pd.DataFrame([[100, 100], [100, 0], [100, 100], [100, 100], [100, 100]],
                       index=idx, columns=cols, dtype=float)
    amt = pd.DataFrame(1e8, index=idx, columns=cols)
    amt["B"] = 1.0  # B 流动性极低
    return {"volume": vol, "amount": amt}


def test_suspended_day_is_false():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=0, adv_window=1, drop_pct=0.0)
    assert m.loc["2020-01-02", "B"] == False  # volume==0 当日停牌


def test_new_listing_masked():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=3, adv_window=1, drop_pct=0.0)
    assert m.iloc[:3]["A"].any() == False  # 前3行次新


def test_low_liquidity_dropped():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=0, adv_window=1, drop_pct=0.5)
    assert m.loc["2020-01-03", "B"] == False  # B 均额在后 50%
