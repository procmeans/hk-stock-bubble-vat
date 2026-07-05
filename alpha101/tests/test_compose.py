import numpy as np
import pandas as pd
import pytest
from alpha101 import compose


def _f(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=list("ABCD"))


def test_zscore_zero_mean_unit_std():
    z = compose.winsorize_zscore(_f([[1.0, 2, 3, 4]]))
    assert z.iloc[0].mean() == pytest.approx(0, abs=1e-9)
    assert z.iloc[0].std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_composite_equal_weight():
    f1 = _f([[1.0, 2, 3, 4]])
    f2 = _f([[4.0, 3, 2, 1]])
    c = compose.composite({1: f1, 2: f2})
    # 两个相反因子等权相加 -> 截面近似相等
    assert c.iloc[0].std(ddof=0) == pytest.approx(0, abs=1e-9)


def test_pick_top_n():
    score = _f([[1.0, 4, 3, 2]])
    picks = compose_pick_helper(score)
    assert list(picks["code"]) == ["B", "C"]


def test_ic_weighting_flips_negative_factor():
    # 因子A与未来收益同向、因子B反向;滞后滚动后 A 权重应>0、B<0(方向对齐)。
    idx = pd.date_range("2020-01-01", periods=160, freq="B")
    cols = list("ABCDE")
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(size=(160, 5)), index=idx, columns=cols)
    fA = fwd.copy()          # 完全同向 -> 每日IC=+1
    fB = -fwd.copy()         # 完全反向 -> 每日IC=-1
    w = compose.rolling_ic_weights({1: fA, 2: fB}, fwd, window=40, lag=5)
    assert w[1].dropna().iloc[-1] > 0.5
    assert w[2].dropna().iloc[-1] < -0.5


def compose_pick_helper(score):
    from alpha101 import select
    return select.pick(score, score.index[0], top_n=2)
