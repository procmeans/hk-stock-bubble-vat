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


def compose_pick_helper(score):
    from alpha101 import select
    return select.pick(score, score.index[0], top_n=2)
