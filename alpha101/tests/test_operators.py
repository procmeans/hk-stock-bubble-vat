import numpy as np
import pandas as pd
import pytest
from alpha101 import operators as op


def _df(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx, columns=["A", "B", "C"])


def test_rank_is_cross_sectional_pct():
    df = _df([[1.0, 2.0, 3.0]])
    assert df.pipe(op.rank).iloc[0].tolist() == pytest.approx([1/3, 2/3, 1.0])


def test_scale_sum_abs_equals_a():
    df = _df([[1.0, -3.0, 0.0]])
    assert op.scale(df, a=1.0).iloc[0].abs().sum() == pytest.approx(1.0)


def test_signedpower_keeps_sign():
    df = _df([[-4.0, 9.0, 2.0]])
    out = op.signedpower(df, 0.5).iloc[0]
    assert out.tolist() == pytest.approx([-2.0, 3.0, np.sqrt(2)])


def test_ew_max_elementwise():
    x = _df([[1.0, 5.0, 3.0]])
    y = _df([[4.0, 2.0, 3.0]])
    assert op.ew_max(x, y).iloc[0].tolist() == [4.0, 5.0, 3.0]
