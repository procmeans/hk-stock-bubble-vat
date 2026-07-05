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


def test_indneutralize_demeans_within_group():
    # A,B 同组 -> 减组均值2;C 单独一组 -> 归零
    df = pd.DataFrame([[1.0, 3.0, 10.0]],
                      index=pd.date_range("2020-01-01", periods=1),
                      columns=["A", "B", "C"])
    groups = pd.Series({"A": "g1", "B": "g1", "C": "g2"})
    out = op.indneutralize(df, groups).iloc[0]
    assert out.tolist() == [-1.0, 1.0, 0.0]


def test_delta():
    df = _df([[1.0, 0, 0], [3.0, 0, 0], [7.0, 0, 0]])[["A"]]
    assert op.delta(df, 1)["A"].tolist()[1:] == [2.0, 4.0]


def test_ts_argmax_position():
    # 窗口内最大值出现在"几天前":最新一天为 d-1,最早为 0
    s = pd.DataFrame({"A": [1.0, 9.0, 2.0, 3.0]},
                     index=pd.date_range("2020-01-01", periods=4))
    out = op.ts_argmax(s, 3)
    # 末窗口 [9,2,3] 最大在第 0 位 -> argmax 索引 0
    assert out["A"].iloc[-1] == 0


def test_ts_rank_last_value_rank():
    s = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]},
                     index=pd.date_range("2020-01-01", periods=4))
    out = op.ts_rank(s, 4)
    # 最后一个值是窗口内最大 -> pct rank = 1.0
    assert out["A"].iloc[-1] == pytest.approx(1.0)


def test_decay_linear_weights():
    s = pd.DataFrame({"A": [1.0, 2.0, 3.0]},
                     index=pd.date_range("2020-01-01", periods=3))
    out = op.decay_linear(s, 3)
    # 权重 3,2,1 归一化 -> (1*1 + 2*2 + 3*3)/6 = 14/6
    assert out["A"].iloc[-1] == pytest.approx(14/6)


def test_correlation_range():
    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=pd.date_range("2020-01-01", periods=5))
    y = pd.DataFrame({"A": [2.0, 4, 6, 8, 10]}, index=pd.date_range("2020-01-01", periods=5))
    assert op.correlation(x, y, 5)["A"].iloc[-1] == pytest.approx(1.0)
