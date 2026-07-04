import numpy as np
import pandas as pd
import pytest
from alpha101 import backtest as bt


def _factor_and_close():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    cols = list("ABCDE")
    # 让因子完全预测次日收益:因子越大,次日涨越多
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    factor = pd.DataFrame(np.tile(np.arange(5), (6, 1)), index=idx, columns=cols, dtype=float)
    # 构造 fwd return 与 factor 同序
    for i in range(len(idx) - 1):
        close.iloc[i + 1] = close.iloc[i] * (1 + 0.01 * np.arange(5))
    return factor, close


def test_forward_return_is_next_day():
    _, close = _factor_and_close()
    fwd = bt.forward_return(close)
    assert fwd.iloc[-1].isna().all()  # 末行无未来
    assert fwd.iloc[0, 1] == pytest.approx(0.01)


def test_ic_positive_when_aligned():
    factor, close = _factor_and_close()
    fwd = bt.forward_return(close)
    ic = bt.ic_series(factor, fwd)
    assert ic.dropna().mean() == pytest.approx(1.0)  # 完美单调 -> rank IC = 1


def test_quantile_returns_monotonic():
    factor, close = _factor_and_close()
    fwd = bt.forward_return(close)
    qret = bt.quantile_returns(factor, fwd, q=5)
    means = qret.mean()
    assert means.iloc[-1] > means.iloc[0]  # top 层收益 > bottom 层
