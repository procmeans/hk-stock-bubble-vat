"""市场中性:动量 top N 等权做多、bottom N 等权做空,净敞口 0、总敞口 1。

A 股无法融券做空全名单,回测结果视为纸面模拟(run.py 会标注)。
"""
import numpy as np

from strategies.momentum import score


def signal(panel, top_n=20, lookback=252, skip=21, rebalance=21):
    close = panel["close"]
    mom = score(close, lookback, skip)
    keep = np.zeros(len(close), dtype=bool)
    keep[::rebalance] = True
    mom[~keep] = np.nan                     # 非调仓日不出新信号
    long_ = mom.rank(axis=1, ascending=False) <= top_n
    short = (mom.rank(axis=1, ascending=True) <= top_n) & ~long_
    w = (
        long_.div((long_.sum(axis=1) * 2).replace(0, np.nan), axis=0)
        - short.div((short.sum(axis=1) * 2).replace(0, np.nan), axis=0)
    )
    w[~keep] = np.nan
    return w.ffill().fillna(0.0)
