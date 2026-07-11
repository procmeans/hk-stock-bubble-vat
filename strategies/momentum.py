"""横截面动量:12-1 月收益 top N 等权,固定间隔调仓。"""
import numpy as np


def score(close, lookback=252, skip=21):
    """跳过最近 skip 日的 lookback 区间收益(避开短期反转)。"""
    return close.shift(skip) / close.shift(lookback) - 1.0


def signal(panel, top_n=20, lookback=252, skip=21, rebalance=21):
    close = panel["close"]
    mom = score(close, lookback, skip)
    keep = np.zeros(len(close), dtype=bool)
    keep[::rebalance] = True
    mom[~keep] = np.nan                     # 非调仓日不出新信号
    top = mom.rank(axis=1, ascending=False) <= top_n
    counts = top.sum(axis=1)
    w = top.div(counts.replace(0, np.nan), axis=0)
    w[~keep] = np.nan
    return w.ffill().fillna(0.0)
