"""均值回归:z-score 低于 entry 买入,回到 exit_ 以上离场,持仓等权。"""
import numpy as np
import pandas as pd


def signal(panel, window=20, entry=-2.0, exit_=0.0):
    close = panel["close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std
    state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    state[z >= exit_] = 0.0
    state[z < entry] = 1.0
    hold = state.ffill().fillna(0.0) > 0
    counts = hold.sum(axis=1)
    return hold.div(counts.replace(0, np.nan), axis=0).fillna(0.0)
