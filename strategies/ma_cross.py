"""双均线:快线上穿慢线持有、下穿离场,信号股等权。"""
import numpy as np


def signal(panel, fast=20, slow=60):
    close = panel["close"]
    hold = close.rolling(fast).mean() > close.rolling(slow).mean()
    counts = hold.sum(axis=1)
    return hold.div(counts.replace(0, np.nan), axis=0).fillna(0.0)
