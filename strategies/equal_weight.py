"""流动性等权:60 日 ADV 前 top N 等权,定期再平衡。

A 股近 4 年的"事实冠军"风格(小微盘等权 + 再平衡收割反转)。
需要 panel["amount"];已知风险:小微盘拥挤、雪崩式回撤、幸存者偏差。
"""
import numpy as np


def signal(panel, top_n=500, rebalance=5):
    adv = panel["amount"].rolling(60, min_periods=1).mean()
    keep = np.zeros(len(adv), dtype=bool)
    keep[::rebalance] = True
    adv[~keep] = np.nan
    top = adv.rank(axis=1, ascending=False) <= top_n
    counts = top.sum(axis=1)
    weights = top.div(counts.replace(0, np.nan), axis=0)
    weights[~keep] = np.nan
    return weights.ffill().fillna(0.0)


def targets(panel, top_n=500, **_):
    """paper 调仓日用:最新 60 日 ADV 前 top N 等权。"""
    adv = panel["amount"].tail(60).mean().dropna().nlargest(top_n)
    if adv.empty:
        return {}
    return {code: 1.0 / len(adv) for code in adv.index}
