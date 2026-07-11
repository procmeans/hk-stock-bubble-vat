"""alpha101 全因子组合适配层:滚动 IC 加权合成 -> 调仓日 top N 等权。

面板无 ind 时行业中性化因子按 alpha101 既有行为跳过。计算较重,
不进 optimize.GRIDS;paper 只在调仓日调用 targets。
"""
import numpy as np


def _score(panel):
    from alpha101 import alphas, compose, universe
    factors = alphas.compute_all(panel)
    mask = universe.liquidity_mask(panel)
    return compose.composite(factors, mask=mask)


def signal(panel, top_n=50, rebalance=5):
    score = _score(panel)
    keep = np.zeros(len(score), dtype=bool)
    keep[::rebalance] = True
    score[~keep] = np.nan
    top = score.rank(axis=1, ascending=False) <= top_n
    counts = top.sum(axis=1)
    weights = top.div(counts.replace(0, np.nan), axis=0)
    weights[~keep] = np.nan
    return weights.ffill().fillna(0.0)


def targets(panel, top_n=50, **_):
    """paper 调仓日用:最新截面 top N 等权。"""
    latest = _score(panel).iloc[-1].dropna()
    top = latest.nlargest(top_n)
    if top.empty:
        return {}
    return {code: 1.0 / len(top) for code in top.index}
