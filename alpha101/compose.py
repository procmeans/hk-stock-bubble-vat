"""因子合成:去极值 -> z-score -> 等权 / IC 加权。"""
import numpy as np
import pandas as pd

from alpha101 import backtest


def winsorize_zscore(factor, n=3):
    med = factor.median(axis=1)
    mad = (factor.sub(med, axis=0)).abs().median(axis=1)
    lo = med - n * 1.4826 * mad
    hi = med + n * 1.4826 * mad
    clipped = factor.clip(lower=lo, upper=hi, axis=0)
    mean = clipped.mean(axis=1)
    std = clipped.std(axis=1, ddof=0).replace(0, np.nan)
    return clipped.sub(mean, axis=0).div(std, axis=0)


def composite(factors, mask=None):
    total = None
    for f in factors.values():
        z = winsorize_zscore(f)
        total = z if total is None else total.add(z, fill_value=0)
    if mask is not None:
        total = total.where(mask.reindex_like(total).fillna(False))
    return total


def rolling_ic_weights(factors, fwd, mask=None, window=120, lag=15, min_periods=None):
    """滚动 IC 定权(含符号,自动方向对齐;严格只用过去信息)。
    每因子每日 IC = spearman(因子, 未来收益);IC 在 t 日算出但要到 t+前瞻期才实现,
    故按 lag(≥前瞻期)滞后后再滚动 window 求均值,作为权重。返回 DataFrame[日期×因子]。"""
    ics = {}
    for k, f in factors.items():
        ff = f.where(mask.reindex_like(f).fillna(False)) if mask is not None else f
        ics[k] = backtest.ic_series(ff, fwd)
    icdf = pd.DataFrame(ics).reindex(fwd.index).sort_index()
    if min_periods is None:
        min_periods = max(20, window // 2)
    return icdf.shift(lag).rolling(window, min_periods=min_periods).mean()


def ic_weighted_composite(factors, weights, mask=None):
    """按 weights[日期×因子] 对各因子 z-score 加权求和。
    权重含符号:负 IC 因子权重为负,自动翻向;弱因子权重小,强因子权重大。"""
    total = None
    for k, f in factors.items():
        if k not in weights.columns:
            continue
        contrib = winsorize_zscore(f).mul(weights[k].reindex(f.index), axis=0)
        total = contrib if total is None else total.add(contrib, fill_value=0)
    if mask is not None:
        total = total.where(mask.reindex_like(total).fillna(False))
    return total
