"""因子合成:去极值 -> z-score -> 等权。"""
import numpy as np
import pandas as pd


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
