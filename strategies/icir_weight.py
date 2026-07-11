"""ICIR 定权多因子:过去一年各因子日度 rank-IC 的 ICIR 作为权重合成打分。

参照 BigQuant 文档 lSr5ySFNmn(滚动训练、ICIR 最大化)。MVP 用直接
ICIR 定权替代原文的梯度上升最大化(思想一致,与 alpha101/compose.py 同源);
原文的基本面因子以价量因子替代(面板所限)。
"""
import numpy as np
import pandas as pd

from strategies.ml import _features


def _row_corr(a, b):
    """逐日截面相关:corr(a_t, b_t)。"""
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    cov = (am * bm).mean(axis=1)
    return cov / (a.std(axis=1, ddof=0) * b.std(axis=1, ddof=0))


def signal(panel, top_n=30, train=252, retrain=5, horizon=5,
           feat_windows=(21, 63, 126), min_ic_obs=20):
    close = panel["close"]
    feats = _features(panel, feat_windows)
    fwd_rank = (close.shift(-horizon) / close - 1.0).rank(axis=1)
    ics = {
        name: _row_corr(f.rank(axis=1), fwd_rank) for name, f in feats.items()
    }

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in range(train, len(close), retrain):
        score = pd.Series(0.0, index=close.columns)
        weighted = False
        for name, factor in feats.items():
            # IC 序列止于 t - horizon:此后前瞻收益在 t 日尚未实现
            ic = ics[name].iloc[max(0, t - train): t - horizon].dropna()
            if len(ic) < min_ic_obs or ic.std(ddof=0) == 0:
                continue
            icir = ic.mean() / ic.std(ddof=0)
            z = factor.iloc[t]
            z = (z - z.mean()) / z.std()
            score = score + icir * z
            weighted = True
        score = score.dropna()
        if not weighted or score.empty:
            continue
        top = score.nlargest(min(top_n, len(score))).index
        row = pd.Series(0.0, index=close.columns)
        row[top] = 1.0 / len(top)
        weights.iloc[t] = row
    return weights.ffill().fillna(0.0)
