"""因子回测评估:IC、分层多空、换手。"""
import numpy as np
import pandas as pd


def forward_return(close):
    return close.shift(-1) / close - 1.0


def ic_series(factor, fwd, method="spearman"):
    f = factor.reindex_like(fwd)
    out = {}
    for day in f.index:
        a, b = f.loc[day], fwd.loc[day]
        m = a.notna() & b.notna()
        if m.sum() >= 5:
            out[day] = a[m].corr(b[m], method=method)
    return pd.Series(out).sort_index()


def ic_stats(ic):
    ic = ic.dropna()
    mean, std = ic.mean(), ic.std()
    return {"ic_mean": mean, "ic_std": std,
            "icir": mean / std if std else np.nan}


def quantile_returns(factor, fwd, q=5):
    f = factor.reindex_like(fwd)
    rows = {}
    for day in f.index:
        a, b = f.loc[day], fwd.loc[day]
        m = a.notna() & b.notna()
        if m.sum() < q:
            continue
        labels = pd.qcut(a[m].rank(method="first"), q, labels=False)
        rows[day] = b[m].groupby(labels).mean()
    return pd.DataFrame(rows).T.sort_index()


def long_short_stats(qret, q=5):
    ls = qret[q - 1] - qret[0]
    ls = ls.dropna()
    ann = ls.mean() * 252
    sharpe = (ls.mean() / ls.std() * np.sqrt(252)) if ls.std() else np.nan
    return {"ls_annual": ann, "ls_sharpe": sharpe}


def turnover(factor, q=5):
    top_sets = []
    for day in factor.index:
        a = factor.loc[day].dropna()
        if len(a) < q:
            top_sets.append(set())
            continue
        n = max(1, len(a) // q)
        top_sets.append(set(a.nlargest(n).index))
    tos = []
    for i in range(1, len(top_sets)):
        prev, cur = top_sets[i - 1], top_sets[i]
        if cur:
            tos.append(len(cur - prev) / len(cur))
    return float(np.mean(tos)) if tos else np.nan


def evaluate(factor, close, q=5):
    fwd = forward_return(close)
    ic = ic_series(factor, fwd)
    qret = quantile_returns(factor, fwd, q=q)
    return {**ic_stats(ic), **long_short_stats(qret, q=q),
            "turnover": turnover(factor, q=q), "ic": ic, "qret": qret}
