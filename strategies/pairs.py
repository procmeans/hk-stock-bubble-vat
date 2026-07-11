"""配对统计套利:训练窗选高相关对,对数价差 z-score 开平仓。

MVP 用相关性 + 静态对冲比替代协整检验(statsmodels ADF 留作升级)。
"""
import numpy as np
import pandas as pd


def top_pairs(close, train=252, n_pairs=5):
    """训练窗内日收益相关性最高、互不重叠的股票对。"""
    window = close.iloc[:train]
    valid = window.columns[window.notna().all()]
    corr = window[valid].pct_change().corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
    ranked = upper.stack().sort_values(ascending=False)
    result, used = [], set()
    for (left, right), _ in ranked.items():
        if left in used or right in used:
            continue
        result.append((left, right))
        used.update((left, right))
        if len(result) == n_pairs:
            break
    return result


def signal(panel, n_pairs=5, train=252, window=20, entry=2.0, exit_=0.5):
    close = panel["close"]
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for left, right in top_pairs(close, train, n_pairs):
        la, lb = np.log(close[left]), np.log(close[right])
        beta = np.polyfit(lb.iloc[:train], la.iloc[:train], 1)[0]
        spread = la - beta * lb
        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        state = pd.Series(np.nan, index=close.index)
        state[z.abs() < exit_] = 0.0
        state[z > entry] = -1.0    # 价差过高:空 left 多 right
        state[z < -entry] = 1.0    # 价差过低:多 left 空 right
        state.iloc[:train] = 0.0   # 训练窗内不交易
        state = state.ffill().fillna(0.0)
        weights[left] += state * 0.5 / n_pairs
        weights[right] -= state * 0.5 / n_pairs
    return weights
