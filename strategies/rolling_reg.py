"""滚动回归选股共用骨架(walk-forward,截面标准化,无未来函数)。

elastic_net 与 lasso 共用:只是 make_model 不同。
"""
import numpy as np
import pandas as pd

from strategies.ml import _features


def _zscore(frame):
    std = frame.std(axis=1).replace(0.0, np.nan)
    z = frame.sub(frame.mean(axis=1), axis=0).div(std, axis=0)
    # 截面同值(std=0)时 z 记 0;原本缺失的保持 NaN
    return z.fillna(0.0).where(frame.notna())


def regression_signal(panel, make_model, top_n, train, retrain, horizon,
                      feat_windows=(21, 63, 126)):
    close = panel["close"]
    feats = {name: _zscore(f) for name, f in _features(panel, feat_windows).items()}
    forward = _zscore(close.shift(-horizon) / close - 1.0)

    X = np.stack([f.values for f in feats.values()], axis=-1)  # (T, N, F)
    y = forward.values
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in range(train, len(close), retrain):
        # 训练集止于 t - horizon:标签在 t 日已完全实现,无泄漏
        window = slice(max(0, t - train), t - horizon)
        train_x = X[window].reshape(-1, X.shape[-1])
        train_y = y[window].reshape(-1)
        ok = ~np.isnan(train_x).any(axis=1) & ~np.isnan(train_y)
        if ok.sum() < 30:
            continue
        model = make_model()
        model.fit(train_x[ok], train_y[ok])

        now = X[t]
        ok_now = ~np.isnan(now).any(axis=1)
        if not ok_now.any():
            continue
        pred = pd.Series(np.nan, index=close.columns)
        pred[ok_now] = model.predict(now[ok_now])
        top = pred.nlargest(min(top_n, int(ok_now.sum()))).index
        row = pd.Series(0.0, index=close.columns)
        row[top] = 1.0 / len(top)
        weights.iloc[t] = row
    return weights.ffill().fillna(0.0)
