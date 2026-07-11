"""机器学习:滚动窗口逻辑回归预测下期截面强弱(walk-forward,无未来函数)。"""
import numpy as np
import pandas as pd


def _features(panel, feat_windows):
    close, volume = panel["close"], panel["volume"]
    short = max(feat_windows[0], 2)
    feats = {f"ret{win}": close.pct_change(win) for win in feat_windows}
    feats["vol"] = close.pct_change().rolling(short * 2).std()
    feats["volu"] = (
        volume.rolling(short).mean() / volume.rolling(short * 3).mean()
    )
    return feats


def signal(panel, top_n=20, train=504, retrain=21, horizon=21,
           feat_windows=(21, 63, 126)):
    from sklearn.linear_model import LogisticRegression

    close = panel["close"]
    feats = _features(panel, feat_windows)
    forward = close.shift(-horizon) / close - 1.0
    label = forward.gt(forward.median(axis=1), axis=0)

    X = np.stack([f.values for f in feats.values()], axis=-1)  # (T, N, F)
    y = label.values
    y_known = forward.notna().values
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in range(train, len(close), retrain):
        # 训练集止于 t - horizon:标签在 t 日已完全实现,无泄漏
        train_x = X[max(0, t - train): t - horizon].reshape(-1, X.shape[-1])
        train_y = y[max(0, t - train): t - horizon].reshape(-1)
        known = y_known[max(0, t - train): t - horizon].reshape(-1)
        ok = ~np.isnan(train_x).any(axis=1) & known
        if ok.sum() < 30 or len(np.unique(train_y[ok])) < 2:
            continue
        model = LogisticRegression(max_iter=200)
        model.fit(train_x[ok], train_y[ok])

        now = X[t]
        ok_now = ~np.isnan(now).any(axis=1)
        if not ok_now.any():
            continue
        prob = pd.Series(np.nan, index=close.columns)
        prob[ok_now] = model.predict_proba(now[ok_now])[:, 1]
        top = prob.nlargest(min(top_n, int(ok_now.sum()))).index
        row = pd.Series(0.0, index=close.columns)
        row[top] = 1.0 / len(top)
        weights.iloc[t] = row
    return weights.ffill().fillna(0.0)
