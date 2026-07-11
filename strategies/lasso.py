"""Lasso 滚动多因子选股:L1 自动筛因子,预测未来 horizon 日收益。

参照 BigQuant 文档 J4Bwgm4r4j:预测 10 日收益、top 20 等权、每 10 日调仓;
原文的基本面因子以价量因子替代(面板所限)。
"""
from strategies.rolling_reg import regression_signal


def signal(panel, top_n=20, train=252, retrain=10, horizon=10,
           feat_windows=(21, 63, 126), alpha=0.005):
    from sklearn.linear_model import Lasso

    return regression_signal(
        panel,
        lambda: Lasso(alpha=alpha, max_iter=2000),
        top_n=top_n, train=train, retrain=retrain, horizon=horizon,
        feat_windows=feat_windows,
    )
