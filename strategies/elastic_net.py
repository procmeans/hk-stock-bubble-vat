"""弹性网络选股:滚动 ElasticNet 回归预测下期截面收益。

参照 BigQuant 文档 qjl2v0tXah:训练窗 240 日、每 20 日调仓、top 30 等权、
alpha=0.005、l1_ratio=0.5;原文的基本面因子以价量因子替代(面板所限)。
"""
from strategies.rolling_reg import regression_signal


def signal(panel, top_n=30, train=240, retrain=20, horizon=20,
           feat_windows=(21, 63, 126), alpha=0.005, l1_ratio=0.5):
    from sklearn.linear_model import ElasticNet

    return regression_signal(
        panel,
        lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000),
        top_n=top_n, train=train, retrain=retrain, horizon=horizon,
        feat_windows=feat_windows,
    )
