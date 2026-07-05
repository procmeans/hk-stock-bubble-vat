"""论文《101 Formulaic Alphas》算子实现。所有算子作用于 DataFrame[日期×股票]。"""
import numpy as np
import pandas as pd


def _d(d):
    """窗口天数:浮点向下取整为 int。"""
    return int(np.floor(d))


# ---- 逐元素 ----
def abs_(x):
    return x.abs()


def log_(x):
    return np.log(x)


def sign_(x):
    return np.sign(x)


def signedpower(x, a):
    return np.sign(x) * (x.abs() ** a)


# ---- 截面 ----
def rank(x):
    """截面百分位排名 [0,1](按行)。"""
    return x.rank(axis=1, pct=True)


def scale(x, a=1.0):
    """截面缩放,使每行 sum(abs)=a。"""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.mul(a).div(denom, axis=0)


# ---- 逐元素二元(见 alphas 里的 min/max/rank 相减)----
def rank_sub(x, y):
    return x - y


def ew_min(x, y):
    """逐元素最小(两个 DataFrame)。"""
    return pd.DataFrame(np.minimum(x.values, y.values), index=x.index, columns=x.columns)


def ew_max(x, y):
    return pd.DataFrame(np.maximum(x.values, y.values), index=x.index, columns=x.columns)


# ---- 时序 ----
def delay(x, d):
    return x.shift(_d(d))


def delta(x, d):
    return x - x.shift(_d(d))


def ts_sum(x, d):
    return x.rolling(_d(d)).sum()


def ts_product(x, d):
    return x.rolling(_d(d)).apply(np.prod, raw=True)


def ts_stddev(x, d):
    return x.rolling(_d(d)).std()


def ts_min(x, d):
    return x.rolling(_d(d)).min()


def ts_max(x, d):
    return x.rolling(_d(d)).max()


def ts_argmin(x, d):
    return x.rolling(_d(d)).apply(np.argmin, raw=True)


def ts_argmax(x, d):
    return x.rolling(_d(d)).apply(np.argmax, raw=True)


def ts_rank(x, d):
    def _r(w):
        return pd.Series(w).rank(pct=True).iloc[-1]
    return x.rolling(_d(d)).apply(_r, raw=True)


def decay_linear(x, d):
    d = _d(d)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d).apply(lambda a: np.dot(a, w), raw=True)


def correlation(x, y, d):
    return x.rolling(_d(d)).corr(y)


def covariance(x, y, d):
    return x.rolling(_d(d)).cov(y)


# 别名:论文中 min(x,d)/max(x,d) 即时序 min/max
min_ = ts_min
max_ = ts_max
