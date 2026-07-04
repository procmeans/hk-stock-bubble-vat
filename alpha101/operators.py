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
