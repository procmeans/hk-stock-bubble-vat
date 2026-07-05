"""论文《101 Formulaic Alphas》算子实现。所有算子作用于 DataFrame[日期×股票]。"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _d(d):
    """窗口天数:浮点向下取整为 int。"""
    return int(np.floor(d))


def _roll_apply_vec(x, d, reducer):
    """把逐窗归约向量化(替代慢的 rolling.apply)。
    reducer 输入形状 (n-d+1, m, d) 的窗口数组、沿最后一维归约,返回 (n-d+1, m)。
    含 NaN 的窗口结果置 NaN,等价于 rolling(d) 默认 min_periods=d 的行为。"""
    d = _d(d)
    v = x.to_numpy(dtype=float)
    n, m = v.shape
    out = np.full((n, m), np.nan)
    if n >= d:
        win = sliding_window_view(v, d, axis=0)     # (n-d+1, m, d)
        res = reducer(win).astype(float)            # (n-d+1, m)
        res[np.isnan(win).any(axis=-1)] = np.nan
        out[d - 1:] = res
    return pd.DataFrame(out, index=x.index, columns=x.columns)


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
    return _roll_apply_vec(x, d, lambda w: np.prod(w, axis=-1))


def ts_stddev(x, d):
    return x.rolling(_d(d)).std()


def ts_min(x, d):
    return x.rolling(_d(d)).min()


def ts_max(x, d):
    return x.rolling(_d(d)).max()


def ts_argmin(x, d):
    return _roll_apply_vec(x, d, lambda w: np.argmin(w, axis=-1))


def ts_argmax(x, d):
    return _roll_apply_vec(x, d, lambda w: np.argmax(w, axis=-1))


def ts_rank(x, d):
    # 窗口内最新值的百分位秩(average 法),向量化:
    # pct = (严格小于个数 + (等于个数含自身+1)/2) / d,与 pandas rank(pct=True).iloc[-1] 等价。
    def _rank_last(w):
        last = w[..., -1:]
        less = (w < last).sum(axis=-1)
        eq = (w == last).sum(axis=-1)
        return (less + (eq + 1) / 2.0) / w.shape[-1]
    return _roll_apply_vec(x, d, _rank_last)


def decay_linear(x, d):
    # 线性衰减加权移动平均:权重 d,d-1,...,1(最近权重最大),归一化。
    # 用移位加权和向量化(等价于对窗口做加权点积,但避免逐窗 Python 调用)。
    d = _d(d)
    w = np.arange(d, 0, -1, dtype=float)   # shift 0(最近)权重 d,...,shift d-1 权重 1
    w /= w.sum()
    out = x * w[0]
    for k in range(1, d):
        out = out + x.shift(k) * w[k]
    return out


def correlation(x, y, d):
    return x.rolling(_d(d)).corr(y)


def covariance(x, y, d):
    return x.rolling(_d(d)).cov(y)


def indneutralize(x, groups):
    # 行业中性化:每个截面(每天)在每个行业组内去均值(demean)。
    # groups: Series(index=股票代码 -> 行业);缺失行业归入 UNKNOWN 组。
    g = pd.Series(groups).reindex(x.columns).fillna("UNKNOWN")
    grp_mean = x.T.groupby(g).transform("mean").T
    return x - grp_mean


# 别名:论文中 min(x,d)/max(x,d) 即时序 min/max
min_ = ts_min
max_ = ts_max
