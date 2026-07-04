"""每日有效股票池 mask。"""
import numpy as np
import pandas as pd


def liquidity_mask(panel, min_days=60, adv_window=20, drop_pct=0.20, st_codes=None):
    vol = panel["volume"]
    amt = panel["amount"]
    mask = pd.DataFrame(True, index=vol.index, columns=vol.columns)

    # 当日停牌:volume 为 0 或 NaN
    mask &= vol.fillna(0) > 0

    # 次新:每只股票前 min_days 个有效交易日置 False
    if min_days > 0:
        traded = vol.fillna(0) > 0
        age = traded.cumsum()
        mask &= age > min_days

    # 流动性:过去 adv_window 日均额,截面后 drop_pct 分位剔除
    if drop_pct > 0:
        advw = amt.rolling(adv_window, min_periods=1).mean()
        thresh = advw.quantile(drop_pct, axis=1)
        mask &= advw.ge(thresh, axis=0)

    # ST
    if st_codes:
        for c in st_codes:
            if c in mask.columns:
                mask[c] = False

    return mask


def apply_mask(factor, mask):
    return factor.where(mask.reindex_like(factor).fillna(False))
