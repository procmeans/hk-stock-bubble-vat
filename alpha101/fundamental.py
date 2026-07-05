"""基本面因子:按公告日对齐的 point-in-time 日频面板 + 价值/质量/成长因子。

设计要点:只用"当日已公告"的财报(按 announce 公告日前向填充),严格避免未来函数。
方向统一为"越大越好"(便宜/高质量/高成长),再交由 compose 的 IC 加权确认符号。
"""
import numpy as np
import pandas as pd

FIELDS = ["eps", "rev_yoy", "profit_yoy", "bvps", "roe", "ocfps", "gross_margin"]


def build_fund_panel(fund_raw, trading_days):
    """季度财报长表 -> {字段: DataFrame[交易日 × 代码]},按公告日前向填充(point-in-time)。"""
    f = fund_raw.copy()
    f["announce"] = pd.to_datetime(f["announce"], errors="coerce")
    f = f.dropna(subset=["announce"])
    f["code"] = f["code"].astype(str)
    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days)))
    panel = {}
    for fld in FIELDS:
        f[fld] = pd.to_numeric(f[fld], errors="coerce")
        pv = f.pivot_table(index="announce", columns="code", values=fld, aggfunc="last").sort_index()
        pv = pv.reindex(pv.index.union(td)).sort_index().ffill().reindex(td)
        panel[fld] = pv
    return panel


def fundamental_factors(fp, close):
    """基本面因子 dict{名称: DataFrame}。close 为价格面板(日频×代码)。"""
    def al(x):
        return x.reindex(index=close.index, columns=close.columns)

    bvps, eps, roe = al(fp["bvps"]), al(fp["eps"]), al(fp["roe"])
    gm, ocf = al(fp["gross_margin"]), al(fp["ocfps"])
    rev_g, prof_g = al(fp["rev_yoy"]), al(fp["profit_yoy"])
    eps_nz = eps.replace(0, np.nan)
    return {
        "f_bp": bvps / close,          # 账面市值比 B/P(高=便宜)—— 价值
        "f_ep": eps / close,           # 盈利收益率 E/P(高=便宜,近似非TTM)—— 价值
        "f_roe": roe,                  # 净资产收益率 —— 质量
        "f_gm": gm,                    # 销售毛利率 —— 质量
        "f_ocfq": ocf / eps_nz,        # 经营现金流/每股收益 —— 盈利质量
        "f_revg": rev_g,               # 营收同比增速 —— 成长
        "f_profg": prof_g,             # 净利同比增速 —— 成长
    }
