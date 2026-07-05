import numpy as np
import pandas as pd
from alpha101 import fundamental as fu


def test_point_in_time_only_after_announce():
    # 一只票两期财报,公告日分别 01-10、04-15。交易日面板应在公告后才出现对应值,之前为 NaN。
    raw = pd.DataFrame({
        "code": ["A", "A"], "report": ["20231231", "20240331"],
        "announce": ["2024-01-10", "2024-04-15"],
        "eps": [1.0, 1.5], "rev_yoy": [10.0, 20.0], "profit_yoy": [5.0, 8.0],
        "bvps": [8.0, 8.5], "roe": [12.0, 13.0], "ocfps": [1.2, 1.4], "gross_margin": [30.0, 31.0],
    })
    td = pd.bdate_range("2024-01-01", "2024-05-01")
    fp = fu.build_fund_panel(raw, td)
    eps = fp["eps"]["A"]
    assert eps.loc["2024-01-05"] != eps.loc["2024-01-05"]      # NaN(公告前)
    assert eps.loc["2024-02-01"] == 1.0                        # 第一期已公告
    assert eps.loc["2024-04-22"] == 1.5                        # 第二期已公告(前向填充为最新)


def test_factors_orientation_and_shape():
    close = pd.DataFrame([[10.0, 20.0]], index=[pd.Timestamp("2024-02-01")], columns=["A", "B"])
    fp = {f: pd.DataFrame([[8.0, 8.0]], index=[pd.Timestamp("2024-02-01")], columns=["A", "B"])
          for f in fu.FIELDS}
    facs = fu.fundamental_factors(fp, close)
    # B/P = bvps/close:A(8/10)应大于 B(8/20)
    assert facs["f_bp"].loc["2024-02-01", "A"] > facs["f_bp"].loc["2024-02-01", "B"]
    assert set(facs) == {"f_bp", "f_ep", "f_roe", "f_gm", "f_ocfq", "f_revg", "f_profg"}
