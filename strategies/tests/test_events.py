import numpy as np
import pandas as pd
import pytest

from strategies import events


def _events(rows):
    return pd.DataFrame(rows, columns=["code", "notice_date", "report_date",
                                       "type", "amp_lower", "amp_upper"])


def test_signal_aligns_weekend_notice_and_holds(make_panel):
    n = 15
    panel = make_panel({"a": [100.0] * n, "b": [100.0] * n})
    idx = panel["close"].index                      # 从 2024-01-01(周一)起的工作日
    ev = _events([
        # 周六公告 -> 信号日为下周一(idx[5])
        ("a", "2024-01-06", "2023-12-31", "预增", 50.0, 80.0),
    ])

    w = events.signal(panel, events=ev, hold=3)

    monday = idx.get_loc(pd.Timestamp("2024-01-08"))
    assert (w["a"].iloc[monday:monday + 3] == 1.0).all()   # 持有 3 个交易日
    assert w["a"].iloc[monday + 3] == 0.0                  # 窗口结束离场
    assert (w["a"].iloc[:monday] == 0).all()


def test_signal_equal_weights_overlapping_events(make_panel):
    n = 10
    panel = make_panel({"a": [100.0] * n, "b": [100.0] * n})
    ev = _events([
        ("a", "2024-01-02", "2023-12-31", "预增", 10, 20),
        ("b", "2024-01-02", "2023-12-31", "扭亏", None, None),
    ])

    w = events.signal(panel, events=ev, hold=5)

    day = panel["close"].index.get_loc(pd.Timestamp("2024-01-02"))
    assert w.iloc[day]["a"] == pytest.approx(0.5)
    assert w.iloc[day]["b"] == pytest.approx(0.5)


def test_signal_filters_negative_types(make_panel):
    panel = make_panel({"a": [100.0] * 8})
    ev = _events([("a", "2024-01-02", "2023-12-31", "首亏", None, None)])

    w = events.signal(panel, events=ev, hold=5)

    assert (w == 0).all().all()


def test_car_hand_computed(make_panel):
    # a 在事件日后每天异常收益 +1%(b 为基准对照,恒定 0%)
    n = 12
    a = [100.0] * 4
    for _ in range(n - 4):
        a.append(a[-1] * 1.01)
    panel = make_panel({"a": a, "b": [100.0] * n})
    ev = _events([("a", "2024-01-04", "2023-12-31", "预增", 10, 20)])

    result = events.car(panel, events=ev, pre=1, post=3, min_events=1)

    row = result.loc["预增"]
    assert row["n_events"] == 1
    # 异常收益 = a 的 1% − 截面均值 0.5% = 0.5%/日,事件日后 3 日 CAR ≈ 1.5%
    assert row["car_post"] == pytest.approx(0.015, rel=0.05)


def test_parse_forecast_page():
    payload = {"result": {"pages": 1, "data": [{
        "SECURITY_CODE": "2714", "SECURITY_NAME_ABBR": "牧原股份",
        "NOTICE_DATE": "2026-07-11 00:00:00", "REPORT_DATE": "2026-06-30 00:00:00",
        "PREDICT_TYPE": "首亏", "ADD_AMP_LOWER": -162.09, "ADD_AMP_UPPER": -152.83,
    }]}}

    rows = events.parse_forecast_page(payload)

    assert rows == [{"code": "002714", "notice_date": "2026-07-11",
                     "report_date": "2026-06-30", "type": "首亏",
                     "amp_lower": -162.09, "amp_upper": -152.83}]
