import numpy as np
import pandas as pd
import pytest
from alpha101 import data


def _raw():
    rows = []
    for code in ["000001", "000002"]:
        for i, day in enumerate(pd.date_range("2020-01-01", periods=3)):
            rows.append(dict(code=code, date=day, open=10+i, high=11+i,
                             low=9+i, close=10.5+i, volume=100+i, amount=(10.5+i)*(100+i)))
    return pd.DataFrame(rows)


def test_build_panel_shapes_and_fields():
    p = data.build_panel(_raw())
    assert set(["open", "high", "low", "close", "volume", "amount", "vwap", "returns"]) <= set(p)
    assert list(p["close"].columns) == ["000001", "000002"]
    assert p["close"].shape == (3, 2)


def test_vwap_is_amount_over_volume():
    p = data.build_panel(_raw())
    assert p["vwap"].iloc[0, 0] == pytest.approx(10.5)  # amount/volume = (10.5*100)/100


def test_returns_first_row_nan():
    p = data.build_panel(_raw())
    assert p["returns"].iloc[0].isna().all()


def test_adv_mean_amount():
    p = data.build_panel(_raw())
    a = data.adv(p, 2)
    expected = (p["amount"].iloc[0, 0] + p["amount"].iloc[1, 0]) / 2
    assert a.iloc[1, 0] == pytest.approx(expected)
