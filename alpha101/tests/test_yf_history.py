import json

import pandas as pd
import pytest

from alpha101.yf_history import (
    build_panel,
    default_universe,
    load_universe,
    normalize_download,
    to_yf_ticker,
)


def test_to_yf_ticker_hk_strips_leading_zero():
    assert to_yf_ticker("00700", "hk") == "0700.HK"
    assert to_yf_ticker("09988", "hk") == "9988.HK"


def test_to_yf_ticker_us_replaces_dot():
    assert to_yf_ticker("BRK.B", "us") == "BRK-B"
    assert to_yf_ticker("NVDA", "us") == "NVDA"


def test_load_universe_keeps_hk_and_us_codes(tmp_path):
    path = tmp_path / "universe.json"
    pd.DataFrame({
        "code": ["00700", "NVDA"],
        "name": ["腾讯控股", "NVIDIA"],
        "g": ["媒体与娱乐", "半导体与半导体设备"],
    }).to_json(path, orient="records", force_ascii=False)

    result = load_universe(path)

    assert result["code"].tolist() == ["00700", "NVDA"]
    assert result["name"].tolist() == ["腾讯控股", "NVIDIA"]
    assert result["g"].tolist() == ["媒体与娱乐", "半导体与半导体设备"]


def test_default_universe_reads_latest_manifest_date(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"dates": ["2026-07-09", "2026-07-10"]})
    )
    (tmp_path / "manifest_us.json").write_text(
        json.dumps({"dates": ["2026-07-08", "2026-07-09"]})
    )

    assert default_universe("hk", tmp_path) == tmp_path / "2026-07-10.json"
    assert default_universe("us", tmp_path) == tmp_path / "us-2026-07-09.json"


def test_normalize_download_maps_tickers_back_to_codes():
    dates = pd.to_datetime(["2026-07-09", "2026-07-10"])
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["0700.HK", "9988.HK"]]
    )
    data = pd.DataFrame(
        [[10.0, 20.0, 11.0, 21.0, 9.0, 19.0, 10.5, 20.5, 1000, 2000],
         [10.5, 20.5, 11.5, 21.5, 10.0, 20.0, 10.8, 20.8, 1500, 2500]],
        index=dates,
        columns=columns,
    )

    result = normalize_download(data, {"0700.HK": "00700", "9988.HK": "09988"})

    assert set(result["code"]) == {"00700", "09988"}
    row = result[(result["code"] == "00700") & (result["date"] == dates[0])].iloc[0]
    assert row["close"] == 10.5
    assert row["volume"] == 1000


def test_normalize_download_drops_all_nan_tickers():
    dates = pd.to_datetime(["2026-07-09"])
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["0700.HK", "DEAD.HK"]]
    )
    data = pd.DataFrame(
        [[10.0, None, 11.0, None, 9.0, None, 10.5, None, 1000, None]],
        index=dates,
        columns=columns,
    )

    result = normalize_download(data, {"0700.HK": "00700", "DEAD.HK": "99999"})

    assert result["code"].tolist() == ["00700"]


def test_build_panel_approximates_vwap_and_amount():
    raw = pd.DataFrame({
        "code": ["00700", "00700"],
        "date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.0, 10.0],
        "close": [10.5, 10.8],
        "volume": [1000, 1500],
    })

    panel = build_panel(raw)

    vwap = (11.0 + 9.0 + 10.5) / 3
    assert panel["vwap"].loc[pd.Timestamp("2026-07-09"), "00700"] == pytest.approx(vwap)
    assert panel["amount"].loc[pd.Timestamp("2026-07-09"), "00700"] == pytest.approx(
        vwap * 1000
    )
    assert panel["returns"].loc[pd.Timestamp("2026-07-10"), "00700"] == pytest.approx(
        10.8 / 10.5 - 1
    )


def test_build_panel_attaches_industries():
    raw = pd.DataFrame({
        "code": ["00700"],
        "date": pd.to_datetime(["2026-07-09"]),
        "open": [10.0],
        "high": [11.0],
        "low": [9.0],
        "close": [10.5],
        "volume": [1000],
    })
    industries = pd.Series({"00700": "媒体与娱乐"})

    panel = build_panel(raw, industries)

    assert panel["ind"].loc["00700"] == "媒体与娱乐"
