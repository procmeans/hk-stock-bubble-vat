from pathlib import Path

import pandas as pd
import pytest

import alpha101.ths_history as ths_history
from alpha101.ths_history import (
    build_panel,
    fetch_history,
    normalize_history_frame,
    read_raw_cache,
    write_raw_cache,
)
from alpha101.ths_today import load_code_pool


def test_normalize_history_frame_maps_thscode_and_time():
    data = pd.DataFrame({
        "thscode": ["000001.SZ"],
        "time": ["2026-07-08"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.0],
        "close": [10.5],
        "volume": [1000],
        "amount": [10500],
    })

    result = normalize_history_frame(data)

    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "date"] == pd.Timestamp("2026-07-08")
    assert result.loc[0, "close"] == 10.5


def test_build_panel_uses_ifind_share_volume_for_vwap():
    raw = pd.DataFrame({
        "code": ["000001", "000001", "000002", "000002"],
        "date": pd.to_datetime(["2026-07-07", "2026-07-08"] * 2),
        "open": [10.0, 10.5, 20.0, 20.5],
        "high": [11.0, 11.0, 21.0, 21.0],
        "low": [9.0, 10.0, 19.0, 20.0],
        "close": [10.5, 10.8, 20.5, 20.8],
        "volume": [1000, 2000, 1000, 2000],
        "amount": [10500, 21600, 20500, 41600],
    })

    panel = build_panel(raw)

    assert panel["vwap"].loc[pd.Timestamp("2026-07-07"), "000001"] == pytest.approx(10.5)
    assert panel["returns"].loc[pd.Timestamp("2026-07-08"), "000001"] == pytest.approx(
        10.8 / 10.5 - 1
    )


def test_load_code_pool_excludes_non_a_share_codes(tmp_path):
    path = tmp_path / "universe.json"
    pd.DataFrame({
        "code": ["000001", "920950", "832000"],
        "name": ["平安银行", "迅安科技", "旧三板"],
    }).to_json(path, orient="records", force_ascii=False)

    result = load_code_pool(path)

    assert result["code"].tolist() == ["000001", "920950"]


def test_pickle_cache_round_trip(tmp_path):
    path = tmp_path / "panel.pkl"
    raw = pd.DataFrame({"code": ["000001"], "date": [pd.Timestamp("2026-07-08")]})

    write_raw_cache(raw, path)
    result = read_raw_cache(path)

    assert result.loc[0, "code"] == "000001"


def test_fetch_history_pins_unadjusted_omit_request(monkeypatch, tmp_path):
    universe = tmp_path / "universe.json"
    pd.DataFrame({"code": ["000001"], "name": ["平安银行"]}).to_json(
        universe,
        orient="records",
        force_ascii=False,
    )
    calls = []

    def fake_history(codes, indicators, start, end, **kwargs):
        calls.append((codes, indicators, start, end, kwargs))
        return pd.DataFrame({
            "thscode": ["000001.SZ"],
            "time": ["2026-07-10"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1000.0],
        })

    monkeypatch.setattr(ths_history.ths_http, "history_quotation", fake_history)
    monkeypatch.setattr(ths_history.time, "sleep", lambda _: None)

    fetch_history(
        universe,
        "2022-07-01",
        "2026-07-10",
        tmp_path / "panel.pkl",
        batch_size=80,
        pause=0,
    )

    assert calls == [
        (
            ["000001.SZ"],
            "open,high,low,close,volume,amount",
            "2022-07-01",
            "2026-07-10",
            {"functionpara": {"CPS": "1", "Fill": "Omit"}},
        )
    ]


def test_root_readme_pins_reproducible_daily_history_provenance():
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    assert (
        "python -m alpha101.ths_history fetch --universe "
        "data/a-2026-07-07.json --start 2022-07-01 --end 2026-07-10 "
        "--cache alpha101/cache/ths_panel.pkl --batch-size 80"
    ) in readme
    for text in [
        "CPS=1",
        "Fill=Omit",
        "code,date,open,high,low,close,volume,amount",
        "4,873,244",
        "5,203",
        "abc0256b0985eca70ef4b4afb88e2cc8934bfb0a7174ed7be35fcd22443ed583",
        "783f2580d90347554111ff0b91ce0df4f5ce654ad62c28b01ac7f1f75a3adc84",
    ]:
        assert text in readme
