from pathlib import Path

import pandas as pd
import pytest

from intraday.data import load_daily_raw, prepare_universe


def _daily_row(code, day, amount, volume=100.0):
    return {
        "code": code,
        "date": day,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": volume,
        "amount": amount,
    }


@pytest.mark.parametrize(
    ("suffix", "expected_reader"),
    [(".pkl", "pickle"), (".PARQUET", "parquet")],
)
def test_load_daily_raw_dispatches_by_suffix(monkeypatch, suffix, expected_reader):
    path = Path(f"daily{suffix}")
    expected = pd.DataFrame({"code": ["000001"]})
    calls = []

    def read_pickle(candidate):
        calls.append(("pickle", candidate))
        return expected

    def read_parquet(candidate):
        calls.append(("parquet", candidate))
        return expected

    monkeypatch.setattr(pd, "read_pickle", read_pickle)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    result = load_daily_raw(path)

    assert result is expected
    assert calls == [(expected_reader, path)]


def test_prepare_universe_uses_lagged_adv_and_deterministic_ties():
    dates = pd.bdate_range("2026-01-01", periods=6)
    amounts = {
        "000001": [100, 100, 1, 1, 1, 1],
        "000002": [50, 50, 200, 200, 200, 200],
        "000003": [50, 50, 40, 40, 40, 40],
    }
    raw = pd.DataFrame(
        [
            _daily_row(code, day, amount)
            for code, values in amounts.items()
            for day, amount in zip(dates, values)
        ]
    )

    plan = prepare_universe(
        raw,
        dates[2],
        dates[-1],
        top=2,
        adv_window=2,
        min_age=0,
    )

    first = plan["ranked_pool"].query("date == @dates[2]")
    assert first["code"].tolist() == ["000001", "000002"]
    assert first["adv20"].tolist() == [100.0, 50.0]
    assert first["liquidity_rank"].tolist() == [1, 2]
    assert plan["candidates"] == ["000001", "000002", "000003"]


def test_prepare_universe_filters_suspension_after_top_rank_without_replacement():
    dates = pd.bdate_range("2025-10-01", periods=65)
    raw = pd.DataFrame(
        [
            _daily_row(code, day, amount)
            for code, amount in [
                ("000001", 300.0),
                ("000002", 200.0),
                ("000003", 100.0),
            ]
            for day in dates
        ]
    )
    raw.loc[
        (raw["code"] == "000001") & (raw["date"] == dates[-1]), "volume"
    ] = 0

    plan = prepare_universe(raw, dates[-2], dates[-1], top=2, min_age=60)

    last = plan["eligible_pool"].query("date == @dates[-1]")
    assert last["code"].tolist() == ["000002"]
    assert "000003" not in last["code"].tolist()


def test_prepare_universe_filters_young_stock_after_top_rank():
    dates = pd.bdate_range("2025-10-01", periods=65)
    established = [
        _daily_row("000002", day, 200.0)
        for day in dates
    ]
    young = [
        _daily_row("000001", day, 300.0)
        for day in dates[-21:]
    ]
    raw = pd.DataFrame(established + young)

    plan = prepare_universe(
        raw,
        dates[-1],
        dates[-1],
        top=2,
        adv_window=20,
        min_age=60,
    )

    ranked = plan["ranked_pool"].query("date == @dates[-1]")
    eligible = plan["eligible_pool"].query("date == @dates[-1]")
    assert ranked["code"].tolist() == ["000001", "000002"]
    assert eligible["code"].tolist() == ["000002"]


def test_prepare_universe_includes_warmup_dates_and_estimates_three_fields():
    dates = pd.bdate_range("2025-12-11", periods=25)
    raw = pd.DataFrame(
        [_daily_row(1, day, 100.0) for day in dates]
    )

    plan = prepare_universe(
        raw,
        dates[20],
        dates[-1],
        top=500,
        adv_window=20,
        min_age=0,
    )

    assert plan["eval_dates"].equals(dates[20:])
    assert plan["fetch_dates"].equals(dates)
    assert plan["candidates"] == ["000001"]
    assert plan["estimated_rows"] == len(dates) * 241
    assert plan["estimated_cells"] == len(dates) * 241 * 3
