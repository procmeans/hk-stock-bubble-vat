import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday.data import (
    day_complete,
    load_daily_raw,
    normalize_minute_day,
    prepare_universe,
    write_day_partition,
)


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


def _minute_times(day="2026-01-12", count=200):
    day = pd.Timestamp(day).normalize()
    morning_count = min(count, 100)
    afternoon_count = max(0, count - morning_count)
    morning = pd.date_range(day + pd.Timedelta(hours=9, minutes=31),
                            periods=morning_count, freq="min")
    afternoon = pd.date_range(day + pd.Timedelta(hours=13, minutes=1),
                              periods=afternoon_count, freq="min")
    return morning.append(afternoon)


def _minute_frame(count=200, amount=5.0, positive_volume_minutes=None):
    volume = np.ones(count)
    if positive_volume_minutes is not None:
        volume[positive_volume_minutes:] = 0.0
    return pd.DataFrame({
        "thscode": "000001.SZ",
        "time": _minute_times(count=count),
        "close": 10.0,
        "volume": volume,
        "amount": amount,
    })


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


def test_prepare_universe_returns_typed_empty_plan_outside_available_dates():
    day = pd.Timestamp("2026-01-02")
    raw = pd.DataFrame([_daily_row("000001", day, 100.0)])

    plan = prepare_universe(raw, "2027-01-01", "2027-01-31")

    assert set(plan) == {
        "eval_dates",
        "fetch_dates",
        "ranked_pool",
        "eligible_pool",
        "candidates",
        "estimated_rows",
        "estimated_cells",
    }
    assert plan["eval_dates"].empty
    assert plan["fetch_dates"].empty
    assert plan["candidates"] == []
    assert plan["estimated_rows"] == 0
    assert plan["estimated_cells"] == 0
    expected_columns = ["date", "code", "adv20", "liquidity_rank"]
    assert plan["ranked_pool"].columns.tolist() == expected_columns
    assert plan["eligible_pool"].columns.tolist() == expected_columns


def test_normalize_minute_day_records_amount_mismatch():
    frame = _minute_frame(count=200, amount=100.0)

    clean, coverage = normalize_minute_day(
        frame,
        pd.Timestamp("2026-01-12"),
        pd.Series({"000001": 25_000.0}),
    )

    assert clean.empty
    assert coverage.loc[0, "minute_count"] == 200
    assert coverage.loc[0, "amount_relative_error"] == pytest.approx(0.2)
    assert coverage.loc[0, "reason"] == "amount_mismatch"


def test_normalize_minute_day_keeps_only_valid_original_minute_fields():
    day = pd.Timestamp("2026-01-12")
    frame = _minute_frame()
    frame["open"] = 9.0
    frame["fill"] = "Original"
    duplicate = frame.iloc[[0]].assign(close=11.0)
    invalid = pd.DataFrame({
        "thscode": ["000001.SZ"] * 6,
        "time": [
            day - pd.Timedelta(days=1) + pd.Timedelta(hours=10),
            day + pd.Timedelta(hours=12),
            day + pd.Timedelta(hours=11, minutes=20),
            day + pd.Timedelta(hours=11, minutes=21),
            day + pd.Timedelta(hours=11, minutes=22),
            "not-a-time",
        ],
        "close": [10.0, 10.0, 0.0, 10.0, 10.0, 10.0],
        "volume": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
        "amount": [5.0, 5.0, 5.0, 5.0, -1.0, 5.0],
        "open": [9.0] * 6,
        "fill": ["Original"] * 6,
    })

    clean, coverage = normalize_minute_day(
        pd.concat([frame, duplicate, invalid], ignore_index=True),
        day,
        pd.Series({"000001": 1_000.0}),
    )

    assert clean.columns.tolist() == ["code", "time", "close", "volume", "amount"]
    assert len(clean) == 200
    assert clean.iloc[0]["close"] == 11.0
    assert clean["time"].is_monotonic_increasing
    assert coverage.loc[0, "reason"] == "ok"


@pytest.mark.parametrize(
    ("count", "positive_volume_minutes", "daily_amount", "reason"),
    [
        (199, 199, 995.0, "too_few_minutes"),
        (200, 29, 1_000.0, "too_few_trades"),
        (200, 30, 1_000.0, "ok"),
        (200, 200, 1_020.0 / 1.02, "ok"),
        (200, 200, 1_020.0 / 1.021, "amount_mismatch"),
    ],
)
def test_normalize_minute_day_applies_quality_thresholds(
    count,
    positive_volume_minutes,
    daily_amount,
    reason,
):
    clean, coverage = normalize_minute_day(
        _minute_frame(
            count=count,
            amount=5.1 if count == 200 and daily_amount != 1_000.0 else 5.0,
            positive_volume_minutes=positive_volume_minutes,
        ),
        pd.Timestamp("2026-01-12"),
        pd.Series({"000001": daily_amount}),
    )

    assert coverage.loc[0, "reason"] == reason
    assert clean.empty == (reason != "ok")


def test_normalize_minute_day_returns_typed_empty_outputs():
    clean, coverage = normalize_minute_day(
        pd.DataFrame(),
        pd.Timestamp("2026-01-12"),
        pd.Series(dtype=float),
    )

    assert clean.columns.tolist() == ["code", "time", "close", "volume", "amount"]
    assert coverage.columns.tolist() == [
        "date", "code", "minute_count", "amount_relative_error", "reason"
    ]
    assert clean.empty
    assert coverage.empty


def test_partition_is_complete_with_explicit_no_data(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")
    frame = pd.DataFrame({
        "code": ["000001"],
        "time": [day],
        "close": [10.0],
        "volume": [100.0],
        "amount": [1_000.0],
    })
    written_paths = []

    def write_fake_parquet(self, path, index):
        path = Path(path)
        written_paths.append(path)
        path.write_text("parquet payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)

    parquet, manifest = write_day_partition(
        frame,
        {"000002": "no_data", "000001": "returned"},
        day,
        tmp_path,
    )

    assert written_paths == [parquet.with_suffix(".parquet.tmp")]
    assert parquet.read_text() == "parquet payload"
    assert json.loads(manifest.read_text()) == {
        "date": "2026-01-12",
        "statuses": {"000001": "returned", "000002": "no_data"},
    }
    assert not parquet.with_suffix(".parquet.tmp").exists()
    assert not manifest.with_suffix(".json.tmp").exists()
    assert day_complete(day, ["000001", "000002"], tmp_path)
    assert not day_complete(day, ["000001", "000002", "000003"], tmp_path)
    assert not day_complete(day, ["000001"], tmp_path)


def test_day_complete_rejects_missing_files_and_unknown_status(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")

    assert not day_complete(day, ["000001"], tmp_path)

    def write_fake_parquet(self, path, index):
        Path(path).write_text("parquet payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    frame = pd.DataFrame({"code": ["000001"]})
    parquet, manifest = write_day_partition(
        frame, {"000001": "error"}, day, tmp_path
    )

    assert parquet.exists() and manifest.exists()
    assert not day_complete(day, ["000001"], tmp_path)

    manifest.write_text("not-json")
    assert not day_complete(day, ["000001"], tmp_path)

    manifest.write_text("[]")
    assert not day_complete(day, ["000001"], tmp_path)


def test_partition_does_not_publish_when_staging_fails(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")

    def fail_while_writing(self, path, index):
        Path(path).write_text("partial parquet")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_while_writing)

    with pytest.raises(OSError, match="simulated interrupted write"):
        write_day_partition(
            pd.DataFrame({"code": ["000001"]}),
            {"000001": "returned"},
            day,
            tmp_path,
        )

    partition_dir = tmp_path / "minute"
    assert not (partition_dir / "2026-01-12.parquet").exists()
    assert not (partition_dir / "2026-01-12.json").exists()
