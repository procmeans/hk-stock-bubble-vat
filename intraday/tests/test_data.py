import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

import intraday.data as intraday_data
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
    assert clean.dtypes.astype(str).to_dict() == {
        "code": "string",
        "time": "datetime64[ns]",
        "close": "float64",
        "volume": "float64",
        "amount": "float64",
    }
    assert coverage.dtypes.astype(str).to_dict() == {
        "date": "datetime64[ns]",
        "code": "string",
        "minute_count": "int64",
        "amount_relative_error": "float64",
        "reason": "string",
    }


def test_normalize_minute_day_records_code_with_zero_valid_minutes():
    day = pd.Timestamp("2026-01-12")
    frame = pd.DataFrame({
        "thscode": ["000002.SZ", "000002.SZ"],
        "time": [day + pd.Timedelta(hours=12), "not-a-time"],
        "close": [10.0, 10.0],
        "volume": [1.0, 1.0],
        "amount": [5.0, 5.0],
    })

    clean, coverage = normalize_minute_day(
        frame,
        day,
        pd.Series({"000002": 10.0}),
    )

    assert clean.empty
    assert coverage[["code", "minute_count", "reason"]].to_dict("records") == [{
        "code": "000002",
        "minute_count": 0,
        "reason": "too_few_minutes",
    }]


@pytest.mark.parametrize(
    "daily_amount",
    [pd.Series(dtype=float), pd.Series({"000001": 0.0})],
    ids=["missing", "zero"],
)
def test_normalize_minute_day_rejects_unusable_daily_amount(daily_amount):
    clean, coverage = normalize_minute_day(
        _minute_frame(),
        pd.Timestamp("2026-01-12"),
        daily_amount,
    )

    assert clean.empty
    assert np.isinf(coverage.loc[0, "amount_relative_error"])
    assert coverage.loc[0, "reason"] == "amount_mismatch"


def test_normalize_minute_day_includes_all_four_session_boundaries():
    day = pd.Timestamp("2026-01-12")
    boundaries = pd.DatetimeIndex([
        day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=11, minutes=30),
        day + pd.Timedelta(hours=13),
        day + pd.Timedelta(hours=15),
    ])
    boundary_rows = pd.DataFrame({
        "thscode": "000001.SZ",
        "time": boundaries,
        "close": 10.0,
        "volume": 1.0,
        "amount": 5.0,
    })

    clean, coverage = normalize_minute_day(
        pd.concat([_minute_frame(count=196), boundary_rows], ignore_index=True),
        day,
        pd.Series({"000001": 1_000.0}),
    )

    assert len(clean) == 200
    assert set(boundaries) <= set(clean["time"])
    assert coverage.loc[0, "reason"] == "ok"


def test_normalize_minute_day_rejects_non_finite_values():
    frame = _minute_frame(count=206)
    for position, (column, value) in enumerate([
        ("close", np.inf),
        ("close", -np.inf),
        ("volume", np.inf),
        ("volume", -np.inf),
        ("amount", np.inf),
        ("amount", -np.inf),
    ]):
        frame.loc[position, column] = value

    clean, coverage = normalize_minute_day(
        frame,
        pd.Timestamp("2026-01-12"),
        pd.Series({"000001": 1_000.0}),
    )

    assert len(clean) == 200
    assert np.isfinite(clean[["close", "volume", "amount"]]).all().all()
    assert coverage.loc[0, "minute_count"] == 200
    assert coverage.loc[0, "reason"] == "ok"


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


@pytest.mark.parametrize(
    ("payload", "codes"),
    [
        ({"statuses": {"000001": "returned"}}, ["000001"]),
        ({
            "date": "2026-01-13",
            "statuses": {"000001": "returned"},
        }, ["000001"]),
        ({"date": "2026-01-12"}, []),
        ({"date": "2026-01-12", "statuses": []}, ["000001"]),
        ({
            "date": "2026-01-12",
            "statuses": {"000001": ["returned"]},
        }, ["000001"]),
        ({
            "date": "2026-01-12",
            "statuses": {"000001": 1},
        }, ["000001"]),
    ],
)
def test_day_complete_rejects_invalid_manifest_structure(
    monkeypatch,
    tmp_path,
    payload,
    codes,
):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text("parquet payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    _, manifest = write_day_partition(
        pd.DataFrame({"code": ["000001"]}),
        {"000001": "returned"},
        day,
        tmp_path,
    )
    manifest.write_text(json.dumps(payload))

    assert day_complete(day, codes, tmp_path) is False


def test_day_complete_rejects_non_string_manifest_keys(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text("parquet payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    write_day_partition(
        pd.DataFrame({"code": ["000001"]}),
        {"000001": "returned"},
        day,
        tmp_path,
    )
    monkeypatch.setattr(
        intraday_data.json,
        "loads",
        lambda payload: {
            "date": "2026-01-12",
            "statuses": {1: "returned"},
        },
    )

    assert day_complete(day, ["000001"], tmp_path) is False


@pytest.mark.parametrize("codes", [[1], [["000001"]], None])
def test_day_complete_rejects_unsafe_requested_codes(monkeypatch, tmp_path, codes):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text("parquet payload")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    write_day_partition(
        pd.DataFrame({"code": ["000001"]}),
        {"000001": "returned"},
        day,
        tmp_path,
    )

    assert day_complete(day, codes, tmp_path) is False


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


@pytest.mark.parametrize("failed_stage", ["parquet", "manifest"])
def test_existing_partition_survives_staging_failure(
    monkeypatch,
    tmp_path,
    failed_stage,
):
    day = pd.Timestamp("2026-01-12")
    original_write_text = Path.write_text

    def write_fake_parquet(self, path, index):
        original_write_text(Path(path), str(self.loc[self.index[0], "close"]))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    old_frame = pd.DataFrame({"code": ["000001"], "close": [10.0]})
    new_frame = pd.DataFrame({"code": ["000001"], "close": [20.0]})
    statuses = {"000001": "returned"}
    parquet, _ = write_day_partition(old_frame, statuses, day, tmp_path)
    assert day_complete(day, ["000001"], tmp_path)

    failure_pending = True

    if failed_stage == "parquet":
        def fail_staging_parquet(self, path, index):
            nonlocal failure_pending
            write_fake_parquet(self, path, index)
            if failure_pending:
                failure_pending = False
                raise OSError("simulated parquet staging failure")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_staging_parquet)
    else:
        def fail_staging_manifest(self, data, *args, **kwargs):
            nonlocal failure_pending
            if self.name.endswith(".json.tmp") and failure_pending:
                failure_pending = False
                raise OSError("simulated manifest staging failure")
            return original_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_staging_manifest)

    with pytest.raises(OSError, match=f"simulated {failed_stage} staging failure"):
        write_day_partition(new_frame, statuses, day, tmp_path)

    assert day_complete(day, ["000001"], tmp_path)
    assert parquet.read_text() == "10.0"

    write_day_partition(new_frame, statuses, day, tmp_path)
    assert day_complete(day, ["000001"], tmp_path)
    assert parquet.read_text() == "20.0"


def test_existing_partition_survives_manifest_invalidation_failure(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text(str(self.loc[self.index[0], "close"]))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    old_frame = pd.DataFrame({"code": ["000001"], "close": [10.0]})
    new_frame = pd.DataFrame({"code": ["000001"], "close": [20.0]})
    statuses = {"000001": "returned"}
    parquet, manifest = write_day_partition(old_frame, statuses, day, tmp_path)
    previous_manifest = manifest.with_suffix(".json.previous")
    old_parquet = parquet.read_text()
    old_manifest = manifest.read_text()

    original_replace = Path.replace
    failure_pending = True

    def fail_invalidation_once(self, target):
        nonlocal failure_pending
        is_invalidation = self == manifest and Path(target) == previous_manifest
        if is_invalidation and failure_pending:
            failure_pending = False
            raise OSError("simulated manifest invalidation failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_invalidation_once)

    with pytest.raises(OSError, match="manifest invalidation failure"):
        write_day_partition(new_frame, statuses, day, tmp_path)

    assert parquet.read_text() == old_parquet
    assert manifest.read_text() == old_manifest
    assert not previous_manifest.exists()
    assert day_complete(day, ["000001"], tmp_path)

    write_day_partition(new_frame, statuses, day, tmp_path)
    assert parquet.read_text() == "20.0"
    assert day_complete(day, ["000001"], tmp_path)
    assert not previous_manifest.exists()


def test_orphaned_previous_manifest_never_completes_and_retry_cleans_it(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text(str(self.loc[self.index[0], "close"]))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    old_frame = pd.DataFrame({"code": ["000001"], "close": [10.0]})
    new_frame = pd.DataFrame({"code": ["000001"], "close": [20.0]})
    statuses = {"000001": "returned"}
    parquet, manifest = write_day_partition(old_frame, statuses, day, tmp_path)
    previous_manifest = manifest.with_suffix(".json.previous")
    manifest.replace(previous_manifest)

    assert parquet.exists()
    assert previous_manifest.exists()
    assert not manifest.exists()
    assert not day_complete(day, ["000001"], tmp_path)

    write_day_partition(new_frame, statuses, day, tmp_path)
    assert parquet.read_text() == "20.0"
    assert day_complete(day, ["000001"], tmp_path)
    assert not previous_manifest.exists()


@pytest.mark.parametrize("failed_suffix", [".parquet.tmp", ".json.tmp"])
def test_existing_partition_is_invalidated_when_publish_fails_and_retry_recovers(
    monkeypatch,
    tmp_path,
    failed_suffix,
):
    day = pd.Timestamp("2026-01-12")

    def write_fake_parquet(self, path, index):
        Path(path).write_text(str(self.loc[self.index[0], "close"]))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_fake_parquet)
    old_frame = pd.DataFrame({"code": ["000001"], "close": [10.0]})
    new_frame = pd.DataFrame({"code": ["000001"], "close": [20.0]})
    statuses = {"000001": "returned"}
    parquet, manifest = write_day_partition(old_frame, statuses, day, tmp_path)
    previous_manifest = manifest.with_suffix(".json.previous")
    assert day_complete(day, ["000001"], tmp_path)

    original_replace = Path.replace
    failure_pending = True

    def fail_publish_once(self, target):
        nonlocal failure_pending
        if self.name.endswith(failed_suffix) and failure_pending:
            failure_pending = False
            raise OSError(f"simulated {failed_suffix} publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish_once)

    with pytest.raises(OSError, match="publish failure"):
        write_day_partition(new_frame, statuses, day, tmp_path)

    assert not manifest.exists()
    assert previous_manifest.exists()
    assert not day_complete(day, ["000001"], tmp_path)

    write_day_partition(new_frame, statuses, day, tmp_path)
    assert day_complete(day, ["000001"], tmp_path)
    assert parquet.read_text() == "20.0"
    assert not previous_manifest.exists()


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


def test_retry_uses_fixed_backoff_then_recovers():
    calls = []
    waits = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.Timeout("temporary")
        return "ok"

    assert intraday_data._retry(flaky, sleeper=waits.append) == "ok"
    assert len(calls) == 3
    assert waits == [1, 2]


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("disconnected"),
        _http_error(429),
        _http_error(500),
        _http_error(503),
    ],
    ids=["connection", "429", "500", "503"],
)
def test_retry_classifies_transient_errors(error):
    calls = []
    waits = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "recovered"

    assert intraday_data._retry(flaky, sleeper=waits.append) == "recovered"
    assert len(calls) == 2
    assert waits == [1]


@pytest.mark.parametrize("status", [400, 401, 404])
def test_retry_raises_permanent_http_errors_immediately(status):
    waits = []
    error = _http_error(status)

    with pytest.raises(requests.HTTPError) as caught:
        intraday_data._retry(lambda: (_ for _ in ()).throw(error),
                             sleeper=waits.append)

    assert caught.value is error
    assert waits == []


def test_retry_stops_after_all_fixed_backoffs():
    calls = []
    waits = []

    def unavailable():
        calls.append(1)
        raise requests.Timeout("still unavailable")

    with pytest.raises(requests.Timeout, match="still unavailable"):
        intraday_data._retry(unavailable, sleeper=waits.append)

    assert len(calls) == 4
    assert waits == [1, 2, 4]


def test_fetch_minute_batches_codes_with_market_suffixes_and_exact_parameters(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")
    plan = {
        "candidates": [1, 600000.0, "430001.BJ", "920001"],
        "fetch_dates": pd.DatetimeIndex([day]),
    }
    raw_daily = pd.DataFrame({
        "code": ["000001.SZ", 600000, 430001.0, "920001.BJ"],
        "date": day,
        "amount": 1_000.0,
    })
    calls = []

    def fake_high_frequency(codes, indicators, starttime, endtime, **kwargs):
        calls.append((codes, indicators, starttime, endtime, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(intraday_data.ths_http, "high_frequency",
                        fake_high_frequency)
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda self, path, index: Path(path).write_text("empty"))

    result = intraday_data.fetch_minute_partitions(
        plan, raw_daily, tmp_path, "token", batch_size=2
    )

    assert [call[0] for call in calls] == [
        ["000001.SZ", "600000.SH"],
        ["430001.BJ", "920001.BJ"],
    ]
    assert all(call[1] == "close,volume,amount" for call in calls)
    assert all(call[2] == "2026-01-12 09:30:00" for call in calls)
    assert all(call[3] == "2026-01-12 15:00:00" for call in calls)
    assert all(call[4] == {
        "functionpara": {
            "CPS": "no",
            "Fill": "Original",
            "Timeformat": "LocalTime",
            "Limitstart": "09:30:00",
            "Limitend": "15:00:00",
        },
        "access_token": "token",
    } for call in calls)
    assert result.empty
    assert day_complete(
        day, ["000001", "600000", "430001", "920001"], tmp_path
    )


def test_fetch_minute_records_partial_statuses_and_reconciles_cps1_amount(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")
    candidates = ["000001", "000002", "600000"]
    plan = {
        "candidates": candidates,
        "fetch_dates": pd.DatetimeIndex([day]),
    }
    raw_daily = pd.DataFrame({
        "code": candidates,
        "date": day,
        "amount": [1_000.0, 2_000.0, 3_000.0],
    })
    returned = _minute_frame(count=200, amount=5.0)
    responses = iter([returned, pd.DataFrame()])
    written = []

    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(
        intraday_data,
        "write_day_partition",
        lambda frame, statuses, candidate_day, root: written.append(
            (frame.copy(), statuses.copy(), candidate_day, root)
        ),
    )

    coverage = intraday_data.fetch_minute_partitions(
        plan, raw_daily, tmp_path, "token", batch_size=2
    )

    clean, statuses, written_day, written_root = written[0]
    assert statuses == {
        "000001": "returned",
        "000002": "no_data",
        "600000": "no_data",
    }
    assert written_day == day
    assert written_root == tmp_path
    assert clean["amount"].sum() == 1_000.0
    assert clean["volume"].sum() == 200.0
    assert coverage[["code", "reason"]].to_dict("records") == [
        {"code": "000001", "reason": "ok"}
    ]


def test_fetch_minute_skips_completed_day(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")
    plan = {
        "candidates": ["000001"],
        "fetch_dates": pd.DatetimeIndex([day]),
    }
    raw_daily = pd.DataFrame({
        "code": ["000001"],
        "date": [day],
        "amount": [1_000.0],
    })
    empty = pd.DataFrame(columns=["code", "time", "close", "volume", "amount"])
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda self, path, index: Path(path).write_text("empty"))
    write_day_partition(empty, {"000001": "no_data"}, day, tmp_path)
    calls = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: calls.append(args) or pd.DataFrame(),
    )

    result = intraday_data.fetch_minute_partitions(
        plan, raw_daily, tmp_path, "token"
    )

    assert calls == []
    assert result.empty


def test_fetch_minute_handles_empty_candidate_batch_without_request(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")
    calls = []
    written = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: calls.append(args) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        intraday_data,
        "write_day_partition",
        lambda frame, statuses, candidate_day, root: written.append(
            (frame.copy(), statuses.copy())
        ),
    )

    result = intraday_data.fetch_minute_partitions(
        {"candidates": [], "fetch_dates": pd.DatetimeIndex([day])},
        pd.DataFrame(columns=["code", "date", "amount"]),
        tmp_path,
        "token",
    )

    assert calls == []
    assert len(written) == 1
    assert written[0][0].empty
    assert written[0][1] == {}
    assert result.empty


def test_fetch_adjusted_daily_uses_cps3_batches_and_normalizes(monkeypatch):
    calls = []
    first = pd.DataFrame({
        "thscode": ["000001.SZ", "000001.SZ", "600000.SH"],
        "time": ["2026-01-12", "2026-01-12", "2026-01-13"],
        "open": ["10", "99", "20"],
        "close": ["11", "99", "21"],
        "amount": [1, 2, 3],
    })
    responses = iter([first, pd.DataFrame()])

    def fake_history(codes, indicators, start, end, **kwargs):
        calls.append((codes, indicators, start, end, kwargs))
        return next(responses)

    monkeypatch.setattr(intraday_data.ths_http, "history_quotation", fake_history)

    result = intraday_data.fetch_adjusted_daily(
        [1, 600000.0, "430001.BJ", "920001"],
        "2026-01-01",
        "2026-01-31",
        "token",
        batch_size=2,
    )

    assert [call[0] for call in calls] == [
        ["000001.SZ", "600000.SH"],
        ["430001.BJ", "920001.BJ"],
    ]
    assert all(call[1] == "open,close" for call in calls)
    assert all(call[2:4] == ("2026-01-01", "2026-01-31") for call in calls)
    assert all(call[4] == {
        "functionpara": {"CPS": "3", "Fill": "Omit"},
        "access_token": "token",
    } for call in calls)
    assert result.columns.tolist() == ["code", "date", "open", "close"]
    assert result.to_dict("records") == [
        {
            "code": "000001",
            "date": pd.Timestamp("2026-01-12"),
            "open": 10,
            "close": 11,
        },
        {
            "code": "600000",
            "date": pd.Timestamp("2026-01-13"),
            "open": 20,
            "close": 21,
        },
    ]


@pytest.mark.parametrize("codes", [[], ["000001"]], ids=["empty-batch", "empty-response"])
def test_fetch_adjusted_daily_rejects_no_rows(monkeypatch, codes):
    calls = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "history_quotation",
        lambda *args, **kwargs: calls.append(args) or pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="adjusted history returned no rows"):
        intraday_data.fetch_adjusted_daily(
            codes, "2026-01-01", "2026-01-31", "token"
        )

    assert len(calls) == bool(codes)


def test_fetch_adjusted_daily_retries_transient_request_error(monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary")
        return pd.DataFrame({
            "thscode": ["000001.SZ"],
            "time": ["2026-01-12"],
            "open": [10],
            "close": [11],
        })

    monkeypatch.setattr(intraday_data.ths_http, "history_quotation", flaky)
    monkeypatch.setattr(intraday_data.time, "sleep", lambda seconds: None)

    result = intraday_data.fetch_adjusted_daily(
        ["000001"], "2026-01-01", "2026-01-31", "token"
    )

    assert len(calls) == 2
    assert result["code"].tolist() == ["000001"]


def test_build_attribute_query_binds_requested_date():
    assert intraday_data.build_attribute_query("2026-01-02") == (
        "2026年1月2日A股，2026年1月2日流通市值，所属同花顺行业"
    )


def test_normalize_attributes_finds_only_matching_dated_float_cap():
    raw = pd.DataFrame({
        "股票代码": ["000001.SZ", 1.0],
        "股票简称": ["平安银行", "重复记录"],
        "a股市值(不含限售股)[20260109]": [9.9e10, 9.8e10],
        "总市值(不含限售股)[20260112]": [8.8e11, 8.7e11],
        "A股市值(含限售股)[20260112]": [7.7e11, 7.6e11],
        "a股市值(不含限售股)[20260112]": [1.2e11, 1.1e11],
        "所属同花顺行业": ["银行-股份制银行", "银行"],
    })

    result = intraday_data.normalize_attributes(raw, pd.Timestamp("2026-01-12"))

    assert result.columns.tolist() == [
        "date", "code", "name", "float_cap", "industry"
    ]
    assert result.to_dict("records") == [{
        "date": pd.Timestamp("2026-01-12"),
        "code": "000001",
        "name": "平安银行",
        "float_cap": 1.2e11,
        "industry": "银行-股份制银行",
    }]


def test_normalize_attributes_requires_matching_dated_float_cap():
    raw = pd.DataFrame({
        "股票代码": ["000001.SZ"],
        "股票简称": ["平安银行"],
        "a股市值(不含限售股)[20260109]": [1.2e11],
        "所属同花顺行业": ["银行"],
    })

    with pytest.raises(
        ValueError,
        match="expected exactly one dated A-share float cap for 20260112; found 0",
    ):
        intraday_data.normalize_attributes(raw, "2026-01-12")


def test_fetch_attributes_queries_each_unique_anchor_once_and_retries(monkeypatch):
    calls = []

    def fake_query(query, **kwargs):
        calls.append((query, kwargs))
        if len(calls) == 1:
            raise requests.Timeout("temporary")
        stamp = "20260112" if "1月12日" in query else "20260119"
        return pd.DataFrame({
            "股票代码": ["000001.SZ"],
            "股票简称": ["平安银行"],
            f"a股市值(不含限售股)[{stamp}]": [1.2e11],
            "所属同花顺行业": ["银行"],
        })

    monkeypatch.setattr(intraday_data.ths_http, "smart_stock_picking", fake_query)
    monkeypatch.setattr(intraday_data.time, "sleep", lambda seconds: None)

    result = intraday_data.fetch_attributes(
        ["2026-01-19", "2026-01-12", "2026-01-12"], "token"
    )

    successful_queries = [query for query, _ in calls[1:]]
    assert successful_queries == [
        intraday_data.build_attribute_query("2026-01-12"),
        intraday_data.build_attribute_query("2026-01-19"),
    ]
    assert all(kwargs == {"access_token": "token", "timeout": 90}
               for _, kwargs in calls)
    assert result["date"].tolist() == [
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-01-19"),
    ]


def test_fetch_attributes_returns_schema_for_empty_anchor_list(monkeypatch):
    calls = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "smart_stock_picking",
        lambda *args, **kwargs: calls.append(args) or pd.DataFrame(),
    )

    result = intraday_data.fetch_attributes([], "token")

    assert calls == []
    assert result.empty
    assert result.columns.tolist() == [
        "date", "code", "name", "float_cap", "industry"
    ]


def test_apply_attribute_filters_drops_case_insensitive_st_and_invalid_cap_without_replacement():
    day = pd.Timestamp("2026-01-12")
    codes = ["000001", "000002", "000003", "000004", "000005", "000006"]
    pool = pd.DataFrame({
        "date": day,
        "code": codes,
        "liquidity_rank": range(1, len(codes) + 1),
    })
    attributes = pd.DataFrame({
        "date": day,
        "code": codes,
        "name": ["平安银行", "st测试", "*St测试", "零市值", "负市值", "无限市值"],
        "float_cap": [1e11, 2e10, 3e10, 0.0, -1.0, np.inf],
        "industry": ["银行"] * len(codes),
    })

    result = intraday_data.apply_attribute_filters(
        pool, attributes, pd.DatetimeIndex([day])
    )

    assert result["code"].tolist() == ["000001"]
    assert result["liquidity_rank"].tolist() == [1]


def test_apply_attribute_filters_allows_four_eval_day_lag_not_five():
    eval_dates = pd.bdate_range("2026-01-12", periods=6)
    pool = pd.DataFrame({
        "date": eval_dates,
        "code": "000001",
    })
    attributes = pd.DataFrame({
        "date": [eval_dates[0], eval_dates[0]],
        "code": ["000001", "000001"],
        "name": ["平安银行", "重复记录"],
        "float_cap": [1e11, 2e11],
        "industry": ["银行", "银行"],
    })

    result = intraday_data.apply_attribute_filters(pool, attributes, eval_dates)

    assert result["date"].tolist() == eval_dates[:5].tolist()
    assert result["code"].tolist() == ["000001"] * 5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "000001"),
        (np.int64(600000), "600000"),
        (1.0, "000001"),
        (np.float64(920001.0), "920001"),
        ("000001", "000001"),
        ("000001.SZ", "000001"),
        ("600000.SH", "600000"),
        ("430001.BJ", "430001"),
        ("920001.BJ", "920001"),
    ],
)
def test_normalize_code_accepts_only_supported_base_representations(
    value,
    expected,
):
    assert intraday_data._normalize_code(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        1.5,
        True,
        None,
        "1",
        "00001",
        "foo000001",
        "０００００１",
        "000001.sz",
        "000001.SH",
        "600000.SZ",
        "920001.SH",
    ],
)
def test_normalize_code_rejects_ambiguous_or_mismatched_values(value):
    with pytest.raises(ValueError, match="invalid stock code"):
        intraday_data._normalize_code(value)


def test_prepare_universe_rejects_malformed_string_code():
    day = pd.Timestamp("2026-01-12")
    raw = pd.DataFrame([_daily_row("1", day, 1_000.0)])

    with pytest.raises(ValueError, match="raw daily contains invalid stock code"):
        prepare_universe(raw, day, day, adv_window=1, min_age=0)


def test_write_partition_normalizes_manifest_code_keys(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index: Path(path).write_text("empty"),
    )

    _, manifest = write_day_partition(
        pd.DataFrame(),
        {1: "returned", "600000.SH": "no_data"},
        day,
        tmp_path,
    )

    assert json.loads(manifest.read_text())["statuses"] == {
        "000001": "returned",
        "600000": "no_data",
    }


def test_fetch_minute_rejects_malformed_wrong_market_and_unrequested_rows(
    monkeypatch,
    tmp_path,
):
    day = pd.Timestamp("2026-01-12")
    valid = _minute_frame(count=200, amount=5.0)
    malformed = valid.assign(thscode="foo000001.SZ")
    wrong_market = valid.assign(thscode="000002.SH")
    unrequested = valid.assign(thscode="000003.SZ")
    response = pd.concat(
        [valid, malformed, wrong_market, unrequested], ignore_index=True
    )
    written = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: response,
    )
    monkeypatch.setattr(
        intraday_data,
        "write_day_partition",
        lambda frame, statuses, candidate_day, root: written.append(
            (frame.copy(), statuses.copy())
        ),
    )

    coverage = intraday_data.fetch_minute_partitions(
        {"candidates": [1, 2.0], "fetch_dates": pd.DatetimeIndex([day])},
        pd.DataFrame({
            "code": ["000001.SZ", 2],
            "date": [day, day],
            "amount": [1_000.0, 1_000.0],
        }),
        tmp_path,
        "token",
    )

    clean, statuses = written[0]
    assert statuses == {"000001": "returned", "000002": "no_data"}
    assert clean["code"].unique().tolist() == ["000001"]
    assert coverage["code"].tolist() == ["000001"]


@pytest.mark.parametrize("missing", ["code", "date", "amount"])
def test_fetch_minute_validates_raw_daily_schema(monkeypatch, tmp_path, missing):
    day = pd.Timestamp("2026-01-12")
    raw = pd.DataFrame({
        "code": ["000001"],
        "date": [day],
        "amount": [1_000.0],
    }).drop(columns=missing)
    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: pytest.fail("network must not be called"),
    )

    with pytest.raises(
        ValueError,
        match=rf"raw_daily missing required columns: {missing}",
    ):
        intraday_data.fetch_minute_partitions(
            {"candidates": [], "fetch_dates": pd.DatetimeIndex([day])},
            raw,
            tmp_path,
            "token",
        )


@pytest.mark.parametrize("daily_amount", [np.nan, 0.0], ids=["missing", "zero"])
def test_fetch_minute_keeps_unusable_daily_amount_as_mismatch(
    monkeypatch,
    tmp_path,
    daily_amount,
):
    day = pd.Timestamp("2026-01-12")
    written = []
    monkeypatch.setattr(
        intraday_data.ths_http,
        "high_frequency",
        lambda *args, **kwargs: _minute_frame(count=200, amount=5.0),
    )
    monkeypatch.setattr(
        intraday_data,
        "write_day_partition",
        lambda frame, statuses, candidate_day, root: written.append(
            (frame.copy(), statuses.copy())
        ),
    )

    coverage = intraday_data.fetch_minute_partitions(
        {
            "candidates": ["000001.SZ"],
            "fetch_dates": pd.DatetimeIndex([day]),
        },
        pd.DataFrame({
            "code": [1.0],
            "date": ["2026-01-12 12:34:56"],
            "amount": [daily_amount],
        }),
        tmp_path,
        "token",
    )

    assert written[0][0].empty
    assert written[0][1] == {"000001": "returned"}
    assert coverage[["code", "reason"]].to_dict("records") == [
        {"code": "000001", "reason": "amount_mismatch"}
    ]


@pytest.mark.parametrize("missing", ["thscode", "time", "open", "close"])
def test_fetch_adjusted_daily_validates_response_schema(monkeypatch, missing):
    response = pd.DataFrame({
        "thscode": ["000001.SZ"],
        "time": ["2026-01-12"],
        "open": [10.0],
        "close": [11.0],
    }).drop(columns=missing)
    monkeypatch.setattr(
        intraday_data.ths_http,
        "history_quotation",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(
        ValueError,
        match=rf"adjusted history missing required columns: {missing}",
    ):
        intraday_data.fetch_adjusted_daily(
            ["000001"], "2026-01-01", "2026-01-31", "token"
        )


def test_fetch_adjusted_daily_filters_invalid_and_unrequested_rows(monkeypatch):
    thscodes = [
        "000001.SZ",
        "000001.SZ",
        "000002.SZ",
        "foo000001.SZ",
        "000001.SH",
        *(["000001.SZ"] * 9),
    ]
    times = ["2026-01-12"] * len(thscodes)
    times[5] = "not-a-date"
    opens = [
        10.0, 99.0, 20.0, 10.0, 10.0, 10.0, np.nan,
        np.inf, 0.0, -1.0, 10.0, 10.0, 10.0, 10.0,
    ]
    closes = [11.0] * len(thscodes)
    closes[10:] = [np.nan, np.inf, 0.0, -1.0]
    response = pd.DataFrame({
        "thscode": thscodes,
        "time": times,
        "open": opens,
        "close": closes,
    })
    monkeypatch.setattr(
        intraday_data.ths_http,
        "history_quotation",
        lambda *args, **kwargs: response,
    )

    result = intraday_data.fetch_adjusted_daily(
        [1.0], "2026-01-01", "2026-01-31", "token"
    )

    assert result.to_dict("records") == [{
        "code": "000001",
        "date": pd.Timestamp("2026-01-12"),
        "open": 10.0,
        "close": 11.0,
    }]


def test_fetch_adjusted_daily_rejects_response_with_no_valid_rows(monkeypatch):
    monkeypatch.setattr(
        intraday_data.ths_http,
        "history_quotation",
        lambda *args, **kwargs: pd.DataFrame({
            "thscode": ["000001.SZ", "000002.SZ"],
            "time": ["not-a-date", "2026-01-12"],
            "open": [10.0, 20.0],
            "close": [11.0, 21.0],
        }),
    )

    with pytest.raises(RuntimeError, match="adjusted history returned no rows"):
        intraday_data.fetch_adjusted_daily(
            ["000001.SZ"], "2026-01-01", "2026-01-31", "token"
        )


@pytest.mark.parametrize("missing", ["股票代码", "股票简称", "所属同花顺行业"])
def test_normalize_attributes_validates_required_schema(missing):
    raw = pd.DataFrame({
        "股票代码": ["000001.SZ"],
        "股票简称": ["平安银行"],
        "A股市值(不含限售股)[20260112]": [1.2e11],
        "所属同花顺行业": ["银行"],
    }).drop(columns=missing)

    with pytest.raises(
        ValueError,
        match=rf"attributes missing required columns: {missing}",
    ):
        intraday_data.normalize_attributes(raw, "2026-01-12")


def test_normalize_attributes_rejects_multiple_exact_float_cap_columns():
    raw = pd.DataFrame({
        "股票代码": ["000001.SZ"],
        "股票简称": ["平安银行"],
        "A股市值(不含限售股)[20260112]": [1.2e11],
        "a股市值(不含限售股)[20260112]": [1.1e11],
        "所属同花顺行业": ["银行"],
    })

    with pytest.raises(
        ValueError,
        match="expected exactly one dated A-share float cap for 20260112; found 2",
    ):
        intraday_data.normalize_attributes(raw, "2026-01-12")


def test_normalize_attributes_preserves_missing_name_and_industry():
    raw = pd.DataFrame({
        "股票代码": [1.0],
        "股票简称": [None],
        "A股市值(不含限售股)[20260112]": [1.2e11],
        "所属同花顺行业": [pd.NA],
    })

    result = intraday_data.normalize_attributes(raw, "2026-01-12")

    assert result.loc[0, "code"] == "000001"
    assert pd.isna(result.loc[0, "name"])
    assert pd.isna(result.loc[0, "industry"])


def test_normalize_attributes_rejects_wrong_market_suffix():
    raw = pd.DataFrame({
        "股票代码": ["000001.SH"],
        "股票简称": ["错误市场"],
        "A股市值(不含限售股)[20260112]": [1.2e11],
        "所属同花顺行业": ["银行"],
    })

    with pytest.raises(ValueError, match="attributes contains invalid stock code"):
        intraday_data.normalize_attributes(raw, "2026-01-12")


def test_apply_attribute_filters_drops_unknown_name():
    day = pd.Timestamp("2026-01-12")
    pool = pd.DataFrame({"date": [day], "code": ["000001"]})
    attributes = pd.DataFrame({
        "date": [day],
        "code": ["000001"],
        "name": [pd.NA],
        "float_cap": [1e11],
        "industry": ["银行"],
    })

    result = intraday_data.apply_attribute_filters(pool, attributes, [day])

    assert result.empty


def test_apply_attribute_filters_sorts_eval_dates_and_normalizes_codes():
    dates = pd.bdate_range("2026-01-12", periods=6)
    pool = pd.DataFrame({
        "date": [dates[4], dates[5]],
        "code": ["000001.SZ", "000001.SZ"],
    })
    attributes = pd.DataFrame({
        "date": [str(dates[0].date())],
        "code": [1.0],
        "name": ["平安银行"],
        "float_cap": [1e11],
        "industry": ["银行"],
    })
    eval_dates = [
        dates[5], dates[1], dates[0], dates[4], dates[2], dates[3], dates[0]
    ]

    result = intraday_data.apply_attribute_filters(pool, attributes, eval_dates)

    assert result[["date", "code"]].to_dict("records") == [{
        "date": dates[4],
        "code": "000001",
    }]
