import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from intraday import data as intraday_data
from intraday.report import write_outputs
from intraday.run import (
    _load_plan,
    build_parser,
    run_fetch,
    run_prepare,
    run_validate,
)


EXPECTED_OUTPUTS = {
    "factor_summary.csv",
    "daily_ic.csv",
    "quantile_returns.csv",
    "portfolio_nav.csv",
    "trades.csv",
    "data_coverage.csv",
    "report.md",
    "factor_ic.png",
    "factor_quantiles.png",
    "portfolio_nav.png",
}


def _tiny_results(quantile_returns=None):
    if quantile_returns is None:
        quantile_returns = pd.DataFrame(
            {
                "factor": ["score", "score"],
                "date": ["2026-01-12", "2026-01-12"],
                "group": [0, 4],
                "return": [0.0, 0.01],
            }
        )
    return {
        "factor_summary": pd.DataFrame(
            {"factor": ["score"], "ic_mean": [0.03]}
        ),
        "daily_ic": pd.DataFrame(
            {"date": ["2026-01-12"], "score": [0.03]}
        ),
        "quantile_returns": quantile_returns,
        "portfolio_nav": pd.DataFrame(
            {"strategy": [1.0], "benchmark": [1.0]}
        ),
        "trades": pd.DataFrame(
            {"date": ["2026-01-13"], "code": ["000001"]}
        ),
        "data_coverage": pd.DataFrame(
            {"date": ["2026-01-12"], "valid": [500]}
        ),
        "portfolio_metrics": {"annual_excess": 0.01, "excess_nw_t": 0.5},
        "disclosures": [
            "固定验证区间 2026-01-12 至 2026-07-10",
            "API 实际区间 2025-12-11 至 2026-07-10",
            "预热起点 2025-12-11",
            "ST 状态最多滞后 4 个交易日",
            "行业列可能不是严格时点数据",
            "六个月初步证据；单边成本 20 bp；剔除统计已列示",
            "综合 RankIC >= 0.03：达到",
        ],
    }


def test_write_outputs_creates_contract_files_and_readable_csvs(tmp_path):
    paths = write_outputs(_tiny_results(), tmp_path)

    assert {path.name for path in paths} == EXPECTED_OUTPUTS
    assert matplotlib.get_backend().lower() == "agg"
    for filename in sorted(EXPECTED_OUTPUTS):
        assert (tmp_path / filename).stat().st_size > 0
    for filename in EXPECTED_OUTPUTS:
        if filename.endswith(".csv"):
            pd.read_csv(tmp_path / filename)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for disclosure in _tiny_results()["disclosures"]:
        assert disclosure in report


def test_write_outputs_handles_empty_frames(tmp_path):
    results = _tiny_results(pd.DataFrame())
    results["daily_ic"] = pd.DataFrame()
    results["portfolio_nav"] = pd.DataFrame()

    paths = write_outputs(results, tmp_path)

    assert {path.name for path in paths} == EXPECTED_OUTPUTS


def test_parser_defaults_are_pinned():
    args = build_parser().parse_args(["prepare"])

    assert args.start == "2026-01-12"
    assert args.end == "2026-07-10"
    assert args.warmup == "2025-12-11"
    assert args.top == 500
    assert args.top_n == 50
    assert args.rebalance == 5
    assert args.cost_bps == 20.0
    assert args.min_count == 400
    assert args.daily_cache == Path("alpha101/cache/ths_panel.pkl")
    assert args.cache == Path("intraday/cache")
    assert args.output == Path("output/intraday_6m")


def _daily_frame(dates, codes=("000001", "000002")):
    return pd.DataFrame(
        [
            {
                "date": day,
                "code": code,
                "open": 10.0 + code_index,
                "high": 10.2 + code_index,
                "low": 9.8 + code_index,
                "close": 10.1 + code_index,
                "volume": 1000.0,
                "amount": float(1000 - 100 * code_index + day_index),
            }
            for day_index, day in enumerate(dates)
            for code_index, code in enumerate(codes)
        ]
    )


def test_prepare_writes_deterministic_atomic_validated_plan(
    tmp_path,
    monkeypatch,
):
    dates = pd.bdate_range("2025-12-01", periods=35)
    raw = _daily_frame(dates)
    args = build_parser().parse_args(
        [
            "prepare",
            "--start",
            dates[25].strftime("%Y-%m-%d"),
            "--end",
            dates[-1].strftime("%Y-%m-%d"),
            "--warmup",
            dates[0].strftime("%Y-%m-%d"),
            "--top",
            "2",
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    monkeypatch.setattr("intraday.data.load_daily_raw", lambda _: raw)

    first = run_prepare(args)
    first_json = (args.cache / "plan.json").read_bytes()
    second = run_prepare(args)

    assert first_json == (args.cache / "plan.json").read_bytes()
    assert not list(args.cache.glob("*.tmp"))
    assert first["candidates"] == second["candidates"] == ["000001", "000002"]
    loaded = _load_plan(args.cache)
    assert loaded["warmup"] == pd.Timestamp(dates[0])
    assert loaded["eval_dates"].equals(dates[25:])
    assert loaded["estimated_cells"] == loaded["estimated_rows"] * 3
    assert loaded["ranked_pool"].groupby("date").size().le(2).all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(
                estimated_cells=payload["estimated_rows"] * 3 + 1
            ),
            "estimated_cells",
        ),
        (
            lambda payload: payload.update(
                candidates=payload["candidates"] + [payload["candidates"][0]]
            ),
            "candidates",
        ),
        (
            lambda payload: payload.update(eval_dates=["not-a-date"]),
            "eval_dates",
        ),
    ],
)
def test_load_plan_rejects_corrupt_payload(tmp_path, monkeypatch, mutate, message):
    dates = pd.bdate_range("2025-12-01", periods=25)
    raw = _daily_frame(dates)
    args = build_parser().parse_args(
        [
            "prepare",
            "--start",
            dates[20].strftime("%Y-%m-%d"),
            "--end",
            dates[-1].strftime("%Y-%m-%d"),
            "--warmup",
            dates[0].strftime("%Y-%m-%d"),
            "--cache",
            str(tmp_path),
            "--top",
            "2",
        ]
    )
    monkeypatch.setattr("intraday.data.load_daily_raw", lambda _: raw)
    run_prepare(args)
    plan_path = tmp_path / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    mutate(payload)
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_plan(tmp_path)


def test_load_plan_rejects_ranked_pool_over_top_quota(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-12-01", periods=25)
    raw = _daily_frame(dates)
    args = build_parser().parse_args(
        [
            "prepare",
            "--start",
            dates[20].strftime("%Y-%m-%d"),
            "--end",
            dates[-1].strftime("%Y-%m-%d"),
            "--warmup",
            dates[0].strftime("%Y-%m-%d"),
            "--cache",
            str(tmp_path),
            "--top",
            "2",
        ]
    )
    monkeypatch.setattr("intraday.data.load_daily_raw", lambda _: raw)
    run_prepare(args)
    plan_path = tmp_path / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["candidates"].append("000003")
    payload["estimated_rows"] = (
        len(payload["candidates"]) * len(payload["fetch_dates"]) * 241
    )
    payload["estimated_cells"] = payload["estimated_rows"] * 3
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    ranked_path = tmp_path / "ranked_pool.parquet"
    ranked = pd.read_parquet(ranked_path)
    extra = ranked.iloc[[0]].assign(code="000003", liquidity_rank=3)
    pd.concat([ranked, extra], ignore_index=True).to_parquet(
        ranked_path,
        index=False,
    )

    with pytest.raises(ValueError, match="top quota"):
        _load_plan(tmp_path)


def _prepared_fetch_case(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-12-01", periods=25)
    raw = _daily_frame(dates)
    args = build_parser().parse_args(
        [
            "fetch",
            "--start",
            dates[20].strftime("%Y-%m-%d"),
            "--end",
            dates[-1].strftime("%Y-%m-%d"),
            "--warmup",
            dates[0].strftime("%Y-%m-%d"),
            "--cache",
            str(tmp_path / "cache"),
            "--rebalance",
            "2",
            "--top",
            "2",
        ]
    )
    monkeypatch.setattr("intraday.data.load_daily_raw", lambda _: raw)
    run_prepare(args)
    return args, raw, dates


def _empty_minute_frame():
    return pd.DataFrame(
        {
            name: pd.Series(dtype=dtype)
            for name, dtype in intraday_data.MINUTE_DTYPES.items()
        }
    )


def test_fetch_uses_one_token_and_atomically_resumes_complete_caches(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    token_calls = []
    calls = {"attributes": 0, "adjusted": 0, "minute": 0}

    def get_token():
        token_calls.append("called")
        return "one-token"

    def fetch_attributes(anchors, token):
        calls["attributes"] += 1
        assert token == "one-token"
        return pd.DataFrame(
            [
                {
                    "date": day,
                    "code": code,
                    "name": f"N{code}",
                    "float_cap": 1_000_000.0,
                    "industry": "I",
                }
                for day in anchors
                for code in plan["candidates"]
            ]
        )

    def fetch_adjusted(codes, start, end, token):
        calls["adjusted"] += 1
        assert token == "one-token"
        assert pd.Timestamp(start) == plan["warmup"]
        assert pd.Timestamp(end) == plan["end"]
        return pd.DataFrame(
            [
                {"date": day, "code": code, "open": 10.0, "close": 10.0}
                for day in dates
                for code in codes
            ]
        )

    def fetch_minutes(fetch_plan, raw, root, token):
        calls["minute"] += 1
        assert token == "one-token"
        for day in fetch_plan["fetch_dates"]:
            if not intraday_data.day_complete(day, fetch_plan["candidates"], root):
                intraday_data.write_day_partition(
                    _empty_minute_frame(),
                    {code: "no_data" for code in fetch_plan["candidates"]},
                    day,
                    root,
                )
        if calls["minute"] == 2:
            return pd.DataFrame(
                [
                    {
                        "date": fetch_plan["fetch_dates"][0],
                        "code": fetch_plan["candidates"][0],
                        "minute_count": 0,
                        "amount_relative_error": 0.123,
                        "reason": "no_data",
                    }
                ]
            )
        return pd.DataFrame(columns=intraday_data.COVERAGE_COLUMNS)

    monkeypatch.setattr("alpha101.ths_http.get_access_token", get_token)
    monkeypatch.setattr("intraday.data.fetch_attributes", fetch_attributes)
    monkeypatch.setattr("intraday.data.fetch_adjusted_daily", fetch_adjusted)
    monkeypatch.setattr("intraday.data.fetch_minute_partitions", fetch_minutes)

    coverage = run_fetch(args)

    assert token_calls == ["called"]
    assert calls == {"attributes": 1, "adjusted": 1, "minute": 1}
    assert not list(args.cache.rglob("*.tmp"))
    assert len(coverage) == len(plan["fetch_dates"]) * len(plan["candidates"])
    assert not coverage.duplicated(["date", "code"]).any()
    assert coverage["reason"].eq("no_data").all()

    token_calls.clear()
    resumed = run_fetch(args)
    assert token_calls == ["called"]
    assert calls == {"attributes": 1, "adjusted": 1, "minute": 2}
    updated = resumed.loc[
        resumed["date"].eq(plan["fetch_dates"][0])
        & resumed["code"].eq(plan["candidates"][0]),
        "amount_relative_error",
    ]
    assert updated.iloc[0] == pytest.approx(0.123)


def test_fetch_does_not_accept_corrupt_complete_minute_partition(
    tmp_path,
    monkeypatch,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    monkeypatch.setattr("alpha101.ths_http.get_access_token", lambda: "token")
    monkeypatch.setattr(
        "intraday.data.fetch_attributes",
        lambda anchors, token: pd.DataFrame(
            [
                {
                    "date": day,
                    "code": code,
                    "name": code,
                    "float_cap": 1.0,
                    "industry": "I",
                }
                for day in anchors
                for code in plan["candidates"]
            ]
        ),
    )
    monkeypatch.setattr(
        "intraday.data.fetch_adjusted_daily",
        lambda codes, start, end, token: pd.DataFrame(
            [
                {"date": day, "code": code, "open": 10.0, "close": 10.0}
                for day in plan["fetch_dates"]
                for code in codes
            ]
        ),
    )

    def fetch_minutes(fetch_plan, raw, root, token):
        for day in fetch_plan["fetch_dates"]:
            intraday_data.write_day_partition(
                _empty_minute_frame(),
                {code: "no_data" for code in fetch_plan["candidates"]},
                day,
                root,
            )
        first, _ = intraday_data._day_paths(fetch_plan["fetch_dates"][0], root)
        first.write_bytes(b"corrupt parquet")
        return pd.DataFrame(columns=intraday_data.COVERAGE_COLUMNS)

    monkeypatch.setattr("intraday.data.fetch_minute_partitions", fetch_minutes)

    with pytest.raises(ValueError, match="minute partition"):
        run_fetch(args)


def test_validate_missing_cache_fails_without_mutating_prepared_cache(
    tmp_path,
    monkeypatch,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    before = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileNotFoundError, match="validation cache"):
        run_validate(args)

    after = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }
    assert after == before


def _minute_times(day):
    day = pd.Timestamp(day).normalize()
    return pd.date_range(
        day + pd.Timedelta(hours=9, minutes=31),
        periods=100,
        freq="min",
    ).append(
        pd.date_range(
            day + pd.Timedelta(hours=13, minutes=1),
            periods=100,
            freq="min",
        )
    )


def _synthetic_minute_partition(day, day_index, codes):
    rows = []
    minute = pd.RangeIndex(200).to_numpy(dtype=float)
    times = _minute_times(day)
    for code_index, code in enumerate(codes):
        phase = day_index * 0.19 + code_index * 0.37
        returns = (
            0.0007 * np.sin(minute * (0.09 + code_index * 0.006) + phase)
            + 0.00015 * np.cos(minute * 0.031 - phase)
        )
        returns[(minute.astype(int) + code_index) % (17 + code_index) == 0] *= (
            2.0 + 0.2 * code_index
        )
        close = (10.0 + code_index) * np.exp(np.cumsum(returns))
        volume = 80.0 + (
            (minute * (code_index + 3) + day_index * 7) % 53
        )
        amount = close * volume
        rows.extend(
            {
                "code": code,
                "time": timestamp,
                "close": price,
                "volume": shares,
                "amount": value,
            }
            for timestamp, price, shares, value in zip(
                times,
                close,
                volume,
                amount,
            )
        )
    return pd.DataFrame(rows)


def _write_offline_validation_case(tmp_path):
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    dates = pd.bdate_range("2026-01-12", periods=25)
    codes = [f"{code:06d}" for code in range(1, 7)]
    ranked = pd.DataFrame(
        [
            {
                "date": day,
                "code": code,
                "adv20": float(10_000_000 - code_index * 1000 + day_index),
                "liquidity_rank": code_index + 1,
            }
            for day_index, day in enumerate(dates)
            for code_index, code in enumerate(codes)
        ]
    )
    eligible = ranked.copy()
    ranked.to_parquet(cache / "ranked_pool.parquet", index=False)
    eligible.to_parquet(cache / "eligible_pool.parquet", index=False)
    estimated_rows = len(dates) * len(codes) * 241
    payload = {
        "schema_version": 1,
        "start": dates[0].strftime("%Y-%m-%d"),
        "end": dates[-1].strftime("%Y-%m-%d"),
        "warmup": dates[0].strftime("%Y-%m-%d"),
        "top": 6,
        "eval_dates": [day.strftime("%Y-%m-%d") for day in dates],
        "fetch_dates": [day.strftime("%Y-%m-%d") for day in dates],
        "candidates": codes,
        "estimated_rows": estimated_rows,
        "estimated_cells": estimated_rows * 3,
        "parameters": {
            "min_count": 6,
            "top_n": 2,
            "rebalance": 5,
            "cost_bps": 20.0,
        },
    }
    (cache / "plan.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    attributes = pd.DataFrame(
        [
            {
                "date": day,
                "code": code,
                "name": f"Company {code}",
                "float_cap": float(1_000_000 * (code_index + 1) ** 1.3),
                "industry": "Synthetic",
            }
            for day in dates[::5]
            for code_index, code in enumerate(codes)
        ]
    )
    attributes.to_parquet(cache / "attributes.parquet", index=False)

    daily_rows = []
    adjusted_rows = []
    coverage_rows = []
    for day_index, day in enumerate(dates):
        minute = _synthetic_minute_partition(day, day_index, codes)
        intraday_data.write_day_partition(
            minute,
            {code: "returned" for code in codes},
            day,
            cache,
        )
        for code_index, code in enumerate(codes):
            open_price = (
                10.0
                + code_index * 0.25
                + day_index * 0.03
                + 0.04 * np.sin((day_index + code_index) / 3)
            )
            close_price = open_price * (
                1 + 0.002 * np.sin(day_index * 0.4 + code_index)
            )
            daily_rows.append(
                {
                    "date": day,
                    "code": code,
                    "open": open_price,
                    "high": max(open_price, close_price) * 1.01,
                    "low": min(open_price, close_price) * 0.99,
                    "close": close_price,
                    "volume": 100_000.0,
                    "amount": open_price * 100_000.0,
                }
            )
            adjusted_rows.append(
                {
                    "date": day,
                    "code": code,
                    "open": open_price,
                    "close": close_price,
                }
            )
            coverage_rows.append(
                {
                    "date": day,
                    "code": code,
                    "minute_count": 200,
                    "amount_relative_error": 0.0,
                    "reason": "ok",
                }
            )
    raw = pd.DataFrame(daily_rows)
    daily_path = tmp_path / "daily.pkl"
    raw.to_pickle(daily_path)
    pd.DataFrame(adjusted_rows).to_parquet(
        cache / "adjusted_daily.parquet",
        index=False,
    )
    pd.DataFrame(coverage_rows).to_parquet(
        cache / "data_coverage.parquet",
        index=False,
    )
    args = build_parser().parse_args(
        [
            "validate",
            "--start",
            dates[0].strftime("%Y-%m-%d"),
            "--end",
            dates[-1].strftime("%Y-%m-%d"),
            "--warmup",
            dates[0].strftime("%Y-%m-%d"),
            "--daily-cache",
            str(daily_path),
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--top",
            "6",
            "--top-n",
            "2",
            "--rebalance",
            "5",
            "--min-count",
            "6",
            "--cost-bps",
            "20",
        ]
    )
    return args, dates, codes


def test_validate_offline_end_to_end_preserves_cache_and_t_plus_one_cost(
    tmp_path,
    monkeypatch,
):
    args, dates, _ = _write_offline_validation_case(tmp_path)
    monkeypatch.setattr(
        "alpha101.ths_http.get_access_token",
        lambda: pytest.fail("validate must not access the network"),
    )
    before = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }

    paths = run_validate(args)

    after = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert {path.name for path in paths} == EXPECTED_OUTPUTS
    tables = {
        filename: pd.read_csv(args.output / filename)
        for filename in EXPECTED_OUTPUTS
        if filename.endswith(".csv")
    }
    trades = tables["trades.csv"]
    strategy = trades.loc[trades["portfolio"].eq("strategy")]
    assert not strategy.empty
    first_signal = pd.Timestamp(strategy["signal_date"].min())
    first_trade = pd.Timestamp(strategy["date"].min())
    assert first_trade == dates[dates.get_loc(first_signal) + 1]
    assert not pd.to_datetime(strategy["date"]).eq(
        pd.to_datetime(strategy["signal_date"])
    ).any()
    assert np.allclose(strategy["cost"], strategy["notional"] * 0.002)
    nav = tables["portfolio_nav.csv"]
    assert (nav["strategy_net"] < nav["strategy_gross"]).any()
    report_text = (args.output / "report.md").read_text(encoding="utf-8")
    for phrase in [
        "固定验证区间",
        "API 实际区间",
        "预热起点",
        "ST 状态最多滞后 4 个交易日",
        "行业列可能不是严格时点数据",
        "六个月初步证据",
        "单边实际成交成本 20.0 bp",
        "剔除统计",
        "达到" if "达到" in report_text else "未达到",
    ]:
        assert phrase in report_text


def test_validate_rejects_each_corrupt_cache_layer_read_only(tmp_path):
    args, dates, _ = _write_offline_validation_case(tmp_path)
    _, manifest = intraday_data._day_paths(dates[0], args.cache)
    cases = [
        (args.cache / "attributes.parquet", b"bad attributes", "attributes cache"),
        (
            args.cache / "adjusted_daily.parquet",
            b"bad adjusted",
            "adjusted daily cache",
        ),
        (
            args.cache / "data_coverage.parquet",
            b"bad coverage",
            "coverage cache",
        ),
        (manifest, b"{}", "minute manifest"),
    ]

    for path, corrupt, message in cases:
        original = path.read_bytes()
        path.write_bytes(corrupt)
        with pytest.raises(ValueError, match=message):
            run_validate(args)
        assert path.read_bytes() == corrupt
        path.write_bytes(original)
    assert not args.output.exists()
