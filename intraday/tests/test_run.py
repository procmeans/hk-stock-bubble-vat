import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import intraday.run as intraday_run
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
    assert args.batch_size == 200
    assert args.daily_cache == Path("alpha101/cache/ths_panel.pkl")
    assert args.cache == Path("intraday/cache")
    assert args.output == Path("output/intraday_6m")


def test_parser_accepts_smaller_operational_fetch_batch():
    args = build_parser().parse_args(["fetch", "--batch-size", "100"])

    assert args.batch_size == 100


def test_all_four_subcommands_share_identical_defaults():
    parsed = {
        command: vars(build_parser().parse_args([command]))
        for command in ("prepare", "fetch", "validate", "all")
    }
    common = {
        command: {key: value for key, value in values.items() if key != "command"}
        for command, values in parsed.items()
    }

    assert common["prepare"] == common["fetch"]
    assert common["prepare"] == common["validate"]
    assert common["prepare"] == common["all"]


def test_main_all_runs_strict_order_and_propagates_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        intraday_run,
        "run_prepare",
        lambda args: calls.append("prepare"),
    )
    monkeypatch.setattr(
        intraday_run,
        "run_fetch",
        lambda args: calls.append("fetch"),
    )
    monkeypatch.setattr(
        intraday_run,
        "run_validate",
        lambda args: calls.append("validate"),
    )

    intraday_run.main(["all"])

    assert calls == ["prepare", "fetch", "validate"]

    calls.clear()

    def fail_fetch(args):
        calls.append("fetch")
        raise RuntimeError("fetch stopped")

    monkeypatch.setattr(intraday_run, "run_fetch", fail_fetch)
    with pytest.raises(RuntimeError, match="fetch stopped"):
        intraday_run.main(["all"])
    assert calls == ["prepare", "fetch"]


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
    assert loaded["eval_dates"][[0, -1]].tolist() == [dates[25], dates[-1]]
    assert loaded["fetch_dates"][[0, -1]].tolist() == [dates[0], dates[-1]]
    assert loaded["estimated_cells"] == loaded["estimated_rows"] * 3
    assert loaded["ranked_pool"].groupby("date").size().le(2).all()
    assert loaded["schema_version"] == 2
    assert (args.cache / "pool_audit.parquet").is_file()
    assert loaded["pool_audit"]["date"].tolist() == dates[25:].tolist()
    assert (
        loaded["pool_audit"]["ranked_count"]
        == loaded["pool_audit"]
        [["age_exclusions", "suspension_exclusions", "daily_eligible_count"]]
        .sum(axis=1)
    ).all()


@pytest.mark.parametrize(
    "corruption",
    ["old_schema", "missing_column", "missing_date", "count_mismatch"],
)
def test_load_plan_requires_strict_versioned_pool_audit(
    tmp_path,
    monkeypatch,
    corruption,
):
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
    audit_path = tmp_path / "pool_audit.parquet"
    if corruption == "old_schema":
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        audit = pd.read_parquet(audit_path)
        if corruption == "missing_column":
            audit = audit.drop(columns="age_exclusions")
        elif corruption == "missing_date":
            audit = audit.iloc[1:]
        else:
            audit.loc[audit.index[0], "daily_eligible_count"] += 1
        audit.to_parquet(audit_path, index=False)

    with pytest.raises(ValueError, match="(schema_version|pool_audit)"):
        _load_plan(tmp_path)


def test_prepare_fixed_default_date_boundaries_are_valid(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-12-11", "2026-07-10")
    raw = _daily_frame(dates)
    args = build_parser().parse_args(
        ["prepare", "--cache", str(tmp_path), "--top", "2"]
    )
    monkeypatch.setattr("intraday.data.load_daily_raw", lambda _: raw)

    plan = run_prepare(args)

    assert plan["eval_dates"][[0, -1]].tolist() == [
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-07-10"),
    ]
    assert plan["fetch_dates"][[0, -1]].tolist() == [
        pd.Timestamp("2025-12-11"),
        pd.Timestamp("2026-07-10"),
    ]


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


@pytest.mark.parametrize(
    "mutation",
    ["drop_eval_start", "drop_fetch_warmup", "prepend_history", "append_future"],
)
def test_load_plan_requires_exact_declared_date_boundaries(
    tmp_path,
    monkeypatch,
    mutation,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    plan_path = args.cache / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if mutation == "drop_eval_start":
        payload["eval_dates"] = payload["eval_dates"][1:]
    elif mutation == "drop_fetch_warmup":
        payload["fetch_dates"] = payload["fetch_dates"][1:]
    elif mutation == "prepend_history":
        prior = pd.Timestamp(payload["warmup"]) - pd.Timedelta(days=1)
        payload["fetch_dates"].insert(0, prior.strftime("%Y-%m-%d"))
    else:
        future = pd.Timestamp(payload["end"]) + pd.Timedelta(days=1)
        payload["fetch_dates"].append(future.strftime("%Y-%m-%d"))
    payload["estimated_rows"] = (
        len(payload["candidates"]) * len(payload["fetch_dates"]) * 241
    )
    payload["estimated_cells"] = payload["estimated_rows"] * 3
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dates"):
        _load_plan(args.cache)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "min_count",
        "top_n",
        "rebalance",
        "cost_bps",
    ],
)
def test_load_plan_requires_valid_pinned_parameters(
    tmp_path,
    monkeypatch,
    mutation,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    plan_path = args.cache / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["parameters"]
    elif mutation == "cost_bps":
        payload["parameters"][mutation] = float("nan")
    else:
        payload["parameters"][mutation] = 0
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="parameters"):
        _load_plan(args.cache)


@pytest.mark.parametrize("runner", [run_fetch, run_validate])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("start", "2025-12-30"),
        ("end", "2026-01-01"),
        ("warmup", "2025-12-02"),
        ("top", 3),
        ("min_count", 399),
        ("top_n", 49),
        ("rebalance", 3),
        ("cost_bps", 10.0),
    ],
)
def test_fetch_and_validate_reject_every_cli_plan_parameter_drift(
    tmp_path,
    monkeypatch,
    runner,
    name,
    value,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    setattr(args, name, value)
    monkeypatch.setattr(
        "alpha101.ths_http.get_access_token",
        lambda: pytest.fail("parameter drift must fail before token acquisition"),
    )

    with pytest.raises(ValueError, match=rf"CLI {name}"):
        runner(args)


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
    assert "batch_size" not in _load_plan(args.cache)["parameters"]
    return args, raw, dates


def _empty_minute_frame():
    return pd.DataFrame(
        {
            name: pd.Series(dtype=dtype)
            for name, dtype in intraday_data.MINUTE_DTYPES.items()
        }
    )


def test_attribute_cache_preserves_nullable_values_for_pool_filtering():
    anchor = pd.Timestamp("2026-01-12")
    anchors = pd.DatetimeIndex([anchor])
    candidates = ["000001", "000002", "000003", "000004"]
    attributes = pd.DataFrame(
        [
            {
                "date": anchor,
                "code": "000001",
                "name": "Company One",
                "float_cap": 1_000_000.0,
                "industry": "Industry A",
            },
            {
                "date": anchor,
                "code": "000002",
                "name": None,
                "float_cap": "unavailable",
                "industry": None,
            },
            {
                "date": anchor,
                "code": "000003",
                "name": "Company Three",
                "float_cap": 0.0,
                "industry": None,
            },
            {
                "date": anchor,
                "code": "000004",
                "name": None,
                "float_cap": 4_000_000.0,
                "industry": "Industry B",
            },
        ]
    )
    pool = pd.DataFrame(
        {"date": [anchor] * len(candidates), "code": candidates}
    )

    cached = intraday_run._validate_attributes(
        attributes,
        anchors,
        candidates,
    )
    filtered = intraday_data.apply_attribute_filters(pool, cached, anchors)

    assert cached[["date", "code"]].to_dict("records") == [
        {"date": anchor, "code": code} for code in candidates
    ]
    assert pd.isna(cached.loc[cached["code"].eq("000002"), "float_cap"]).all()
    assert cached.loc[cached["code"].eq("000003"), "float_cap"].eq(0).all()
    assert cached[["name", "industry"]].isna().any().all()
    assert filtered["code"].tolist() == ["000001"]


@pytest.mark.parametrize(
    "corruption",
    ["missing_key", "duplicate_key", "bad_code", "bad_date", "missing_schema"],
)
def test_attribute_cache_still_rejects_structural_corruption(corruption):
    anchor = pd.Timestamp("2026-01-12")
    anchors = pd.DatetimeIndex([anchor])
    candidates = ["000001", "000002"]
    attributes = pd.DataFrame(
        {
            "date": [anchor, anchor],
            "code": candidates,
            "name": ["One", "Two"],
            "float_cap": [1.0, np.nan],
            "industry": ["A", None],
        }
    )
    if corruption == "missing_key":
        attributes = attributes.iloc[:1]
    elif corruption == "duplicate_key":
        attributes.loc[1, "code"] = "000001"
    elif corruption == "bad_code":
        attributes.loc[1, "code"] = "bad"
    elif corruption == "bad_date":
        attributes["date"] = attributes["date"].astype(object)
        attributes.loc[1, "date"] = "bad"
    else:
        attributes = attributes.drop(columns="industry")

    with pytest.raises(ValueError, match="attributes cache"):
        intraday_run._validate_attributes(attributes, anchors, candidates)


def _fake_attribute_result(anchors, candidates):
    frame = pd.DataFrame(
        [
            {
                "date": day,
                "code": code,
                "name": f"Company {code}",
                "float_cap": 1_000_000.0,
                "industry": "I",
            }
            for day in anchors
            for code in candidates
        ]
    )
    metadata = [
        {
            "date": day.strftime("%Y-%m-%d"),
            "query": intraday_data.build_attribute_query(day),
            "columns": [
                "股票代码",
                "股票简称",
                f"A股市值(不含限售股)[{day:%Y%m%d}]",
                "所属同花顺行业",
            ],
            "row_count": len(candidates),
        }
        for day in anchors
    ]
    return frame, metadata


def _install_no_data_fetch_fakes(
    monkeypatch,
    plan,
    dates,
    adjusted_calls,
    attribute_calls=None,
):
    monkeypatch.setattr("alpha101.ths_http.get_access_token", lambda: "token")

    def fetch_attributes(anchors, token, *, return_metadata=False):
        if attribute_calls is not None:
            attribute_calls.append("called")
        frame, metadata = _fake_attribute_result(anchors, plan["candidates"])
        return (frame, metadata) if return_metadata else frame

    monkeypatch.setattr("intraday.data.fetch_attributes", fetch_attributes)

    def fetch_adjusted(codes, start, end, token):
        adjusted_calls.append("called")
        return pd.DataFrame(
            [
                {
                    "date": day,
                    "code": code,
                    "open": 10.0 + code_index,
                    "close": 10.0 + code_index,
                }
                for day in dates
                for code_index, code in enumerate(codes)
                if not (day == dates[1] and code_index == 0)
            ]
        )

    def fetch_minutes(fetch_plan, raw, root, token, batch_size):
        assert batch_size == 200
        for day in fetch_plan["fetch_dates"]:
            if not intraday_data.day_complete(day, fetch_plan["candidates"], root):
                intraday_data.write_day_partition(
                    _empty_minute_frame(),
                    {code: "no_data" for code in fetch_plan["candidates"]},
                    day,
                    root,
                )
        return pd.DataFrame(columns=intraday_data.COVERAGE_COLUMNS)

    monkeypatch.setattr("intraday.data.fetch_adjusted_daily", fetch_adjusted)
    monkeypatch.setattr("intraday.data.fetch_minute_partitions", fetch_minutes)


def test_fetch_reuses_attributes_only_with_matching_completion_manifest(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    attribute_calls = []
    adjusted_calls = []
    _install_no_data_fetch_fakes(
        monkeypatch,
        plan,
        dates,
        adjusted_calls,
        attribute_calls,
    )

    run_fetch(args)

    parquet_path = args.cache / "attributes.parquet"
    manifest_path = args.cache / "attributes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchors = plan["eval_dates"][:: args.rebalance]
    assert manifest["schema_version"] == 1
    assert manifest["request"] == {
        "anchors": [day.strftime("%Y-%m-%d") for day in anchors],
        "candidates": plan["candidates"],
    }
    assert manifest["metadata"] == _fake_attribute_result(
        anchors,
        plan["candidates"],
    )[1]
    assert manifest["frame"]["columns"] == [
        "date", "code", "name", "float_cap", "industry"
    ]
    assert manifest["frame"]["row_count"] == len(anchors) * len(
        plan["candidates"]
    )
    assert len(manifest["frame"]["sha256"]) == 64
    assert attribute_calls == ["called"]
    assert adjusted_calls == ["called"]

    immutable_paths = [
        args.cache / "adjusted_daily.parquet",
        args.cache / "adjusted_daily.json",
        args.cache / "data_coverage.parquet",
        *sorted((args.cache / "minute").iterdir()),
    ]
    immutable = {path: path.read_bytes() for path in immutable_paths}

    manifest_path.unlink()
    run_fetch(args)
    assert attribute_calls == ["called", "called"]
    assert adjusted_calls == ["called"]
    assert {path: path.read_bytes() for path in immutable_paths} == immutable

    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["metadata"][0]["query"] = "tampered query"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    run_fetch(args)
    assert attribute_calls == ["called", "called", "called"]
    assert adjusted_calls == ["called"]

    tampered = pd.read_parquet(parquet_path)
    tampered.loc[tampered.index[0], "float_cap"] += 1.0
    tampered.to_parquet(parquet_path, index=False)
    run_fetch(args)
    assert attribute_calls == ["called", "called", "called", "called"]
    assert adjusted_calls == ["called"]

    run_fetch(args)
    assert attribute_calls == ["called", "called", "called", "called"]
    assert adjusted_calls == ["called"]


def test_attributes_manifest_is_published_last_and_failure_is_not_reusable(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    attribute_calls = []
    adjusted_calls = []
    _install_no_data_fetch_fakes(
        monkeypatch,
        plan,
        dates,
        adjusted_calls,
        attribute_calls,
    )
    manifest_path = args.cache / "attributes.json"
    real_replace = Path.replace

    def fail_manifest_publish(path, target):
        if Path(target) == manifest_path:
            raise OSError("attributes manifest publish failure")
        return real_replace(path, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_manifest_publish)
        with pytest.raises(OSError, match="attributes manifest publish failure"):
            run_fetch(args)

    assert (args.cache / "attributes.parquet").is_file()
    assert not manifest_path.exists()
    assert not list(args.cache.rglob("*.tmp"))
    run_fetch(args)
    assert attribute_calls == ["called", "called"]
    assert adjusted_calls == ["called"]


def test_fetch_reuses_adjusted_only_with_matching_completion_manifest(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    adjusted_calls = []
    _install_no_data_fetch_fakes(
        monkeypatch,
        plan,
        dates,
        adjusted_calls,
    )

    run_fetch(args)

    parquet_path = args.cache / "adjusted_daily.parquet"
    manifest_path = args.cache / "adjusted_daily.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["request"] == {
        "start": plan["warmup"].strftime("%Y-%m-%d"),
        "end": plan["end"].strftime("%Y-%m-%d"),
        "candidates": plan["candidates"],
    }
    assert len(manifest["frame"]["sha256"]) == 64
    assert manifest["frame"]["row_count"] < len(dates) * len(plan["candidates"])
    assert adjusted_calls == ["called"]

    partial = pd.read_parquet(parquet_path).groupby("code").nth(0).reset_index()
    partial.to_parquet(parquet_path, index=False)
    manifest_path.unlink()
    run_fetch(args)
    assert adjusted_calls == ["called", "called"]

    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_manifest["schema_version"] = 0
    manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    run_fetch(args)
    assert adjusted_calls == ["called", "called", "called"]

    tampered = pd.read_parquet(parquet_path)
    tampered.loc[tampered.index[0], "open"] += 1.0
    tampered.to_parquet(parquet_path, index=False)
    run_fetch(args)
    assert adjusted_calls == ["called", "called", "called", "called"]

    run_fetch(args)
    assert adjusted_calls == ["called", "called", "called", "called"]


def test_adjusted_manifest_is_published_last_and_failure_is_not_reusable(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    adjusted_calls = []
    _install_no_data_fetch_fakes(
        monkeypatch,
        plan,
        dates,
        adjusted_calls,
    )
    manifest_path = args.cache / "adjusted_daily.json"
    real_replace = Path.replace

    def fail_manifest_publish(path, target):
        if Path(target) == manifest_path:
            raise OSError("manifest publish failure")
        return real_replace(path, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_manifest_publish)
        with pytest.raises(OSError, match="manifest publish failure"):
            run_fetch(args)

    assert (args.cache / "adjusted_daily.parquet").is_file()
    assert not manifest_path.exists()
    assert not list(args.cache.rglob("*.tmp"))
    run_fetch(args)
    assert adjusted_calls == ["called", "called"]


def test_fetch_uses_one_token_and_atomically_resumes_complete_caches(
    tmp_path,
    monkeypatch,
):
    args, _, dates = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    args.batch_size = 100
    token_calls = []
    calls = {"attributes": 0, "adjusted": 0, "minute": 0}
    batch_sizes = []

    def get_token():
        token_calls.append("called")
        return "one-token"

    def fetch_attributes(anchors, token, *, return_metadata=False):
        calls["attributes"] += 1
        assert token == "one-token"
        frame, metadata = _fake_attribute_result(anchors, plan["candidates"])
        return (frame, metadata) if return_metadata else frame

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

    def fetch_minutes(fetch_plan, raw, root, token, batch_size):
        calls["minute"] += 1
        batch_sizes.append(batch_size)
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
                        "amount_relative_error": np.nan,
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
    assert batch_sizes == [100]
    assert not list(args.cache.rglob("*.tmp"))
    assert len(coverage) == len(plan["fetch_dates"]) * len(plan["candidates"])
    assert not coverage.duplicated(["date", "code"]).any()
    assert coverage["reason"].eq("no_data").all()

    token_calls.clear()
    resumed = run_fetch(args)
    assert token_calls == ["called"]
    assert calls == {"attributes": 1, "adjusted": 1, "minute": 2}
    assert batch_sizes == [100, 100]
    updated = resumed.loc[
        resumed["date"].eq(plan["fetch_dates"][0])
        & resumed["code"].eq(plan["candidates"][0]),
        "amount_relative_error",
    ]
    assert pd.isna(updated.iloc[0])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_fetch_rejects_invalid_batch_size_before_token(
    tmp_path,
    monkeypatch,
    batch_size,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    args.batch_size = batch_size
    monkeypatch.setattr(
        "alpha101.ths_http.get_access_token",
        lambda: pytest.fail("invalid batch size must fail before token"),
    )

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        run_fetch(args)


def test_fetch_does_not_accept_corrupt_complete_minute_partition(
    tmp_path,
    monkeypatch,
):
    args, _, _ = _prepared_fetch_case(tmp_path, monkeypatch)
    plan = _load_plan(args.cache)
    monkeypatch.setattr("alpha101.ths_http.get_access_token", lambda: "token")
    def fetch_attributes(anchors, token, *, return_metadata=False):
        frame, metadata = _fake_attribute_result(anchors, plan["candidates"])
        return (frame, metadata) if return_metadata else frame

    monkeypatch.setattr("intraday.data.fetch_attributes", fetch_attributes)
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

    def fetch_minutes(fetch_plan, raw, root, token, batch_size):
        assert batch_size == 200
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
    pd.DataFrame(
        {
            "date": dates,
            "ranked_count": len(codes),
            "age_exclusions": 0,
            "suspension_exclusions": 0,
            "daily_eligible_count": len(codes),
        }
    ).to_parquet(cache / "pool_audit.parquet", index=False)
    estimated_rows = len(dates) * len(codes) * 241
    payload = {
        "schema_version": 2,
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
    attribute_anchors = dates[::5]
    attribute_metadata = _fake_attribute_result(attribute_anchors, codes)[1]
    attribute_manifest = intraday_run._attributes_manifest_payload(
        attributes,
        attribute_anchors,
        codes,
        attribute_metadata,
    )
    (cache / "attributes.json").write_text(
        json.dumps(attribute_manifest),
        encoding="utf-8",
    )

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
    adjusted_frame = pd.DataFrame(adjusted_rows)
    adjusted_frame.to_parquet(
        cache / "adjusted_daily.parquet",
        index=False,
    )
    adjusted_manifest = intraday_run._adjusted_manifest_payload(
        adjusted_frame,
        codes,
        dates[0],
        dates[-1],
    )
    (cache / "adjusted_daily.json").write_text(
        json.dumps(adjusted_manifest),
        encoding="utf-8",
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


def test_complete_coverage_rebuilds_from_authoritative_daily_sidecars(tmp_path):
    args, _, _ = _write_offline_validation_case(tmp_path)
    plan = _load_plan(args.cache)
    stale = pd.read_parquet(args.cache / "data_coverage.parquet")
    stale.loc[stale.index[0], ["minute_count", "reason"]] = [0, "no_data"]
    delta = stale.iloc[[0]].copy()

    rebuilt = intraday_run._complete_coverage(
        plan,
        args.cache,
        stale,
        delta,
    )

    authoritative = pd.concat(
        [
            intraday_data.read_day_coverage(
                day,
                plan["candidates"],
                args.cache,
            )
            for day in plan["fetch_dates"]
        ],
        ignore_index=True,
    ).sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, authoritative)


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
    pool_audit = tables["data_coverage.csv"].loc[
        tables["data_coverage.csv"]["record_type"].eq("pool")
    ]
    assert pool_audit[
        [
            "ranked_count",
            "age_exclusions",
            "suspension_exclusions",
            "daily_eligible_count",
            "eligible_count",
            "missing_or_stale_attribute_exclusions",
            "st_exclusions",
            "invalid_float_cap_exclusions",
            "final_count",
        ]
    ].to_dict("records") == [
        {
            "ranked_count": 6.0,
            "age_exclusions": 0.0,
            "suspension_exclusions": 0.0,
            "daily_eligible_count": 6.0,
            "eligible_count": 6.0,
            "missing_or_stale_attribute_exclusions": 0.0,
            "st_exclusions": 0.0,
            "invalid_float_cap_exclusions": 0.0,
            "final_count": 6.0,
        }
        for _ in dates
    ]
    report_text = (args.output / "report.md").read_text(encoding="utf-8")
    for phrase in [
        "固定验证区间",
        "API 实际区间",
        "预热起点",
        "ST 状态最多滞后 4 个交易日",
        "行业列可能不是严格时点数据",
        "六个月初步证据",
        "单边实际成交成本 20.0 bp",
        "年龄 0 个股日",
        "停牌 0 个股日",
        "属性缺失/陈旧 0 个股日",
        "ST 0 个股日",
        "无效流通市值 0 个股日",
        "分钟 ok率：150/150 = 100.00%",
        "最终 pool/ranked 覆盖率：150/150 = 100.00%",
        "双边总成交额换手",
        "达到" if "达到" in report_text else "未达到",
    ]:
        assert phrase in report_text


def test_validate_rejects_each_corrupt_cache_layer_read_only(tmp_path):
    args, dates, _ = _write_offline_validation_case(tmp_path)
    _, manifest = intraday_data._day_paths(dates[0], args.cache)
    cases = [
        (args.cache / "attributes.parquet", b"bad attributes", "attributes cache"),
        (args.cache / "attributes.json", b"{}", "attributes cache"),
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
        (
            args.cache / "adjusted_daily.json",
            b"{}",
            "adjusted daily cache",
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


@pytest.mark.parametrize("corruption", ["missing", "tampered"])
def test_validate_requires_untampered_attributes_manifest_read_only(
    tmp_path,
    corruption,
):
    args, _, _ = _write_offline_validation_case(tmp_path)
    manifest_path = args.cache / "attributes.json"
    if corruption == "missing":
        manifest_path.unlink()
        error = FileNotFoundError
        message = "attributes_manifest"
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["metadata"][0]["columns"][0] = "tampered"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        error = ValueError
        message = "attributes cache"
    before = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }

    with pytest.raises(error, match=message):
        run_validate(args)

    after = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not args.output.exists()


def test_validate_requires_adjusted_completion_manifest_read_only(tmp_path):
    args, _, _ = _write_offline_validation_case(tmp_path)
    manifest_path = args.cache / "adjusted_daily.json"
    manifest_path.unlink()
    before = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileNotFoundError, match="adjusted_daily_manifest"):
        run_validate(args)

    after = {
        path.relative_to(args.cache): path.read_bytes()
        for path in args.cache.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "case",
    [
        "empty_ok_partition",
        "missing_ok_code",
        "fake_ok_count",
        "failed_has_rows",
        "no_data_has_rows",
        "no_data_nonzero_count",
    ],
)
def test_validate_rejects_minute_partition_coverage_inconsistency(
    tmp_path,
    case,
):
    args, dates, codes = _write_offline_validation_case(tmp_path)
    parquet_path, manifest_path = intraday_data._day_paths(dates[0], args.cache)
    sidecar_path = intraday_data._day_coverage_path(dates[0], args.cache)
    coverage_path = args.cache / "data_coverage.parquet"
    minute = pd.read_parquet(parquet_path)
    daily_coverage = pd.read_parquet(sidecar_path)
    coverage = pd.read_parquet(coverage_path)
    first = coverage["date"].eq(dates[0]) & coverage["code"].eq(codes[0])
    daily_first = (
        daily_coverage["date"].eq(dates[0])
        & daily_coverage["code"].eq(codes[0])
    )

    if case == "empty_ok_partition":
        minute = minute.iloc[0:0]
    elif case == "missing_ok_code":
        minute = minute.loc[minute["code"].ne(codes[0])]
    elif case == "fake_ok_count":
        coverage.loc[first, "minute_count"] = 201
        daily_coverage.loc[daily_first, "minute_count"] = 201
    elif case == "failed_has_rows":
        coverage.loc[first, "reason"] = "too_few_trades"
        daily_coverage.loc[daily_first, "reason"] = "too_few_trades"
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["statuses"][codes[0]] = "no_data"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        coverage.loc[
            first,
            ["reason", "minute_count", "amount_relative_error"],
        ] = ["no_data", 0, np.nan]
        daily_coverage.loc[
            daily_first,
            ["reason", "minute_count", "amount_relative_error"],
        ] = ["no_data", 0, np.nan]
        if case == "no_data_nonzero_count":
            coverage.loc[first, "minute_count"] = 1
            daily_coverage.loc[daily_first, "minute_count"] = 1
            minute = minute.loc[minute["code"].ne(codes[0])]

    minute.to_parquet(parquet_path, index=False)
    daily_coverage.to_parquet(sidecar_path, index=False)
    coverage.to_parquet(coverage_path, index=False)

    with pytest.raises(ValueError, match=r"minute (coverage|partition|manifest)"):
        run_validate(args)


@pytest.mark.parametrize(
    ("reason", "count", "amount_error"),
    [
        ("too_few_trades", 199, 0.0),
        ("amount_mismatch", 199, 0.03),
        ("ok", 200, 0.03),
        ("amount_mismatch", 200, 0.01),
        ("ok", 200, np.nan),
    ],
)
def test_validate_rejects_impossible_coverage_reason_thresholds(
    tmp_path,
    reason,
    count,
    amount_error,
):
    args, dates, codes = _write_offline_validation_case(tmp_path)
    parquet_path, _ = intraday_data._day_paths(dates[0], args.cache)
    sidecar_path = intraday_data._day_coverage_path(dates[0], args.cache)
    coverage_path = args.cache / "data_coverage.parquet"
    minute = pd.read_parquet(parquet_path)
    daily_coverage = pd.read_parquet(sidecar_path)
    coverage = pd.read_parquet(coverage_path)
    first = coverage["date"].eq(dates[0]) & coverage["code"].eq(codes[0])
    daily_first = (
        daily_coverage["date"].eq(dates[0])
        & daily_coverage["code"].eq(codes[0])
    )
    coverage.loc[first, ["reason", "minute_count", "amount_relative_error"]] = [
        reason,
        count,
        amount_error,
    ]
    daily_coverage.loc[
        daily_first,
        ["reason", "minute_count", "amount_relative_error"],
    ] = [reason, count, amount_error]
    if reason != "ok":
        minute = minute.loc[minute["code"].ne(codes[0])]
    minute.to_parquet(parquet_path, index=False)
    daily_coverage.to_parquet(sidecar_path, index=False)
    coverage.to_parquet(coverage_path, index=False)

    with pytest.raises(ValueError, match="minute coverage"):
        run_validate(args)
