"""Command-line orchestration for the six-month intraday validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha101 import ths_http
from intraday import data, evaluate, factors, portfolio, preprocess, report

DEFAULT_START = "2026-01-12"
DEFAULT_END = "2026-07-10"
DEFAULT_WARMUP = "2025-12-11"
DEFAULT_DAILY = Path("alpha101/cache/ths_panel.pkl")
DEFAULT_CACHE = Path("intraday/cache")
DEFAULT_OUTPUT = Path("output/intraday_6m")
PLAN_SCHEMA_VERSION = 1
ADJUSTED_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, fetch, and validate A-share intraday factors."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "fetch", "validate", "all"):
        item = subparsers.add_parser(command)
        item.add_argument("--start", default=DEFAULT_START)
        item.add_argument("--end", default=DEFAULT_END)
        item.add_argument("--warmup", default=DEFAULT_WARMUP)
        item.add_argument("--top", type=int, default=500)
        item.add_argument("--daily-cache", type=Path, default=DEFAULT_DAILY)
        item.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        item.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        item.add_argument("--top-n", type=int, default=50)
        item.add_argument("--rebalance", type=int, default=5)
        item.add_argument("--cost-bps", type=float, default=20.0)
        item.add_argument("--min-count", type=int, default=400)
    return parser


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_day(value, name: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(day):
        raise ValueError(f"{name} must be a valid date")
    return day


def _normalized_parameters(values, context: str) -> dict:
    if isinstance(values, dict):
        source = values
    else:
        source = {
            "min_count": values.min_count,
            "top_n": values.top_n,
            "rebalance": values.rebalance,
            "cost_bps": values.cost_bps,
        }
    expected = {"min_count", "top_n", "rebalance", "cost_bps"}
    if set(source) != expected:
        raise ValueError(
            f"{context} parameters must contain exactly: "
            + ", ".join(sorted(expected))
        )
    result = {
        name: _positive_integer(source[name], f"{context} parameters {name}")
        for name in ("min_count", "top_n", "rebalance")
    }
    try:
        cost_bps = float(source["cost_bps"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} parameters cost_bps must be finite and nonnegative"
        ) from exc
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(
            f"{context} parameters cost_bps must be finite and nonnegative"
        )
    result["cost_bps"] = cost_bps
    return result


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sorted_pool(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["code"] = data._normalize_code_series(result["code"], "pool")
    order = [column for column in ["date", "liquidity_rank", "code"] if column in result]
    return result.sort_values(order, kind="mergesort").reset_index(drop=True)


def run_prepare(args):
    """Create the deterministic plan and pools, with plan.json written last."""
    top = _positive_integer(args.top, "top")
    start = _normalize_day(args.start, "start")
    end = _normalize_day(args.end, "end")
    warmup = _normalize_day(args.warmup, "warmup")
    if not warmup <= start <= end:
        raise ValueError("dates must satisfy warmup <= start <= end")
    parameters = _normalized_parameters(args, "CLI")

    raw = data.load_daily_raw(args.daily_cache)
    plan = data.prepare_universe(raw, start, end, top=top)
    if len(plan["eval_dates"]) == 0:
        raise ValueError("prepared plan has no evaluation dates")
    if plan["eval_dates"][0] != start or plan["eval_dates"][-1] != end:
        raise ValueError(
            "prepared eval_dates boundaries must equal requested start/end"
        )
    ranked_pool = _sorted_pool(plan["ranked_pool"])
    eligible_pool = _sorted_pool(plan["eligible_pool"])
    if "date" not in raw:
        raise ValueError("raw daily missing required column: date")
    fetch_calendar = pd.DatetimeIndex(
        data._normalize_dates(raw["date"], "raw daily").drop_duplicates()
    ).sort_values()
    fetch_dates = fetch_calendar[
        (fetch_calendar >= warmup) & (fetch_calendar <= end)
    ]
    if (
        len(fetch_dates) == 0
        or fetch_dates[0] != warmup
        or fetch_dates[-1] != end
    ):
        raise ValueError(
            "prepared fetch_dates boundaries must equal requested warmup/end"
        )
    estimated_rows = len(plan["candidates"]) * len(fetch_dates) * 241
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "warmup": warmup.strftime("%Y-%m-%d"),
        "top": top,
        "eval_dates": [day.strftime("%Y-%m-%d") for day in plan["eval_dates"]],
        "fetch_dates": [day.strftime("%Y-%m-%d") for day in fetch_dates],
        "candidates": list(plan["candidates"]),
        "estimated_rows": estimated_rows,
        "estimated_cells": estimated_rows * 3,
        "parameters": parameters,
    }
    cache = Path(args.cache)
    _atomic_write_parquet(ranked_pool, cache / "ranked_pool.parquet")
    _atomic_write_parquet(eligible_pool, cache / "eligible_pool.parquet")
    _atomic_write_json(payload, cache / "plan.json")
    loaded = _load_plan(cache)
    print(f"evaluation days: {len(loaded['eval_dates'])}")
    print(f"candidate union: {len(loaded['candidates'])}")
    print(f"estimated rows: {loaded['estimated_rows']}")
    print(f"estimated cells: {loaded['estimated_cells']}")
    print(f"warmup disclosure: {loaded['warmup'].date()}")
    return loaded


def _load_date_index(values, name: str) -> pd.DatetimeIndex:
    if not isinstance(values, list) or not values:
        raise ValueError(f"plan {name} must be a non-empty list")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, format="%Y-%m-%d"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"plan {name} contains invalid dates") from exc
    dates = dates.normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"plan {name} must be unique and increasing")
    return dates


def _load_pool(path: Path, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"cannot read {name}: {path}") from exc
    required = {"date", "code"}
    if name == "ranked_pool":
        required |= {"adv20", "liquidity_rank"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")
    try:
        frame = _sorted_pool(frame)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid date or code") from exc
    if frame.duplicated(["date", "code"]).any():
        raise ValueError(f"{name} contains duplicate date/code rows")
    return frame


def _load_plan(root) -> dict:
    """Load and structurally validate a complete prepared-cache plan."""
    root = Path(root)
    required_paths = {
        "plan": root / "plan.json",
        "ranked_pool": root / "ranked_pool.parquet",
        "eligible_pool": root / "eligible_pool.parquet",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing prepared cache files: {', '.join(sorted(missing))}"
        )
    try:
        payload = json.loads(required_paths["plan"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("plan.json is unreadable or corrupt") from exc
    if not isinstance(payload, dict):
        raise ValueError("plan.json must contain an object")
    required_keys = {
        "schema_version",
        "start",
        "end",
        "warmup",
        "top",
        "eval_dates",
        "fetch_dates",
        "candidates",
        "estimated_rows",
        "estimated_cells",
        "parameters",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        raise ValueError(f"plan missing required keys: {', '.join(missing_keys)}")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported plan schema_version")

    start = _normalize_day(payload["start"], "plan start")
    end = _normalize_day(payload["end"], "plan end")
    warmup = _normalize_day(payload["warmup"], "plan warmup")
    if not warmup <= start <= end:
        raise ValueError("plan dates must satisfy warmup <= start <= end")
    top = _positive_integer(payload["top"], "plan top")
    eval_dates = _load_date_index(payload["eval_dates"], "eval_dates")
    fetch_dates = _load_date_index(payload["fetch_dates"], "fetch_dates")
    if eval_dates[0] != start or eval_dates[-1] != end:
        raise ValueError("plan eval_dates boundaries must equal start/end dates")
    if fetch_dates[0] != warmup or fetch_dates[-1] != end:
        raise ValueError("plan fetch_dates boundaries must equal warmup/end dates")
    if ((fetch_dates < warmup) | (fetch_dates > end)).any():
        raise ValueError("plan fetch_dates fall outside warmup/end dates")
    if not eval_dates.isin(fetch_dates).all():
        raise ValueError("plan fetch_dates must contain every eval_date")
    parameters = _normalized_parameters(payload["parameters"], "plan")

    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("plan candidates must be a list")
    try:
        normalized_candidates = [data._normalize_code(value) for value in candidates]
    except ValueError as exc:
        raise ValueError("plan candidates contain invalid codes") from exc
    if normalized_candidates != sorted(set(normalized_candidates)):
        raise ValueError("plan candidates must be sorted and unique")

    for key in ("estimated_rows", "estimated_cells"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"plan {key} must be a nonnegative integer")
    expected_rows = len(normalized_candidates) * len(fetch_dates) * 241
    if payload["estimated_rows"] != expected_rows:
        raise ValueError("plan estimated_rows is inconsistent")
    if payload["estimated_cells"] != payload["estimated_rows"] * 3:
        raise ValueError("plan estimated_cells is inconsistent")

    ranked_pool = _load_pool(required_paths["ranked_pool"], "ranked_pool")
    eligible_pool = _load_pool(required_paths["eligible_pool"], "eligible_pool")
    eval_set = set(eval_dates)
    candidate_set = set(normalized_candidates)
    for name, frame in (("ranked_pool", ranked_pool), ("eligible_pool", eligible_pool)):
        if not set(frame["date"]).issubset(eval_set):
            raise ValueError(f"{name} contains dates outside eval_dates")
        if not set(frame["code"]).issubset(candidate_set):
            raise ValueError(f"{name} contains codes outside candidates")
    if set(ranked_pool["code"]) != candidate_set:
        raise ValueError("plan candidates do not match ranked_pool union")
    counts = ranked_pool.groupby("date", observed=True).size()
    if counts.gt(top).any():
        raise ValueError("ranked_pool exceeds top quota")
    for _, group in ranked_pool.groupby("date", observed=True):
        ranks = pd.to_numeric(group["liquidity_rank"], errors="coerce")
        if ranks.isna().any() or sorted(ranks.astype(int)) != list(range(1, len(group) + 1)):
            raise ValueError("ranked_pool has invalid liquidity_rank quota")
        adv = pd.to_numeric(group["adv20"], errors="coerce")
        if not (np.isfinite(adv) & adv.gt(0)).all():
            raise ValueError("ranked_pool has invalid adv20")
    ranked_keys = set(map(tuple, ranked_pool[["date", "code"]].itertuples(index=False, name=None)))
    eligible_keys = set(map(tuple, eligible_pool[["date", "code"]].itertuples(index=False, name=None)))
    if not eligible_keys.issubset(ranked_keys):
        raise ValueError("eligible_pool must be a subset of ranked_pool")

    return {
        **payload,
        "start": start,
        "end": end,
        "warmup": warmup,
        "eval_dates": eval_dates,
        "fetch_dates": fetch_dates,
        "candidates": normalized_candidates,
        "parameters": parameters,
        "ranked_pool": ranked_pool,
        "eligible_pool": eligible_pool,
    }


def _normalize_keyed_frame(
    frame: pd.DataFrame,
    required,
    context: str,
) -> pd.DataFrame:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")
    result = frame.copy()
    try:
        result["date"] = data._normalize_dates(result["date"], context)
        result["code"] = data._normalize_code_series(result["code"], context)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} contains invalid date or code") from exc
    if result.duplicated(["date", "code"]).any():
        raise ValueError(f"{context} contains duplicate date/code rows")
    return result.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def _validate_attributes(
    frame: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    candidates: list[str],
) -> pd.DataFrame:
    result = _normalize_keyed_frame(
        frame,
        ["date", "code", "name", "float_cap", "industry"],
        "attributes cache",
    )
    expected = {(day, code) for day in anchors for code in candidates}
    result = result.loc[result["code"].isin(candidates)].copy()
    actual = set(result[["date", "code"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("attributes cache does not cover every anchor/candidate")
    result["float_cap"] = pd.to_numeric(result["float_cap"], errors="coerce")
    valid = np.isfinite(result["float_cap"]) & result["float_cap"].gt(0)
    if not valid.all():
        raise ValueError("attributes cache contains invalid float_cap")
    if result[["name", "industry"]].isna().any().any():
        raise ValueError("attributes cache contains missing name or industry")
    return result


def _validate_adjusted(
    frame: pd.DataFrame,
    candidates: list[str],
    warmup: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    result = _normalize_keyed_frame(
        frame,
        ["date", "code", "open", "close"],
        "adjusted daily cache",
    )
    if set(result["code"]) != set(candidates):
        raise ValueError("adjusted daily cache does not cover every candidate")
    if result["date"].min() < warmup or result["date"].max() > end:
        raise ValueError("adjusted daily cache dates fall outside requested range")
    for column in ("open", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not (np.isfinite(result[column]) & result[column].gt(0)).all():
            raise ValueError(f"adjusted daily cache contains invalid {column}")
    return result


def _adjusted_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(b"intraday-adjusted-daily-v1\n")
    for row in frame[["date", "code", "open", "close"]].itertuples(
        index=False,
        name=None,
    ):
        day, code, open_price, close_price = row
        canonical = (
            f"{pd.Timestamp(day):%Y-%m-%d}\t{code}\t"
            f"{float(open_price).hex()}\t{float(close_price).hex()}\n"
        )
        digest.update(canonical.encode("ascii"))
    return digest.hexdigest()


def _adjusted_manifest_payload(
    frame: pd.DataFrame,
    candidates: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    counts = frame.groupby("code", observed=True).size().reindex(candidates)
    return {
        "schema_version": ADJUSTED_SCHEMA_VERSION,
        "request": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "candidates": candidates,
        },
        "frame": {
            "columns": ["date", "code", "open", "close"],
            "row_count": len(frame),
            "date_min": frame["date"].min().strftime("%Y-%m-%d"),
            "date_max": frame["date"].max().strftime("%Y-%m-%d"),
            "code_counts": {
                code: int(counts.loc[code]) for code in candidates
            },
            "sha256": _adjusted_content_hash(frame),
        },
    }


def _read_adjusted_cache(
    parquet_path: Path,
    manifest_path: Path,
    candidates: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    required: bool,
):
    if not parquet_path.is_file() or not manifest_path.is_file():
        if required:
            missing = []
            if not parquet_path.is_file():
                missing.append("adjusted_daily.parquet")
            if not manifest_path.is_file():
                missing.append("adjusted_daily.json")
            raise FileNotFoundError(
                "missing validation cache files: " + ", ".join(missing)
            )
        return None
    try:
        frame = pd.read_parquet(parquet_path)
        normalized = _validate_adjusted(frame, candidates, start, end)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _adjusted_manifest_payload(
            normalized,
            candidates,
            start,
            end,
        )
        if payload != expected:
            raise ValueError("adjusted daily completion manifest mismatch")
        return normalized
    except Exception as exc:
        if required:
            raise ValueError(
                "adjusted daily cache or completion manifest is corrupt"
            ) from exc
        return None


def _publish_adjusted_cache(
    frame: pd.DataFrame,
    parquet_path: Path,
    manifest_path: Path,
    payload: dict,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary_parquet, index=False)
        temporary_manifest.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_parquet.replace(parquet_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_parquet.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _read_valid_cache(path: Path, validator, *args):
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
        return validator(frame, *args)
    except (OSError, ValueError):
        return None


def _read_manifest(day, root, candidates: list[str]) -> dict[str, str]:
    _, manifest_path = data._day_paths(day, root)
    if not data.day_complete(day, candidates, root):
        raise ValueError(f"minute manifest incomplete for {pd.Timestamp(day).date()}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"minute manifest corrupt for {pd.Timestamp(day).date()}"
        ) from exc
    return payload["statuses"]


def _read_minute_partition(day, root, candidates: list[str]) -> tuple[pd.DataFrame, dict]:
    statuses = _read_manifest(day, root, candidates)
    parquet_path, _ = data._day_paths(day, root)
    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(
            f"minute partition unreadable for {pd.Timestamp(day).date()}"
        ) from exc
    missing = sorted(set(data.MINUTE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            f"minute partition missing columns for {pd.Timestamp(day).date()}: "
            + ", ".join(missing)
        )
    if frame.empty:
        return frame, statuses
    try:
        normalized_codes = data._normalize_code_series(
            frame["code"],
            "minute partition",
        )
        times = pd.to_datetime(frame["time"], errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"minute partition has invalid keys for {pd.Timestamp(day).date()}"
        ) from exc
    if not set(normalized_codes).issubset(
        {code for code, status in statuses.items() if status == "returned"}
    ):
        raise ValueError("minute partition contains a non-returned code")
    if not times.dt.normalize().eq(pd.Timestamp(day).normalize()).all():
        raise ValueError("minute partition contains rows from another date")
    if pd.DataFrame({"code": normalized_codes, "time": times}).duplicated().any():
        raise ValueError("minute partition contains duplicate code/time rows")
    for column in ("close", "volume", "amount"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"minute partition contains invalid {column}")
    return frame, statuses


def _normalize_coverage(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    result = _normalize_keyed_frame(
        frame,
        data.COVERAGE_COLUMNS,
        context,
    )
    result["minute_count"] = pd.to_numeric(
        result["minute_count"], errors="coerce"
    )
    if (
        result["minute_count"].isna().any()
        or result["minute_count"].lt(0).any()
        or not np.equal(result["minute_count"] % 1, 0).all()
    ):
        raise ValueError(f"{context} contains invalid minute_count")
    result["minute_count"] = result["minute_count"].astype("int64")
    result["amount_relative_error"] = pd.to_numeric(
        result["amount_relative_error"], errors="coerce"
    )
    allowed = {
        "ok",
        "too_few_minutes",
        "too_few_trades",
        "amount_mismatch",
        "no_data",
    }
    if not result["reason"].isin(allowed).all():
        raise ValueError(f"{context} contains invalid reason")
    return result


def _existing_coverage(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=data.COVERAGE_COLUMNS)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError("coverage cache is unreadable or corrupt") from exc
    return _normalize_coverage(frame, "coverage cache")


def _validate_day_minute_coverage(
    day: pd.Timestamp,
    frame: pd.DataFrame,
    statuses: dict[str, str],
    indexed_coverage: pd.DataFrame,
) -> None:
    if frame.empty:
        clean_counts = pd.Series(dtype="int64")
    else:
        clean_codes = data._normalize_code_series(
            frame["code"],
            "minute partition",
        )
        clean_counts = clean_codes.value_counts()
    for code, status in statuses.items():
        row = indexed_coverage.loc[(day, code)]
        reason = row["reason"]
        minute_count = int(row["minute_count"])
        amount_error = float(row["amount_relative_error"])
        clean_count = int(clean_counts.get(code, 0))

        if status == "no_data":
            if (
                reason != "no_data"
                or minute_count != 0
                or clean_count != 0
                or not np.isnan(amount_error)
            ):
                raise ValueError(
                    f"minute coverage conflicts with no_data manifest: "
                    f"{day.date()} {code}"
                )
            continue

        if reason == "no_data":
            raise ValueError(
                f"minute coverage conflicts with returned manifest: "
                f"{day.date()} {code}"
            )
        if np.isnan(amount_error) or amount_error < 0:
            raise ValueError(
                f"minute coverage has invalid amount error: {day.date()} {code}"
            )
        if reason == "ok":
            if minute_count < 200:
                raise ValueError(
                    f"minute coverage marks too-short data as ok: "
                    f"{day.date()} {code}"
                )
            if clean_count != minute_count:
                raise ValueError(
                    f"minute coverage count does not match clean parquet: "
                    f"{day.date()} {code}"
                )
            if not np.isfinite(amount_error) or amount_error > 0.02:
                raise ValueError(
                    f"minute coverage ok amount error exceeds 2%: "
                    f"{day.date()} {code}"
                )
        else:
            if clean_count != 0:
                raise ValueError(
                    f"minute coverage failed code remains in clean parquet: "
                    f"{day.date()} {code}"
                )
            if reason == "too_few_minutes" and minute_count >= 200:
                raise ValueError(
                    f"minute coverage too_few_minutes count is inconsistent: "
                    f"{day.date()} {code}"
                )
            if reason in {"too_few_trades", "amount_mismatch"} and minute_count < 200:
                raise ValueError(
                    f"minute coverage {reason} count is inconsistent: "
                    f"{day.date()} {code}"
                )
            if reason == "amount_mismatch" and amount_error <= 0.02:
                raise ValueError(
                    f"minute coverage amount_mismatch error is inconsistent: "
                    f"{day.date()} {code}"
                )


def _complete_coverage(
    plan: dict,
    root: Path,
    previous: pd.DataFrame,
    delta: pd.DataFrame,
) -> pd.DataFrame:
    pieces = [frame for frame in (previous, delta) if not frame.empty]
    combined = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=data.COVERAGE_COLUMNS)
    )
    if not combined.empty:
        combined = combined.drop_duplicates(["date", "code"], keep="last")
        combined = _normalize_coverage(combined, "merged coverage")
    rows = []
    day_partitions = {}
    combined_keys = set(
        combined[["date", "code"]].itertuples(index=False, name=None)
    )
    expected_keys = set()
    for day in plan["fetch_dates"]:
        frame, statuses = _read_minute_partition(
            day,
            root,
            plan["candidates"],
        )
        day_partitions[day] = (frame, statuses)
        for code, status in statuses.items():
            key = (day, code)
            expected_keys.add(key)
            if status == "no_data" and key not in combined_keys:
                rows.append(
                    {
                        "date": day,
                        "code": code,
                        "minute_count": 0,
                        "amount_relative_error": np.nan,
                        "reason": "no_data",
                    }
                )
            elif status == "returned" and key not in combined_keys:
                raise ValueError(
                    f"coverage missing returned code {code} on {day.date()}"
                )
    if rows:
        combined = pd.concat([combined, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.loc[
        combined[["date", "code"]]
        .apply(tuple, axis=1)
        .isin(expected_keys)
    ]
    combined = _normalize_coverage(combined, "complete coverage")
    actual_keys = set(
        combined[["date", "code"]].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise ValueError("coverage does not exactly cover the prepared plan")
    indexed = combined.set_index(["date", "code"])
    for day, (frame, statuses) in day_partitions.items():
        _validate_day_minute_coverage(
            day,
            frame,
            statuses,
            indexed,
        )
    return combined


def _assert_cli_matches_plan(args, plan: dict) -> None:
    for name in ("start", "end", "warmup"):
        if _normalize_day(getattr(args, name), f"CLI {name}") != plan[name]:
            raise ValueError(f"CLI {name} does not match prepared plan")
    if args.top != plan["top"]:
        raise ValueError("CLI top does not match prepared plan")
    parameters = _normalized_parameters(args, "CLI")
    for name, expected in plan["parameters"].items():
        if parameters[name] != expected:
            raise ValueError(f"CLI {name} does not match prepared plan")


def run_fetch(args):
    """Resume the immutable prepared plan with one access token per run."""
    plan = _load_plan(args.cache)
    _assert_cli_matches_plan(args, plan)
    raw = data.load_daily_raw(args.daily_cache)
    token = ths_http.get_access_token()
    root = Path(args.cache)
    anchors = plan["eval_dates"][:: _positive_integer(args.rebalance, "rebalance")]

    attributes_path = root / "attributes.parquet"
    attributes = _read_valid_cache(
        attributes_path,
        _validate_attributes,
        anchors,
        plan["candidates"],
    )
    if attributes is None:
        attributes = _validate_attributes(
            data.fetch_attributes(anchors, token),
            anchors,
            plan["candidates"],
        )
        _atomic_write_parquet(attributes, attributes_path)

    adjusted_path = root / "adjusted_daily.parquet"
    adjusted_manifest_path = root / "adjusted_daily.json"
    adjusted = _read_adjusted_cache(
        adjusted_path,
        adjusted_manifest_path,
        plan["candidates"],
        plan["warmup"],
        plan["end"],
        required=False,
    )
    if adjusted is None:
        adjusted = _validate_adjusted(
            data.fetch_adjusted_daily(
                plan["candidates"],
                plan["warmup"],
                plan["end"],
                token,
            ),
            plan["candidates"],
            plan["warmup"],
            plan["end"],
        )
        adjusted_manifest = _adjusted_manifest_payload(
            adjusted,
            plan["candidates"],
            plan["warmup"],
            plan["end"],
        )
        _publish_adjusted_cache(
            adjusted,
            adjusted_path,
            adjusted_manifest_path,
            adjusted_manifest,
        )

    coverage_path = root / "data_coverage.parquet"
    previous = _existing_coverage(coverage_path)
    delta = data.fetch_minute_partitions(plan, raw, root, token)
    if delta.empty:
        delta = pd.DataFrame(columns=data.COVERAGE_COLUMNS)
    else:
        delta = _normalize_coverage(delta, "coverage delta")
    coverage = _complete_coverage(plan, root, previous, delta)
    _atomic_write_parquet(coverage, coverage_path)
    print(f"minute coverage rows: {len(coverage)}")
    print(f"completed minute dates: {len(plan['fetch_dates'])}")
    return coverage


def _load_validation_caches(args, plan: dict):
    root = Path(args.cache)
    paths = {
        "attributes": root / "attributes.parquet",
        "adjusted_daily": root / "adjusted_daily.parquet",
        "adjusted_daily_manifest": root / "adjusted_daily.json",
        "data_coverage": root / "data_coverage.parquet",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing validation cache files: {', '.join(sorted(missing))}"
        )
    anchors = plan["eval_dates"][:: _positive_integer(args.rebalance, "rebalance")]
    try:
        attributes_frame = pd.read_parquet(paths["attributes"])
    except Exception as exc:
        raise ValueError("attributes cache is unreadable or corrupt") from exc
    attributes = _validate_attributes(
        attributes_frame,
        anchors,
        plan["candidates"],
    )
    adjusted = _read_adjusted_cache(
        paths["adjusted_daily"],
        paths["adjusted_daily_manifest"],
        plan["candidates"],
        plan["warmup"],
        plan["end"],
        required=True,
    )
    coverage = _existing_coverage(paths["data_coverage"])
    return attributes, adjusted, coverage


def _validate_coverage_and_partitions(
    plan: dict,
    root: Path,
    coverage: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    expected = {
        (day, code)
        for day in plan["fetch_dates"]
        for code in plan["candidates"]
    }
    actual = set(
        coverage[["date", "code"]].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError("coverage does not exactly cover the prepared plan")
    indexed = coverage.set_index(["date", "code"])
    partitions = []
    for day in plan["fetch_dates"]:
        frame, statuses = _read_minute_partition(
            day,
            root,
            plan["candidates"],
        )
        partitions.append((day, frame))
        _validate_day_minute_coverage(
            day,
            frame,
            statuses,
            indexed,
        )
    return partitions


def _normalize_raw_daily(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "open", "high", "low", "close"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(
            f"raw daily missing required columns: {', '.join(missing)}"
        )
    result = raw.copy()
    result["date"] = data._normalize_dates(result["date"], "raw daily")
    result["code"] = data._normalize_code_series(result["code"], "raw daily")
    if result.duplicated(["date", "code"]).any():
        raise ValueError("raw daily contains duplicate date/code rows")
    for column in portfolio.RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.set_index(["date", "code"])[portfolio.RAW_COLUMNS]
        .sort_index()
    )


def _pool_coverage(plan: dict, final_pool: pd.DataFrame) -> pd.DataFrame:
    dates = plan["eval_dates"]
    result = pd.DataFrame({"date": dates})
    for name, frame in (
        ("ranked_count", plan["ranked_pool"]),
        ("daily_eligible_count", plan["eligible_pool"]),
        ("final_count", final_pool),
    ):
        counts = frame.groupby("date", observed=True).size()
        result[name] = result["date"].map(counts).fillna(0).astype(int)
    result["pre_attribute_exclusions"] = (
        result["ranked_count"] - result["daily_eligible_count"]
    )
    result["attribute_exclusions"] = (
        result["daily_eligible_count"] - result["final_count"]
    )
    result["record_type"] = "pool"
    return result


def _threshold_disclosures(
    summary: pd.DataFrame,
    metrics: dict[str, float],
) -> list[str]:
    score_rows = summary.loc[summary["factor"].eq("score")]
    score = score_rows.iloc[0] if not score_rows.empty else pd.Series(dtype=float)
    ic_mean = score.get("ic_mean", np.nan)
    monotonicity = score.get("monotonicity", np.nan)
    strategy_total = metrics.get("strategy_total", np.nan)
    benchmark_total = metrics.get("benchmark_total", np.nan)
    achieved_ic = bool(np.isfinite(ic_mean) and ic_mean >= 0.03)
    achieved_monotonicity = bool(
        np.isfinite(monotonicity) and monotonicity >= 0.8
    )
    achieved_excess = bool(
        np.isfinite(strategy_total)
        and np.isfinite(benchmark_total)
        and strategy_total > benchmark_total
    )
    word = lambda achieved: "达到" if achieved else "未达到"
    return [
        f"固定阈值 综合 RankIC >= 0.03：{word(achieved_ic)}",
        f"固定阈值 五组单调性 >= 0.8：{word(achieved_monotonicity)}",
        f"固定阈值 扣费后累计超额 > 0：{word(achieved_excess)}",
    ]


def run_validate(args):
    """Read the completed cache only, evaluate it, and write report outputs."""
    plan = _load_plan(args.cache)
    _assert_cli_matches_plan(args, plan)

    attributes, adjusted, minute_coverage = _load_validation_caches(args, plan)
    partitions = _validate_coverage_and_partitions(
        plan,
        Path(args.cache),
        minute_coverage,
    )
    raw = data.load_daily_raw(args.daily_cache)
    raw_daily = _normalize_raw_daily(raw)

    final_pool = data.apply_attribute_filters(
        plan["eligible_pool"],
        attributes,
        plan["eval_dates"],
    )
    factor_data = factors.factor_panels(
        partitions,
        plan["candidates"],
        plan["fetch_dates"],
    )
    min_count = _positive_integer(args.min_count, "min_count")
    processed = preprocess.preprocess_panels(
        factor_data,
        final_pool,
        attributes,
        min_count=min_count,
    )
    scored = {
        name: frame.reindex(
            index=plan["eval_dates"],
            columns=plan["candidates"],
        )
        for name, frame in preprocess.compose(processed).items()
    }

    adjusted_open = (
        adjusted.pivot(index="date", columns="code", values="open")
        .reindex(index=plan["eval_dates"], columns=plan["candidates"])
        .sort_index(axis=1)
    )
    summary, daily_ic, quantiles = evaluate.evaluate_factors(
        scored,
        adjusted_open,
        min_count=min_count,
    )
    rebalance = _positive_integer(args.rebalance, "rebalance")
    top_n = _positive_integer(args.top_n, "top_n")
    targets = portfolio.build_targets(
        scored["score"],
        final_pool,
        every=rebalance,
        top_n=top_n,
        min_count=min_count,
    )
    benchmark_targets = portfolio.build_benchmark_targets(
        final_pool,
        targets.keys(),
    )
    strategy_net = portfolio.simulate(
        targets,
        adjusted_open,
        raw_daily,
        args.cost_bps,
    )
    strategy_gross = portfolio.simulate(
        targets,
        adjusted_open,
        raw_daily,
        0,
    )
    benchmark_net = portfolio.simulate(
        benchmark_targets,
        adjusted_open,
        raw_daily,
        args.cost_bps,
    )
    benchmark_gross = portfolio.simulate(
        benchmark_targets,
        adjusted_open,
        raw_daily,
        0,
    )
    metrics = portfolio.portfolio_metrics(strategy_net, benchmark_net)
    nav = pd.concat(
        {
            "strategy_net": strategy_net["nav"],
            "strategy_gross": strategy_gross["nav"],
            "benchmark_net": benchmark_net["nav"],
            "benchmark_gross": benchmark_gross["nav"],
        },
        axis=1,
    ).rename_axis("date").reset_index()
    trades = pd.concat(
        [
            strategy_net["trades"].assign(portfolio="strategy"),
            benchmark_net["trades"].assign(portfolio="benchmark"),
        ],
        ignore_index=True,
    )
    pool_coverage = _pool_coverage(plan, final_pool)
    coverage_output = pd.concat(
        [
            minute_coverage.assign(record_type="minute"),
            pool_coverage,
        ],
        ignore_index=True,
        sort=False,
    )

    adjusted_start = adjusted["date"].min().strftime("%Y-%m-%d")
    adjusted_end = adjusted["date"].max().strftime("%Y-%m-%d")
    score_sample_days = int(daily_ic.get("score", pd.Series(dtype=float)).notna().sum())
    quality_exclusions = int(minute_coverage["reason"].ne("ok").sum())
    pool_exclusions = int(
        pool_coverage[
            ["pre_attribute_exclusions", "attribute_exclusions"]
        ].to_numpy().sum()
    )
    disclosures = [
        f"固定验证区间 {plan['start'].date()} 至 {plan['end'].date()}",
        f"API 实际区间 {adjusted_start} 至 {adjusted_end}",
        f"预热起点 {plan['warmup'].date()}",
        "ST 状态最多滞后 4 个交易日",
        "行业列可能不是严格时点数据，仅按可获得分类口径处理",
        f"六个月初步证据；样本 {len(plan['eval_dates'])} 个交易日，"
        f"有效综合 RankIC 样本日 {score_sample_days}",
        f"参数固定：top={plan['top']}，top_n={top_n}，"
        f"rebalance={rebalance}，min_count={min_count}",
        f"单边实际成交成本 {float(args.cost_bps)} bp；同时报告毛值与净值",
        f"剔除统计：分钟质量/无数据 {quality_exclusions} 个股日，"
        f"股票池/属性 {pool_exclusions} 个股日",
        *_threshold_disclosures(summary, metrics),
    ]
    results = {
        "factor_summary": summary,
        "daily_ic": daily_ic.reset_index(),
        "quantile_returns": quantiles,
        "portfolio_nav": nav,
        "trades": trades,
        "data_coverage": coverage_output,
        "portfolio_metrics": metrics,
        "disclosures": disclosures,
    }
    return report.write_outputs(results, args.output)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        run_prepare(args)
        run_fetch(args)
        run_validate(args)
        return
    actions = {
        "prepare": run_prepare,
        "fetch": run_fetch,
        "validate": run_validate,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
