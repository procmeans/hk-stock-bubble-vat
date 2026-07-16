# A股分钟量价三因子六个月验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 下载并缓存动态流动性前 500 候选并集的六个月 iFinD 一分钟数据，验证 RSkew、CPV、Smart Money 及等权合成，并完成带 T+1 开盘、一字板和 20 bp 成本的组合回测。

**Architecture:** 新增独立 `intraday/` 研究包，复用现有 A 股日线缓存和 iFinD HTTP 客户端。原始分钟行情按日原子缓存，随后依次压缩成日因子、做时点中性化、统计 5 日 RankIC/分层，并由事件式开盘成交器回测策略与相同口径基准。

**Tech Stack:** Python 3.14、pandas、numpy、matplotlib、pyarrow、requests、pytest；不新增依赖。

## Global Constraints

- 固定验证区间：2026-01-12 至 2026-07-10；分钟预热从 2025-12-11 开始。
- 动态池：T-1 可见的 ADV20 前 500，之后剔除年龄不足 60 日、T 日停牌和 ST；不递补。
- 分钟字段仅 `close,volume,amount`，`Fill=Original`，每天至少 200 根有效记录。
- 分钟成交额与不复权日线成交额相对误差必须不超过 2%。
- 因子：20 日窗口、至少 15 个有效股票日；有效截面至少 400 只。
- 方向与权重固定：`-RSkew`、`-CPV_mean`、`-CPV_std`、`-SmartQ`；三个逻辑块等权。
- 收益使用 `CPS=3` 后复权开盘价；一字板与数据校验使用 `CPS=1` 不复权日线。
- T 日收盘信号、T+1 开盘成交、每 5 日调仓、Top 50 等权、双边实际成交额各收 20 bp。
- 不调参、不按样本内结果翻转方向、不使用 Level-2、不修改现有模拟盘行为。
- 每个生产行为必须先有会因缺失行为而失败的测试，再写最小实现。

---

### Task 1: 扩展 iFinD HTTP 行情客户端

**Files:**
- Modify: `alpha101/ths_http.py:85-108`
- Test: `alpha101/tests/test_ths_http.py`

**Interfaces:**
- Produces: `history_quotation(..., functionpara: dict | None = None) -> DataFrame`
- Produces: `high_frequency(codes, indicators, starttime, endtime, functionpara=None, ...) -> DataFrame`

- [ ] **Step 1: 写请求载荷失败测试**

```python
def test_history_quotation_includes_functionpara(monkeypatch):
    seen = {}
    monkeypatch.setattr(ths_http, "post", lambda endpoint, payload, **kwargs:
                        seen.update(endpoint=endpoint, payload=payload) or {"tables": []})
    ths_http.history_quotation(
        ["000001.SZ"], ["open", "close"], "2026-01-01", "2026-01-31",
        functionpara={"CPS": "3", "Fill": "Omit"}, access_token="token",
    )
    assert seen == {"endpoint": "cmd_history_quotation", "payload": {
        "codes": "000001.SZ", "indicators": "open,close",
        "startdate": "2026-01-01", "enddate": "2026-01-31",
        "functionpara": {"CPS": "3", "Fill": "Omit"},
    }}


def test_high_frequency_posts_and_flattens(monkeypatch):
    seen = {}
    def fake_post(endpoint, payload, **kwargs):
        seen.update(endpoint=endpoint, payload=payload)
        return {"tables": [{"thscode": "000001.SZ", "time": ["2026-01-12 09:30"],
                            "table": {"close": [10.0], "volume": [100.0],
                                      "amount": [1000.0]}}]}
    monkeypatch.setattr(ths_http, "post", fake_post)
    result = ths_http.high_frequency(
        ["000001.SZ"], ["close", "volume", "amount"],
        "2026-01-12 09:30:00", "2026-01-12 15:00:00",
        functionpara={"Fill": "Original", "Timeformat": "LocalTime"},
    )
    assert seen["endpoint"] == "high_frequency"
    assert seen["payload"]["functionpara"]["Fill"] == "Original"
    assert result.loc[0, "amount"] == 1000.0
```

- [ ] **Step 2: 运行并确认因签名/函数缺失而失败**

Run: `.venv/bin/python -m pytest alpha101/tests/test_ths_http.py -q`

Expected: `history_quotation()` 拒绝 `functionpara`，且 `high_frequency` 不存在。

- [ ] **Step 3: 实现两个薄封装**

```python
def history_quotation(codes, indicators, startdate, enddate,
                      functionpara=None, access_token=None,
                      refresh_token=None, timeout=60):
    payload = {
        "codes": join_if_sequence(codes),
        "indicators": join_if_sequence(indicators),
        "startdate": startdate,
        "enddate": enddate,
    }
    if functionpara is not None:
        payload["functionpara"] = functionpara
    data = post("cmd_history_quotation", payload, access_token=access_token,
                refresh_token=refresh_token, timeout=timeout)
    return tables_to_dataframe(data)


def high_frequency(codes, indicators, starttime, endtime,
                   functionpara=None, access_token=None,
                   refresh_token=None, timeout=60):
    payload = {
        "codes": join_if_sequence(codes),
        "indicators": join_if_sequence(indicators),
        "starttime": starttime,
        "endtime": endtime,
    }
    if functionpara is not None:
        payload["functionpara"] = functionpara
    data = post("high_frequency", payload, access_token=access_token,
                refresh_token=refresh_token, timeout=timeout)
    return tables_to_dataframe(data)
```

- [ ] **Step 4: 运行客户端测试**

Run: `.venv/bin/python -m pytest alpha101/tests/test_ths_http.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add alpha101/ths_http.py alpha101/tests/test_ths_http.py
git commit -m "feat(ifind): support minute and adjusted history requests"
```

---

### Task 2: 动态流动性股票池与准备清单

**Files:**
- Create: `intraday/__init__.py`
- Create: `intraday/data.py`
- Create: `intraday/tests/__init__.py`
- Create: `intraday/tests/test_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `load_daily_raw(path: Path) -> DataFrame`
- Produces: `prepare_universe(raw, start, end, top=500, adv_window=20, min_age=60) -> dict`
- Dict keys: `eval_dates`, `fetch_dates`, `ranked_pool`, `eligible_pool`, `candidates`, `estimated_rows`, `estimated_cells`

- [ ] **Step 1: 写 T-1、年龄、停牌和并列测试**

```python
def test_prepare_universe_uses_lagged_adv_and_no_replacement():
    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = []
    amounts = {"000001": [100, 100, 1, 1, 1, 1],
               "000002": [50, 50, 200, 200, 200, 200],
               "000003": [50, 50, 40, 40, 40, 40]}
    for code, values in amounts.items():
        for day, amount in zip(dates, values):
            rows.append({"code": code, "date": day, "open": 10, "high": 10,
                         "low": 10, "close": 10, "volume": 100,
                         "amount": amount})
    raw = pd.DataFrame(rows)
    plan = prepare_universe(raw, dates[2], dates[-1], top=2,
                            adv_window=2, min_age=0)
    first_day = dates[2]
    first = plan["ranked_pool"].query("date == @first_day")
    assert first["code"].tolist() == ["000001", "000002"]
    assert plan["candidates"] == ["000001", "000002", "000003"]


def test_prepare_universe_filters_after_top_rank():
    dates = pd.bdate_range("2025-10-01", periods=65)
    raw = pd.DataFrame([
        {"code": code, "date": day, "open": 10.0, "high": 10.0,
         "low": 10.0, "close": 10.0, "volume": 100.0,
         "amount": amount}
        for code, amount in [("000001", 300.0), ("000002", 200.0),
                             ("000003", 100.0)]
        for day in dates
    ])
    raw.loc[(raw["code"] == "000001") & (raw["date"] == dates[-1]), "volume"] = 0
    plan = prepare_universe(raw, dates[-2], dates[-1],
                            top=2, min_age=60)
    last_day = dates[-1]
    last = plan["eligible_pool"].query("date == @last_day")
    assert "000001" not in last["code"].tolist()
    assert len(last) == 1  # 不从第 3 名递补
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: `ModuleNotFoundError: intraday`。

- [ ] **Step 3: 实现准备逻辑**

```python
def load_daily_raw(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_pickle(path)


def prepare_universe(raw, start, end, top=500, adv_window=20, min_age=60):
    data = raw.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    amount = data.pivot(index="date", columns="code", values="amount").sort_index()
    volume = data.pivot(index="date", columns="code", values="volume").sort_index()
    adv = amount.rolling(adv_window, min_periods=adv_window).mean().shift(1)
    age = volume.fillna(0).gt(0).cumsum()
    eval_dates = amount.loc[pd.Timestamp(start):pd.Timestamp(end)].index
    rows, eligible = [], []
    for day in eval_dates:
        ranked = adv.loc[day].dropna().sort_values(ascending=False,
                                                   kind="mergesort")
        ranked = ranked.rename_axis("code").reset_index(name="adv20")
        ranked = ranked.sort_values(["adv20", "code"], ascending=[False, True]).head(top)
        ranked.insert(0, "date", day)
        ranked["liquidity_rank"] = np.arange(1, len(ranked) + 1)
        rows.append(ranked)
        ok = ranked[ranked["code"].map(age.loc[day]).fillna(0).ge(min_age)]
        ok = ok[ok["code"].map(volume.loc[day]).fillna(0).gt(0)]
        eligible.append(ok)
    ranked_pool = pd.concat(rows, ignore_index=True)
    eligible_pool = pd.concat(eligible, ignore_index=True)
    candidates = sorted(ranked_pool["code"].unique())
    first_pos = amount.index.get_loc(eval_dates[0])
    fetch_dates = amount.index[max(0, first_pos - adv_window):
                               amount.index.get_loc(eval_dates[-1]) + 1]
    estimated_rows = len(candidates) * len(fetch_dates) * 241
    return {"eval_dates": eval_dates, "fetch_dates": fetch_dates,
            "ranked_pool": ranked_pool, "eligible_pool": eligible_pool,
            "candidates": candidates, "estimated_rows": estimated_rows,
            "estimated_cells": estimated_rows * 3}
```

`load_daily_raw` 按后缀读取 pickle/parquet；`.gitignore` 追加 `intraday/cache/`。

- [ ] **Step 4: 运行数据测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add .gitignore intraday/__init__.py intraday/data.py intraday/tests
git commit -m "feat(intraday): prepare lagged liquidity universe"
```

---

### Task 3: 分钟数据清洗、覆盖率与原子分区

**Files:**
- Modify: `intraday/data.py`
- Modify: `intraday/tests/test_data.py`

**Interfaces:**
- Produces: `normalize_minute_day(frame, day, daily_amount) -> tuple[DataFrame, DataFrame]`
- Produces: `write_day_partition(frame, statuses, day, root) -> tuple[Path, Path]`
- Produces: `day_complete(day, codes, root) -> bool`

- [ ] **Step 1: 写质量门槛和完成清单测试**

```python
def test_normalize_minute_day_records_amount_mismatch():
    times = pd.date_range("2026-01-12 09:30", periods=200, freq="min")
    frame = pd.DataFrame({"thscode": "000001.SZ", "time": times,
                          "close": 10.0, "volume": 10.0, "amount": 100.0})
    clean, coverage = normalize_minute_day(
        frame, pd.Timestamp("2026-01-12"), pd.Series({"000001": 25000.0}))
    assert clean.empty
    assert coverage.loc[0, "reason"] == "amount_mismatch"


def test_partition_is_complete_with_explicit_no_data(tmp_path):
    day = pd.Timestamp("2026-01-12")
    frame = pd.DataFrame({"code": ["000001"], "time": [day], "close": [10.0],
                          "volume": [100.0], "amount": [1000.0]})
    statuses = {"000001": "returned", "000002": "no_data"}
    write_day_partition(frame, statuses, day, tmp_path)
    assert day_complete(day, ["000001", "000002"], tmp_path)
    assert not day_complete(day, ["000001", "000002", "000003"], tmp_path)
```

- [ ] **Step 2: 运行并确认函数缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: import error 指向新增函数。

- [ ] **Step 3: 实现清洗与原子写入**

```python
def normalize_minute_day(frame, day, daily_amount):
    if frame.empty:
        columns = ["code", "time", "close", "volume", "amount"]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=[
            "date", "code", "minute_count", "amount_relative_error", "reason"])
    data = frame.copy()
    data["code"] = data["thscode"].astype(str).str.slice(0, 6).str.zfill(6)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    for column in ["close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.drop_duplicates(["code", "time"], keep="last")
    valid_day = data["time"].dt.normalize().eq(pd.Timestamp(day).normalize())
    valid_time = valid_day & (((data["time"].dt.time >= time(9, 30)) &
                  (data["time"].dt.time <= time(11, 30))) | \
                 ((data["time"].dt.time >= time(13, 0)) &
                  (data["time"].dt.time <= time(15, 0))))
    data = data[valid_time & data["close"].gt(0) & data["volume"].ge(0) &
                data["amount"].ge(0)].sort_values(["code", "time"])
    kept, rows = [], []
    for code, group in data.groupby("code", sort=True):
        expected = float(daily_amount.get(code, np.nan))
        rel = abs(group["amount"].sum() - expected) / expected if expected > 0 else np.inf
        reason = "ok"
        if len(group) < 200:
            reason = "too_few_minutes"
        elif group["volume"].gt(0).sum() < 30:
            reason = "too_few_trades"
        elif rel > 0.02:
            reason = "amount_mismatch"
        if reason == "ok":
            kept.append(group)
        rows.append({"date": pd.Timestamp(day), "code": code,
                     "minute_count": len(group), "amount_relative_error": rel,
                     "reason": reason})
    clean = pd.concat(kept, ignore_index=True) if kept else data.iloc[0:0].copy()
    return clean, pd.DataFrame(rows)
```

`write_day_partition` 使用同目录 `.tmp` 文件、`Path.replace` 原子替换，并将
`{"date": ..., "statuses": ...}` 写成 JSON；`day_complete` 精确比较请求代码集
和允许状态集合 `{"returned", "no_data"}`。

```python
def _day_paths(day, root):
    stem = pd.Timestamp(day).strftime("%Y-%m-%d")
    return Path(root) / "minute" / f"{stem}.parquet", \
           Path(root) / "minute" / f"{stem}.json"


def write_day_partition(frame, statuses, day, root):
    parquet, manifest = _day_paths(day, root)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    temp_parquet, temp_manifest = parquet.with_suffix(".parquet.tmp"), \
                                  manifest.with_suffix(".json.tmp")
    frame.to_parquet(temp_parquet, index=False)
    temp_manifest.write_text(json.dumps({
        "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
        "statuses": dict(sorted(statuses.items())),
    }, ensure_ascii=False, sort_keys=True))
    temp_parquet.replace(parquet)
    temp_manifest.replace(manifest)
    return parquet, manifest


def day_complete(day, codes, root):
    parquet, manifest = _day_paths(day, root)
    if not parquet.exists() or not manifest.exists():
        return False
    payload = json.loads(manifest.read_text())
    statuses = payload.get("statuses", {})
    return set(statuses) == set(codes) and set(statuses.values()) <= {"returned", "no_data"}
```

- [ ] **Step 4: 运行数据测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/data.py intraday/tests/test_data.py
git commit -m "feat(intraday): validate and atomically cache minute days"
```

---

### Task 4: iFinD 下载编排、复权日线与时点属性

**Files:**
- Modify: `intraday/data.py`
- Modify: `intraday/tests/test_data.py`

**Interfaces:**
- Produces: `fetch_minute_partitions(plan, raw_daily, root, access_token, batch_size=200) -> DataFrame`
- Produces: `fetch_adjusted_daily(codes, start, end, access_token, batch_size=200) -> DataFrame`
- Produces: `build_attribute_query(day) -> str`
- Produces: `normalize_attributes(frame, day) -> DataFrame[date,code,name,float_cap,industry]`
- Produces: `fetch_attributes(anchor_dates, access_token) -> DataFrame`
- Produces: `apply_attribute_filters(eligible_pool, attributes, eval_dates) -> DataFrame`

- [ ] **Step 1: 写断点续传、重试、CPS 和动态列测试**

```python
def test_fetch_adjusted_daily_uses_cps3(monkeypatch):
    seen = []
    monkeypatch.setattr(data.ths_http, "history_quotation",
        lambda *args, **kwargs: seen.append(kwargs) or pd.DataFrame())
    fetch_adjusted_daily(["000001"], "2026-01-01", "2026-01-31", "token")
    assert seen[0]["functionpara"] == {"CPS": "3", "Fill": "Omit"}


def test_normalize_attributes_finds_dated_float_cap():
    raw = pd.DataFrame({"股票代码": ["000001.SZ"], "股票简称": ["平安银行"],
                        "a股市值(不含限售股)[20260112]": [1.2e11],
                        "所属同花顺行业": ["银行-股份制银行"]})
    out = normalize_attributes(raw, pd.Timestamp("2026-01-12"))
    assert out.loc[0, "code"] == "000001"
    assert out.loc[0, "float_cap"] == 1.2e11


def test_apply_attribute_filters_drops_st_without_replacement():
    dates = pd.bdate_range("2026-01-12", periods=3)
    pool = pd.DataFrame({"date": dates.repeat(2),
                         "code": ["000001", "000002"] * 3})
    attrs = pd.DataFrame({"date": [dates[0], dates[0]],
                          "code": ["000001", "000002"],
                          "name": ["平安银行", "ST测试"],
                          "float_cap": [1e11, 2e10], "industry": ["银行", "工业"]})
    result = apply_attribute_filters(pool, attrs, dates)
    assert result["code"].unique().tolist() == ["000001"]


def test_fetch_minute_skips_completed_day(monkeypatch, tmp_path):
    day = pd.Timestamp("2026-01-12")
    plan = {"candidates": ["000001"], "fetch_dates": pd.DatetimeIndex([day])}
    raw_daily = pd.DataFrame({"code": ["000001"], "date": [day],
                              "amount": [1000.0]})
    empty = pd.DataFrame(columns=["code", "time", "close", "volume", "amount"])
    write_day_partition(empty, {"000001": "no_data"}, day, tmp_path)
    called = []
    monkeypatch.setattr(data.ths_http, "high_frequency",
                        lambda *a, **k: called.append(a) or pd.DataFrame())
    fetch_minute_partitions(plan, raw_daily, tmp_path, "token")
    assert called == []


def test_retry_uses_fixed_backoff_then_recovers():
    calls, waits = [], []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.Timeout("temporary")
        return "ok"
    assert _retry(flaky, sleeper=waits.append) == "ok"
    assert len(calls) == 3
    assert waits == [1, 2]
```

- [ ] **Step 2: 运行并确认函数缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: 新接口 import 失败。

- [ ] **Step 3: 实现下载器**

```python
def _retry(call, waits=(1, 2, 4), sleeper=time.sleep):
    last = None
    for attempt in range(len(waits) + 1):
        try:
            return call()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            if attempt == len(waits):
                raise
            sleeper(waits[attempt])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 429 and (status is None or status < 500):
                raise
            last = exc
            if attempt == len(waits):
                raise
            sleeper(waits[attempt])
    raise last


def fetch_minute_partitions(plan, raw_daily, root, access_token, batch_size=200):
    coverage_rows = []
    daily = raw_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    for day in plan["fetch_dates"]:
        if day_complete(day, plan["candidates"], root):
            continue
        frames = []
        for batch in chunks(plan["candidates"], batch_size):
            thscodes = [to_thscode(code) for code in batch]
            frame = _retry(lambda: ths_http.high_frequency(
                thscodes, "close,volume,amount",
                f"{day:%Y-%m-%d} 09:30:00", f"{day:%Y-%m-%d} 15:00:00",
                functionpara={"CPS": "no", "Fill": "Original",
                              "Timeformat": "LocalTime", "Limitstart": "09:30:00",
                              "Limitend": "15:00:00"},
                access_token=access_token,
            ))
            frames.append(frame)
        joined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        returned = set(joined.get("thscode", pd.Series(dtype=str)).astype(str).str[:6])
        statuses = {code: ("returned" if code in returned else "no_data")
                    for code in plan["candidates"]}
        amounts = daily[daily["date"].eq(day)].set_index("code")["amount"]
        clean, coverage = normalize_minute_day(joined, day, amounts)
        write_day_partition(clean, statuses, day, root)
        coverage_rows.append(coverage)
    return pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
```

```python
def build_attribute_query(day):
    day = pd.Timestamp(day)
    prefix = f"{day.year}年{day.month}月{day.day}日"
    return f"{prefix}A股，{prefix}流通市值，所属同花顺行业"


def normalize_attributes(frame, day):
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    cap = next((c for c in frame.columns
                if stamp in str(c) and "市值" in str(c) and "限售" in str(c)), None)
    if cap is None:
        raise ValueError(f"missing dated float cap for {stamp}")
    result = pd.DataFrame({
        "date": pd.Timestamp(day),
        "code": frame["股票代码"].astype(str).str.extract(r"(\d{6})", expand=False),
        "name": frame["股票简称"].astype(str),
        "float_cap": pd.to_numeric(frame[cap], errors="coerce"),
        "industry": frame["所属同花顺行业"].astype(str),
    })
    return result.dropna(subset=["code"]).drop_duplicates("code")
```

`fetch_adjusted_daily` 使用 `functionpara={"CPS":"3","Fill":"Omit"}`，标准化为
`code,date,open,close`。`fetch_attributes` 每个锚点只请求一次并拼接缓存。

```python
def fetch_adjusted_daily(codes, start, end, access_token, batch_size=200):
    frames = []
    for batch in chunks(codes, batch_size):
        frame = ths_http.history_quotation(
            [to_thscode(code) for code in batch], "open,close", start, end,
            functionpara={"CPS": "3", "Fill": "Omit"},
            access_token=access_token,
        )
        if not frame.empty:
            frame["code"] = frame["thscode"].astype(str).str[:6]
            frame["date"] = pd.to_datetime(frame["time"]).dt.normalize()
            frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frames.append(frame[["code", "date", "open", "close"]])
    if not frames:
        raise RuntimeError("iFinD adjusted history returned no rows")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])


def fetch_attributes(anchor_dates, access_token):
    frames = []
    for day in anchor_dates:
        raw = ths_http.smart_stock_picking(build_attribute_query(day),
                                           access_token=access_token, timeout=90)
        frames.append(normalize_attributes(raw, day))
    return pd.concat(frames, ignore_index=True)


def apply_attribute_filters(eligible_pool, attributes, eval_dates):
    rows = []
    anchors = sorted(pd.to_datetime(attributes["date"].unique()))
    date_positions = {day: pos for pos, day in enumerate(pd.DatetimeIndex(eval_dates))}
    for day, members in eligible_pool.groupby("date", sort=True):
        prior = [anchor for anchor in anchors if anchor <= day]
        if not prior:
            continue
        anchor = prior[-1]
        if date_positions[day] - date_positions.get(anchor, date_positions[day]) > 4:
            continue
        dated = attributes[attributes["date"].eq(anchor)].copy()
        dated = dated[~dated["name"].astype(str).str.match(r"^\*?ST", case=False, na=False)]
        dated = dated[np.isfinite(dated["float_cap"]) & dated["float_cap"].gt(0)]
        kept = members[members["code"].isin(dated["code"])]
        rows.append(kept)
    return pd.concat(rows, ignore_index=True) if rows else eligible_pool.iloc[0:0].copy()
```

- [ ] **Step 4: 运行数据测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_data.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/data.py intraday/tests/test_data.py
git commit -m "feat(intraday): fetch resumable minute and point-in-time data"
```

---

### Task 5: 分钟数据压缩为日因子

**Files:**
- Create: `intraday/factors.py`
- Create: `intraday/tests/test_factors.py`

**Interfaces:**
- Produces: `minute_day_factors(frame: DataFrame) -> dict[str, float]`
- Produces: `factor_panels(partitions, codes, dates, window=20, min_periods=15) -> dict[str, DataFrame]`
- Keys: `rskew`, `cpv_mean`, `cpv_std`, `smart`

- [ ] **Step 1: 写公式与聪明钱 20% 边界测试**

```python
def test_minute_day_factors_match_formulas():
    close = np.exp(np.cumsum([0.0, 0.01, -0.02, 0.03]))
    frame = pd.DataFrame({"time": pd.date_range("2026-01-12 09:30", periods=4, freq="min"),
                          "close": close, "volume": [10.0, 20.0, 30.0, 40.0],
                          "amount": close * [10.0, 20.0, 30.0, 40.0]})
    result = minute_day_factors(frame)
    r = np.array([0.01, -0.02, 0.03])
    expected = np.sqrt(3) * (r ** 3).sum() / ((r ** 2).sum() ** 1.5)
    assert result["rskew_day"] == pytest.approx(expected)
    assert result["cpv_day"] == pytest.approx(frame.close.corr(frame.volume))


def test_smart_money_includes_threshold_crossing_minute():
    frame = pd.DataFrame({"time": pd.date_range("2026-01-12 09:30", periods=3, freq="min"),
                          "close": [10.0, 12.0, 12.1], "volume": [1.0, 80.0, 19.0],
                          "amount": [10.0, 960.0, 229.9]})
    result = minute_day_factors(frame)
    expected = 12.0 / (1199.9 / 100.0)
    assert result["smart_q_day"] == pytest.approx(expected)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_factors.py -q`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现日因子**

```python
def minute_day_factors(frame):
    data = frame.sort_values("time").copy()
    data["r"] = np.log(data["close"] / data["close"].shift(1))
    r = data["r"].dropna()
    ss = float((r ** 2).sum())
    rskew = np.sqrt(len(r)) * float((r ** 3).sum()) / ss ** 1.5 if ss > 0 else np.nan
    traded = data[data["volume"] > 0].copy()
    cpv = traded["close"].corr(traded["volume"])
    traded["smartness"] = traded["r"].abs() / np.sqrt(traded["volume"])
    ranked = traded.dropna(subset=["smartness"]).sort_values(
        ["smartness", "time"], ascending=[False, True])
    if ranked.empty or ranked["volume"].sum() <= 0:
        return {"rskew_day": rskew, "cpv_day": cpv, "smart_q_day": np.nan}
    target = 0.20 * ranked["volume"].sum()
    smart = ranked[ranked["volume"].cumsum().shift(fill_value=0) < target]
    vwap_smart = smart["amount"].sum() / smart["volume"].sum()
    vwap_all = traded["amount"].sum() / traded["volume"].sum()
    return {"rskew_day": rskew, "cpv_day": cpv,
            "smart_q_day": vwap_smart / vwap_all}
```

`factor_panels` 逐分区/代码调用上述函数，转成日期×代码面板；`rskew` 和 `smart`
取 20 日均值，`cpv_mean/cpv_std` 分别取 20 日均值/标准差，均使用
`min_periods=15`。

```python
def factor_panels(partitions, codes, dates, window=20, min_periods=15):
    rows = []
    for day, frame in partitions:
        for code, group in frame.groupby("code", sort=True):
            values = minute_day_factors(group)
            rows.append({"date": pd.Timestamp(day), "code": code, **values})
    daily = pd.DataFrame(rows)
    def panel(column):
        return daily.pivot(index="date", columns="code", values=column).reindex(
            index=pd.DatetimeIndex(dates), columns=codes)
    skew_day, cpv_day, smart_day = panel("rskew_day"), panel("cpv_day"), panel("smart_q_day")
    return {
        "rskew": skew_day.rolling(window, min_periods=min_periods).mean(),
        "cpv_mean": cpv_day.rolling(window, min_periods=min_periods).mean(),
        "cpv_std": cpv_day.rolling(window, min_periods=min_periods).std(ddof=1),
        "smart": smart_day.rolling(window, min_periods=min_periods).mean(),
    }
```

- [ ] **Step 4: 运行因子测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_factors.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/factors.py intraday/tests/test_factors.py
git commit -m "feat(intraday): compute daily skew CPV and smart-money factors"
```

---

### Task 6: 截面中性化与三逻辑块等权合成

**Files:**
- Create: `intraday/preprocess.py`
- Create: `intraday/tests/test_preprocess.py`

**Interfaces:**
- Produces: `winsorize_mad(values, n=5.0) -> Series`
- Produces: `neutralize_day(values, float_cap, industry, min_count=400) -> Series`
- Produces: `preprocess_panels(factors, pools, attributes, min_count=400) -> dict[str, DataFrame]`
- Produces: `compose(processed) -> dict`，新增 `cpv_block` 和 `score`

- [ ] **Step 1: 写极值、中性暴露和等权测试**

```python
def test_neutralize_removes_size_and_industry_exposure():
    idx = list("ABCDEF")
    cap = pd.Series([10, 20, 30, 10, 20, 30], index=idx, dtype=float)
    ind = pd.Series(["x", "x", "x", "y", "y", "y"], index=idx)
    values = np.log(cap) * 2 + ind.map({"x": 1.0, "y": -1.0})
    values = values + pd.Series([-.2, .1, .1, .2, -.1, -.1], index=idx)
    out = neutralize_day(values, cap, ind, min_count=6)
    assert out.corr(np.log(cap)) == pytest.approx(0.0, abs=1e-10)
    assert out.groupby(ind).mean().abs().max() < 1e-10
    assert out.std(ddof=0) == pytest.approx(1.0)


def test_compose_weights_three_logic_blocks_equally():
    f = pd.DataFrame([[1.0, -1.0]], columns=["A", "B"])
    processed = {"rskew": f, "cpv_mean": f, "cpv_std": f, "smart": -f}
    result = compose(processed)
    assert result["cpv_block"].iloc[0, 0] > 0
    assert result["score"].iloc[0, 0] == pytest.approx(1 / 3)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_preprocess.py -q`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现预处理**

```python
def winsorize_mad(values, n=5.0):
    med = values.median()
    mad = (values - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return values.copy()
    width = n * 1.4826 * mad
    return values.clip(med - width, med + width)


def _zscore(values):
    std = values.std(ddof=0)
    return (values - values.mean()) / std if std > 0 else values * np.nan


def neutralize_day(values, float_cap, industry, min_count=400):
    frame = pd.concat({"y": values, "cap": float_cap, "industry": industry}, axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["cap"] > 0]
    if len(frame) < min_count:
        return pd.Series(np.nan, index=values.index)
    y = _zscore(winsorize_mad(frame["y"]))
    dummies = pd.get_dummies(frame["industry"], drop_first=True, dtype=float)
    x = pd.concat([pd.Series(1.0, index=frame.index, name="const"),
                   np.log(frame["cap"]).rename("log_cap"), dummies], axis=1)
    beta = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0]
    residual = pd.Series(y.to_numpy() - x.to_numpy() @ beta, index=frame.index)
    result = pd.Series(np.nan, index=values.index)
    result.loc[residual.index] = _zscore(residual)
    return result
```

`preprocess_panels` 逐日期套用当日池和最多向后填充 4 日的属性；方向乘以 -1。
`compose` 先对 `cpv_mean/cpv_std` 等权并再次 z-score，再与 RSkew、Smart 等权。

```python
DIRECTIONS = {"rskew": -1.0, "cpv_mean": -1.0, "cpv_std": -1.0, "smart": -1.0}


def preprocess_panels(factors, pools, attributes, min_count=400):
    dates = sorted(set().union(*(frame.index for frame in factors.values())))
    codes = sorted(set().union(*(frame.columns for frame in factors.values())))
    attrs = attributes.sort_values(["date", "code"]).set_index(["date", "code"])
    pool_dates = pd.DatetimeIndex(sorted(pd.to_datetime(pools["date"].unique())))
    date_positions = {day: pos for pos, day in enumerate(pool_dates)}
    anchor_dates = sorted(pd.to_datetime(attributes["date"].unique()))
    results = {name: pd.DataFrame(np.nan, index=dates, columns=codes)
               for name in factors}
    for day in dates:
        members = pools.loc[pools["date"].eq(day), "code"]
        if len(members) < min_count:
            continue
        prior = [anchor for anchor in anchor_dates if anchor <= day]
        if not prior:
            continue
        anchor = prior[-1]
        if date_positions[day] - date_positions[anchor] > 4:
            continue
        latest = attrs.xs(anchor, level="date")
        for name, panel in factors.items():
            values = panel.loc[day].reindex(members) * DIRECTIONS[name]
            results[name].loc[day, members] = neutralize_day(
                values, latest["float_cap"], latest["industry"], min_count=min_count)
    return results


def compose(processed):
    cpv = (processed["cpv_mean"] + processed["cpv_std"]) / 2
    cpv = cpv.apply(_zscore, axis=1)
    score = (processed["rskew"] + cpv + processed["smart"]) / 3
    return {**processed, "cpv_block": cpv, "score": score}
```

- [ ] **Step 4: 运行预处理测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_preprocess.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/preprocess.py intraday/tests/test_preprocess.py
git commit -m "feat(intraday): neutralize and compose three factor blocks"
```

---

### Task 7: 5 日 RankIC、Newey-West 与五 cohort 分层

**Files:**
- Create: `intraday/evaluate.py`
- Create: `intraday/tests/test_evaluate.py`

**Interfaces:**
- Produces: `forward_open_return(open_prices, horizon=5) -> DataFrame`
- Produces: `newey_west_t(values, lags=4) -> float`
- Produces: `rank_ic(factor, forward, min_count=400) -> Series`
- Produces: `quantile_cohorts(factor, open_prices, q=5, horizon=5, min_count=400) -> DataFrame`
- Produces: `evaluate_factors(factors, open_prices, min_count=400) -> tuple[summary, daily_ic, quantiles]`

- [ ] **Step 1: 写时点、HAC 和分层测试**

```python
def test_forward_return_is_t1_to_t6():
    idx = pd.bdate_range("2026-01-01", periods=8)
    prices = pd.DataFrame({"A": np.arange(10.0, 18.0)}, index=idx)
    out = forward_open_return(prices, horizon=5)
    assert out.loc[idx[0], "A"] == pytest.approx(16.0 / 11.0 - 1)


def test_rank_ic_and_quantiles_are_monotonic():
    idx = pd.bdate_range("2026-01-01", periods=10)
    cols = list("ABCDE")
    factor = pd.DataFrame(np.tile(np.arange(5), (10, 1)), index=idx, columns=cols)
    opens = pd.DataFrame(100.0, index=idx, columns=cols)
    for i in range(1, len(idx)):
        opens.iloc[i] = opens.iloc[i - 1] * (1 + np.arange(5) * .001)
    fwd = forward_open_return(opens, horizon=2)
    ic = rank_ic(factor, fwd, min_count=5)
    assert ic.dropna().eq(1.0).all()
    qret = quantile_cohorts(factor, opens, q=5, horizon=2, min_count=5)
    assert qret.mean().is_monotonic_increasing
```

```python
def test_newey_west_matches_bartlett_definition():
    values = pd.Series([0.01, -0.02, 0.03, 0.01])
    x = values.to_numpy()
    d = x - x.mean()
    variance = d @ d / len(x)
    variance += 2 * 0.5 * (d[1:] @ d[:-1] / len(x))
    expected = x.mean() / np.sqrt(max(variance, 0) / len(x))
    assert newey_west_t(values, lags=1) == pytest.approx(expected)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_evaluate.py -q`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现统计核心**

```python
def forward_open_return(open_prices, horizon=5):
    return open_prices.shift(-(horizon + 1)) / open_prices.shift(-1) - 1.0


def newey_west_t(values, lags=4):
    x = pd.Series(values).dropna().to_numpy(float)
    n = len(x)
    if n < 2:
        return np.nan
    demeaned = x - x.mean()
    long_var = float(demeaned @ demeaned / n)
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(demeaned[lag:] @ demeaned[:-lag] / n)
        long_var += 2 * (1 - lag / (lags + 1)) * gamma
    se = np.sqrt(max(long_var, 0) / n)
    return float(x.mean() / se) if se > 0 else np.nan


def rank_ic(factor, forward, min_count=400):
    rows = {}
    for day in factor.index.intersection(forward.index):
        joined = pd.concat([factor.loc[day], forward.loc[day]], axis=1).dropna()
        if len(joined) >= min_count:
            rows[day] = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
    return pd.Series(rows, dtype=float).sort_index()
```

```python
def quantile_cohorts(factor, open_prices, q=5, horizon=5, min_count=400):
    daily = open_prices.pct_change()
    records = []
    for day in factor.index.intersection(open_prices.index):
        values = factor.loc[day].dropna()
        if len(values) < min_count:
            continue
        labels = pd.qcut(values.rank(method="first"), q, labels=False)
        entry = open_prices.index.get_loc(day) + 1
        exit_ = entry + horizon
        if exit_ >= len(open_prices.index):
            continue
        for ret_pos in range(entry + 1, exit_ + 1):
            returns = daily.iloc[ret_pos]
            for group in range(q):
                members = labels[labels == group].index
                records.append({"date": daily.index[ret_pos], "group": group,
                                "return": returns.reindex(members).mean()})
    rows = pd.DataFrame(records)
    return rows.pivot_table(index="date", columns="group", values="return",
                            aggfunc="mean").sort_index()
```

`evaluate_factors` 对每个因子调用上述接口，汇总 IC 均值/标准差/ICIR/正值率/
HAC t、Q5-Q1，以及五组编号对平均收益的 Spearman 单调性。

```python
def evaluate_factors(factors, open_prices, min_count=400):
    forward = forward_open_return(open_prices, horizon=5)
    summaries, daily_columns, quantile_rows = [], {}, []
    for name, factor in factors.items():
        ic = rank_ic(factor, forward, min_count=min_count)
        qret = quantile_cohorts(factor, open_prices, q=5, horizon=5,
                                min_count=min_count)
        daily_columns[name] = ic
        means = qret.mean()
        monotonicity = means.corr(pd.Series(means.index, index=means.index),
                                  method="spearman")
        summaries.append({"factor": name, "ic_mean": ic.mean(),
                          "ic_std": ic.std(ddof=1),
                          "icir": ic.mean() / ic.std(ddof=1),
                          "positive_ic_rate": ic.gt(0).mean(),
                          "ic_nw_t": newey_west_t(ic, lags=4),
                          "q5_q1_mean": means.iloc[-1] - means.iloc[0],
                          "monotonicity": monotonicity})
        long = qret.stack().rename("return").reset_index()
        long.insert(0, "factor", name)
        quantile_rows.append(long)
    return (pd.DataFrame(summaries), pd.DataFrame(daily_columns),
            pd.concat(quantile_rows, ignore_index=True))
```

- [ ] **Step 4: 运行评估测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_evaluate.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/evaluate.py intraday/tests/test_evaluate.py
git commit -m "feat(intraday): evaluate five-day IC and quantile cohorts"
```

---

### Task 8: T+1 开盘事件式组合回测

**Files:**
- Create: `intraday/portfolio.py`
- Create: `intraday/tests/test_portfolio.py`

**Interfaces:**
- Consumes: `intraday.evaluate.newey_west_t`
- Produces: `is_one_price_limit(day_row, previous_close) -> tuple[bool,bool]`
- Produces: `build_targets(score, pools, every=5, top_n=50) -> dict[Timestamp, Series]`
- Produces: `build_benchmark_targets(pools, signal_dates) -> dict[Timestamp, Series]`
- Produces: `simulate(targets, adjusted_open, raw_daily, cost_bps=20) -> dict`
- Result keys: `nav`, `returns`, `turnover`, `cost`, `trades`
- Produces: `portfolio_metrics(strategy, benchmark) -> dict`

- [ ] **Step 1: 写 T/T+1、成本、涨跌停和现金测试**

```python
def test_one_price_limit_direction():
    up = pd.Series({"open": 11, "high": 11, "low": 11, "close": 11})
    down = pd.Series({"open": 9, "high": 9, "low": 9, "close": 9})
    assert is_one_price_limit(up, 10) == (True, False)
    assert is_one_price_limit(down, 10) == (False, True)


def test_signal_trades_next_open_and_charges_cost():
    dates = pd.bdate_range("2026-01-01", periods=4)
    opens = pd.DataFrame({"A": [10, 10, 11, 11]}, index=dates)
    raw = make_raw_ohlc(opens)
    targets = {dates[0]: pd.Series({"A": 1.0})}
    result = simulate(targets, opens, raw, cost_bps=20)
    assert result["trades"].iloc[0]["date"] == dates[1]
    assert result["trades"].iloc[0]["cost"] > 0
    assert result["nav"].loc[dates[1]] < 1.0


def test_blocked_buy_stays_cash_and_blocked_sell_is_held():
    dates = pd.bdate_range("2026-01-01", periods=5)
    opens = pd.DataFrame({"A": [10, 11, 10, 9, 9]}, index=dates)
    raw = make_raw_ohlc(opens)
    raw.loc[(dates[1], "A"), ["high", "low", "close"]] = 11
    result = simulate({dates[0]: pd.Series({"A": 1.0})}, opens, raw, cost_bps=0)
    assert result["trades"].empty
    assert result["nav"].eq(1.0).all()


def make_raw_ohlc(opens):
    long = opens.stack().rename("open").to_frame()
    long.index.names = ["date", "code"]
    long["high"] = long["open"] * 1.01
    long["low"] = long["open"] * 0.99
    long["close"] = long["open"]
    return long


def test_blocked_sell_keeps_existing_position():
    dates = pd.bdate_range("2026-01-01", periods=5)
    opens = pd.DataFrame({"A": [10, 10, 9, 9, 9]}, index=dates)
    raw = make_raw_ohlc(opens)
    raw.loc[(dates[2], "A"), ["high", "low", "close"]] = 9
    targets = {dates[0]: pd.Series({"A": 1.0}), dates[1]: pd.Series(dtype=float)}
    result = simulate(targets, opens, raw, cost_bps=0)
    execution_day = dates[2]
    assert not result["trades"].query("date == @execution_day and side == 'sell'").shape[0]
    assert result["nav"].loc[dates[3]] == pytest.approx(.9)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_portfolio.py -q`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现事件式成交器**

```python
def is_one_price_limit(row, previous_close):
    prices = np.round([row["open"], row["high"], row["low"], row["close"]], 2)
    if not np.isfinite(prices).all() or len(set(prices)) != 1 or previous_close <= 0:
        return False, False
    move = prices[0] / previous_close - 1
    return move >= 0.045, move <= -0.045


def _blocked(raw_daily, dates, pos, day, code):
    if pos == 0 or (day, code) not in raw_daily.index:
        return False, False
    previous_key = (dates[pos - 1], code)
    if previous_key not in raw_daily.index:
        return False, False
    return is_one_price_limit(raw_daily.loc[(day, code)],
                              raw_daily.loc[previous_key, "close"])


def _trade_row(signal_date, day, code, side, shares, price, notional, cost):
    return {"signal_date": signal_date, "date": day, "code": code,
            "side": side, "shares": shares, "price": price,
            "notional": notional, "cost": cost, "status": "filled"}
```

`simulate` 为每个开盘日维护 `cash` 和合成可分割 `shares`：先以后复权开盘标记
NAV；若前一交易日有目标，先卖允许卖出的负缺口，再买允许买入的正缺口；买入
现金需求按 `notional * (1 + cost_rate)` 计算，不足时同比缩放；每笔保存
`signal_date,date,code,side,shares,price,notional,cost,status`。每日保存实际成交额/
开盘前 NAV 为单边换手。分别以 `cost_bps=0` 和 20 运行以得到毛/净结果。

`build_targets` 以第一个 score 覆盖至少 400 只的日期为锚，之后每 5 日取 Top50；
基准目标对同日 eligible pool 等权。`portfolio_metrics` 输出总/年化、Sharpe、
信息率、回撤、换手、月胜率和超额 Newey-West t。

```python
def build_targets(score, pools, every=5, top_n=50, min_count=400):
    targets, live = {}, score.notna().sum(axis=1).ge(min_count)
    dates = score.index[live]
    if dates.empty:
        return targets
    calendar = score.index[score.index.get_loc(dates[0])::every]
    for day in calendar:
        members = pools.loc[pools["date"].eq(day), "code"]
        ranked = score.loc[day].reindex(members).dropna().sort_index()
        ranked = ranked.sort_values(ascending=False, kind="mergesort").head(top_n)
        if not ranked.empty:
            targets[day] = pd.Series(1 / len(ranked), index=ranked.index)
    return targets


def build_benchmark_targets(pools, signal_dates):
    targets = {}
    for day in signal_dates:
        members = sorted(pools.loc[pools["date"].eq(day), "code"].unique())
        if members:
            targets[day] = pd.Series(1 / len(members), index=members)
    return targets
```

实现核心循环如下；`_blocked` 从 `raw_daily.loc[(day, code)]` 和前一交易日未复权
收盘调用 `is_one_price_limit`：

```python
def simulate(targets, adjusted_open, raw_daily, cost_bps=20):
    cost_rate = cost_bps / 1e4
    cash, shares, nav_rows, turnover_rows, cost_rows, trades = 1.0, {}, {}, {}, {}, []
    dates = adjusted_open.index
    valuation_open = adjusted_open.ffill()
    for pos, day in enumerate(dates):
        prices = adjusted_open.loc[day]
        marks = valuation_open.loc[day]
        nav_before = cash + sum(qty * marks.get(code, np.nan)
                                for code, qty in shares.items()
                                if np.isfinite(marks.get(code, np.nan)))
        traded_notional = day_cost = 0.0
        target = targets.get(dates[pos - 1]) if pos > 0 else None
        if target is not None:
            desired = target * nav_before
            codes = sorted(set(shares) | set(target.index))
            for code in codes:
                price = prices.get(code, np.nan)
                current = shares.get(code, 0.0) * price if np.isfinite(price) else np.nan
                need = desired.get(code, 0.0) - current if np.isfinite(current) else 0.0
                buy_blocked, sell_blocked = _blocked(raw_daily, dates, pos, day, code)
                if need < 0 and not sell_blocked:
                    notional = min(-need, current)
                    fee = notional * cost_rate
                    shares[code] = shares.get(code, 0.0) - notional / price
                    cash += notional - fee
                    traded_notional += notional
                    day_cost += fee
                    trades.append(_trade_row(dates[pos - 1], day, code, "sell",
                                             notional / price, price, notional, fee))
            buy_needs = {}
            for code in target.index:
                price = prices.get(code, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                buy_blocked, _ = _blocked(raw_daily, dates, pos, day, code)
                current = shares.get(code, 0.0) * price
                if not buy_blocked and desired[code] > current:
                    buy_needs[code] = desired[code] - current
            required = sum(buy_needs.values()) * (1 + cost_rate)
            scale = min(1.0, cash / required) if required > 0 else 0.0
            for code, need in sorted(buy_needs.items()):
                price, notional = prices[code], need * scale
                fee = notional * cost_rate
                shares[code] = shares.get(code, 0.0) + notional / price
                cash -= notional + fee
                traded_notional += notional
                day_cost += fee
                trades.append(_trade_row(dates[pos - 1], day, code, "buy",
                                         notional / price, price, notional, fee))
        nav_rows[day] = cash + sum(qty * marks.get(code, np.nan)
                                   for code, qty in shares.items()
                                   if np.isfinite(marks.get(code, np.nan)))
        turnover_rows[day] = traded_notional / nav_before if nav_before > 0 else np.nan
        cost_rows[day] = day_cost
    nav = pd.Series(nav_rows, name="nav")
    return {"nav": nav, "returns": nav.pct_change().fillna(0.0),
            "turnover": pd.Series(turnover_rows), "cost": pd.Series(cost_rows),
            "trades": pd.DataFrame(trades)}


def portfolio_metrics(strategy, benchmark):
    strat, bench = strategy["returns"].align(benchmark["returns"], join="inner")
    excess = strat - bench
    n = len(strat)
    strat_total = float((1 + strat).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    annual = lambda total: (1 + total) ** (252 / n) - 1 if n else np.nan
    equity = (1 + strat).cumprod()
    monthly = pd.DataFrame({"s": strat, "b": bench}).groupby(strat.index.to_period("M")) \
        .apply(lambda x: (1 + x).prod() - 1)
    return {"strategy_total": strat_total, "benchmark_total": bench_total,
            "strategy_annual": annual(strat_total), "benchmark_annual": annual(bench_total),
            "annual_excess": annual(strat_total) - annual(bench_total),
            "sharpe": strat.mean() / strat.std(ddof=0) * np.sqrt(252),
            "information_ratio": excess.mean() / excess.std(ddof=0) * np.sqrt(252),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "monthly_win_rate": float(monthly["s"].gt(monthly["b"]).mean()),
            "excess_nw_t": newey_west_t(excess, lags=4),
            "annual_turnover": strategy["turnover"].sum() * 252 / n}
```

- [ ] **Step 4: 运行组合测试**

Run: `.venv/bin/python -m pytest intraday/tests/test_portfolio.py -q`

Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add intraday/portfolio.py intraday/tests/test_portfolio.py
git commit -m "feat(intraday): backtest next-open portfolios with trade constraints"
```

---

### Task 9: 报告、CLI 与离线端到端测试

**Files:**
- Create: `intraday/report.py`
- Create: `intraday/run.py`
- Create: `intraday/tests/test_run.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `report.write_outputs(results, output_dir) -> list[Path]`
- Produces CLI: `python -m intraday.run prepare|fetch|validate|all`
- Produces: `run_prepare(args)`, `run_fetch(args)`, `run_validate(args)`

- [ ] **Step 1: 写完整产物和 CLI 参数测试**

```python
def test_write_outputs_creates_contract_files(tmp_path):
    tiny_results = {
        "factor_summary": pd.DataFrame({"factor": ["score"], "ic_mean": [.03]}),
        "daily_ic": pd.DataFrame({"date": ["2026-01-12"], "score": [.03]}),
        "quantile_returns": pd.DataFrame({0: [.0], 4: [.01]}),
        "portfolio_nav": pd.DataFrame({"strategy": [1.0], "benchmark": [1.0]}),
        "trades": pd.DataFrame({"date": ["2026-01-13"], "code": ["000001"]}),
        "data_coverage": pd.DataFrame({"date": ["2026-01-12"], "valid": [500]}),
        "portfolio_metrics": {"annual_excess": .01, "excess_nw_t": .5},
        "disclosures": ["六个月初步证据"],
    }
    paths = write_outputs(tiny_results, tmp_path)
    names = {path.name for path in paths}
    assert {"factor_summary.csv", "daily_ic.csv", "quantile_returns.csv",
            "portfolio_nav.csv", "trades.csv", "data_coverage.csv", "report.md",
            "factor_ic.png", "factor_quantiles.png", "portfolio_nav.png"} <= names
    assert "六个月初步证据" in (tmp_path / "report.md").read_text()


def test_parser_defaults_are_pinned():
    args = build_parser().parse_args(["prepare"])
    assert args.start == "2026-01-12"
    assert args.end == "2026-07-10"
    assert args.top == 500
```

加入一个 25 日、6 只股票、合成分钟分区的 `validate` 端到端测试；通过参数把
`min_count/top_n` 降至测试规模，核对 T+1 日期、20 bp 成本和全部 CSV 可解析。

- [ ] **Step 2: 运行并确认模块缺失**

Run: `.venv/bin/python -m pytest intraday/tests/test_run.py -q`

Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现报告与命令入口**

`build_parser()` 固定以下公共默认值：

```python
DEFAULT_START = "2026-01-12"
DEFAULT_END = "2026-07-10"
DEFAULT_WARMUP = "2025-12-11"
DEFAULT_DAILY = Path("alpha101/cache/ths_panel.pkl")
DEFAULT_CACHE = Path("intraday/cache")
DEFAULT_OUTPUT = Path("output/intraday_6m")
```

`prepare` 保存 `plan.json/ranked_pool.parquet/eligible_pool.parquet` 并打印日期数、
候选并集、预计行数和单元数；`fetch` 获取一次 access token，依次补属性、后复权
日线和分钟分区；`validate` 只读缓存计算因子、评估、策略/基准毛净回测并写报告；
`all` 顺序执行三者。任何缺少缓存、覆盖不足或未完成日期都非零退出。

`write_outputs` 使用 pandas 写 UTF-8 CSV，matplotlib `Agg` 写三张图；Markdown
固定披露日期、参数、覆盖、API 实际区间、ST 最多 4 日滞后、行业时点限制、
样本长度、成本、剔除统计，以及每个预设阈值是否达到。

```python
TABLE_FILES = {
    "factor_summary": "factor_summary.csv", "daily_ic": "daily_ic.csv",
    "quantile_returns": "quantile_returns.csv", "portfolio_nav": "portfolio_nav.csv",
    "trades": "trades.csv", "data_coverage": "data_coverage.csv",
}


def write_outputs(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, filename in TABLE_FILES.items():
        path = output / filename
        results[key].to_csv(path, index=True, encoding="utf-8-sig")
        paths.append(path)
    report = output / "report.md"
    metric_lines = [f"- {key}: {value}" for key, value in
                    sorted(results["portfolio_metrics"].items())]
    report.write_text("# A股分钟因子六个月验证\n\n## 组合指标\n\n" +
                      "\n".join(metric_lines) + "\n\n## 限制与披露\n\n" +
                      "\n".join(f"- {x}" for x in results["disclosures"]),
                      encoding="utf-8")
    paths.append(report)
    paths.extend(_write_figures(results, output))
    return paths


def _write_figures(results, output):
    paths = []
    figures = {
        "factor_ic.png": results["daily_ic"].cumsum(),
        "factor_quantiles.png": results["quantile_returns"],
        "portfolio_nav.png": results["portfolio_nav"],
    }
    for filename, frame in figures.items():
        ax = frame.plot(figsize=(10, 5), title=filename.removesuffix(".png"))
        fig = ax.get_figure()
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths
```

`build_parser` 的完整命令骨架：

```python
def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "fetch", "validate", "all"]:
        item = sub.add_parser(command)
        item.add_argument("--start", default=DEFAULT_START)
        item.add_argument("--end", default=DEFAULT_END)
        item.add_argument("--top", type=int, default=500)
        item.add_argument("--daily-cache", type=Path, default=DEFAULT_DAILY)
        item.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        item.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        item.add_argument("--top-n", type=int, default=50)
        item.add_argument("--rebalance", type=int, default=5)
        item.add_argument("--cost-bps", type=float, default=20.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    actions = {"prepare": run_prepare, "fetch": run_fetch, "validate": run_validate}
    if args.command == "all":
        run_prepare(args)
        run_fetch(args)
        run_validate(args)
    else:
        actions[args.command](args)
```

命令处理函数使用以下确定性数据流：

```python
def run_prepare(args):
    raw = data.load_daily_raw(args.daily_cache)
    plan = data.prepare_universe(raw, args.start, args.end, top=args.top)
    args.cache.mkdir(parents=True, exist_ok=True)
    plan["ranked_pool"].to_parquet(args.cache / "ranked_pool.parquet", index=False)
    plan["eligible_pool"].to_parquet(args.cache / "eligible_pool.parquet", index=False)
    payload = {"eval_dates": [str(x.date()) for x in plan["eval_dates"]],
               "fetch_dates": [str(x.date()) for x in plan["fetch_dates"]],
               "candidates": plan["candidates"],
               "estimated_rows": plan["estimated_rows"],
               "estimated_cells": plan["estimated_cells"]}
    (args.cache / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"evaluation days: {len(plan['eval_dates'])}")
    print(f"candidate union: {len(plan['candidates'])}")
    print(f"estimated cells: {plan['estimated_cells']}")
    return plan


def _load_plan(root):
    payload = json.loads((Path(root) / "plan.json").read_text())
    return {**payload,
            "eval_dates": pd.DatetimeIndex(payload["eval_dates"]),
            "fetch_dates": pd.DatetimeIndex(payload["fetch_dates"]),
            "ranked_pool": pd.read_parquet(Path(root) / "ranked_pool.parquet"),
            "eligible_pool": pd.read_parquet(Path(root) / "eligible_pool.parquet")}


def run_fetch(args):
    plan = _load_plan(args.cache)
    raw = data.load_daily_raw(args.daily_cache)
    token = ths_http.get_access_token()
    attributes_path = args.cache / "attributes.parquet"
    adjusted_path = args.cache / "adjusted_daily.parquet"
    if not attributes_path.exists():
        anchors = plan["eval_dates"][::args.rebalance]
        data.fetch_attributes(anchors, token).to_parquet(attributes_path, index=False)
    if not adjusted_path.exists():
        data.fetch_adjusted_daily(plan["candidates"], args.start, args.end, token) \
            .to_parquet(adjusted_path, index=False)
    coverage = data.fetch_minute_partitions(plan, raw, args.cache, token)
    if not coverage.empty:
        coverage_path = args.cache / "data_coverage.parquet"
        previous = pd.read_parquet(coverage_path) if coverage_path.exists() else coverage.iloc[0:0]
        pd.concat([previous, coverage], ignore_index=True) \
            .drop_duplicates(["date", "code"], keep="last") \
            .to_parquet(coverage_path, index=False)


def run_validate(args):
    plan = _load_plan(args.cache)
    raw = data.load_daily_raw(args.daily_cache)
    attributes = pd.read_parquet(args.cache / "attributes.parquet")
    pool = data.apply_attribute_filters(plan["eligible_pool"], attributes,
                                        plan["eval_dates"])
    partitions = [(day, pd.read_parquet(data._day_paths(day, args.cache)[0]))
                  for day in plan["fetch_dates"]]
    factor_data = factors.factor_panels(partitions, plan["candidates"],
                                        plan["fetch_dates"])
    processed = preprocess.preprocess_panels(factor_data, pool, attributes)
    scored = preprocess.compose(processed)
    scored = {name: frame.reindex(plan["eval_dates"])
              for name, frame in scored.items()}
    adjusted = pd.read_parquet(args.cache / "adjusted_daily.parquet")
    adjusted_open = adjusted.pivot(index="date", columns="code", values="open") \
        .reindex(plan["eval_dates"])
    summary, daily_ic, quantiles = evaluate.evaluate_factors(scored, adjusted_open)
    targets = portfolio.build_targets(scored["score"], pool, every=args.rebalance,
                                      top_n=args.top_n)
    benchmark_targets = portfolio.build_benchmark_targets(pool, targets.keys())
    raw_daily = raw.assign(date=pd.to_datetime(raw["date"])) \
        .set_index(["date", "code"])[["open", "high", "low", "close"]].sort_index()
    strategy_net = portfolio.simulate(targets, adjusted_open, raw_daily, args.cost_bps)
    strategy_gross = portfolio.simulate(targets, adjusted_open, raw_daily, 0)
    benchmark_net = portfolio.simulate(benchmark_targets, adjusted_open, raw_daily,
                                       args.cost_bps)
    benchmark_gross = portfolio.simulate(benchmark_targets, adjusted_open, raw_daily, 0)
    metrics = portfolio.portfolio_metrics(strategy_net, benchmark_net)
    nav = pd.concat({"strategy_net": strategy_net["nav"],
                     "strategy_gross": strategy_gross["nav"],
                     "benchmark_net": benchmark_net["nav"],
                     "benchmark_gross": benchmark_gross["nav"]}, axis=1)
    trades = pd.concat([strategy_net["trades"].assign(portfolio="strategy"),
                        benchmark_net["trades"].assign(portfolio="benchmark")],
                       ignore_index=True)
    minute_coverage = pd.read_parquet(args.cache / "data_coverage.parquet") \
        .assign(record_type="minute")
    pool_counts = plan["ranked_pool"].groupby("date").size().rename("ranked_count").to_frame()
    pool_counts["daily_eligible_count"] = plan["eligible_pool"].groupby("date").size()
    pool_counts["final_count"] = pool.groupby("date").size()
    pool_counts = pool_counts.reset_index().assign(record_type="pool")
    coverage = pd.concat([minute_coverage, pool_counts], ignore_index=True, sort=False)
    score_stats = summary.set_index("factor").loc["score"]
    threshold_lines = [
        f"综合 RankIC >= 0.03: {'达到' if score_stats['ic_mean'] >= .03 else '未达到'}",
        f"五组单调性 >= 0.8: {'达到' if score_stats['monotonicity'] >= .8 else '未达到'}",
        f"扣费后累计超额 > 0: {'达到' if metrics['strategy_total'] > metrics['benchmark_total'] else '未达到'}",
    ]
    results = {"factor_summary": summary, "daily_ic": daily_ic,
               "quantile_returns": quantiles, "portfolio_nav": nav,
               "trades": trades, "data_coverage": coverage,
               "portfolio_metrics": metrics,
               "disclosures": ["六个月初步证据",
                               f"实际验证区间 {args.start} 至 {args.end}",
                               f"有效 RankIC 样本日 {daily_ic['score'].notna().sum()}",
                               "ST 状态最多滞后 4 个交易日",
                               "行业列可能是当前分类口径",
                               f"单边实际成交成本 {args.cost_bps} bp",
                               *threshold_lines]}
    return report.write_outputs(results, args.output)
```

- [ ] **Step 4: 运行新增包和全仓测试**

Run: `.venv/bin/python -m pytest intraday/tests alpha101/tests strategies/tests -q`

Expected: 全部通过，无 warning/error traceback。

- [ ] **Step 5: 提交**

```bash
git add intraday/report.py intraday/run.py intraday/tests/test_run.py README.md
git commit -m "feat(intraday): add reproducible validation CLI and report"
```

---

### Task 10: 真实数据准备、下载与六个月验证

**Files:**
- Create locally/ignored: `intraday/cache/**`
- Create: `output/intraday_6m/**`

**Interfaces:**
- Consumes: 完整 `intraday.run` CLI
- Produces: 设计规格约定的真实数据报告和审计表

- [ ] **Step 1: 运行本地准备并核对固定规模**

Run:

```bash
.venv/bin/python -m intraday.run prepare \
  --start 2026-01-12 --end 2026-07-10 --top 500
```

Expected: `119` 个验证日、`1104` 只候选并集、约 `1.11e8` 个分钟行情单元；若
现有固定缓存产生不同结果，停止并在报告中解释数据文件差异，不静默修改日期。

- [ ] **Step 2: 加载令牌并启动可恢复下载**

Run:

```bash
set -a
source .env
set +a
.venv/bin/python -m intraday.run fetch \
  --start 2026-01-12 --end 2026-07-10 --top 500
```

Expected: 每个交易日打印完成/跳过/无数据数量；中断后原命令可安全重跑，不重复
正式分区。不得打印 refresh/access token。

- [ ] **Step 3: 检查下载完整性后运行验证**

Run:

```bash
.venv/bin/python -m intraday.run validate \
  --start 2026-01-12 --end 2026-07-10 --top 500 \
  --top-n 50 --rebalance 5 --cost-bps 20
```

Expected: `output/intraday_6m/` 中十项约定产物全部存在且非空；报告不含
`TBD/TODO/NaN` 形式的未解释占位。

- [ ] **Step 4: 对产物做机器校验**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("output/intraday_6m")
for name in ["factor_summary.csv", "daily_ic.csv", "quantile_returns.csv",
             "portfolio_nav.csv", "trades.csv", "data_coverage.csv"]:
    frame = pd.read_csv(root / name)
    assert len(frame) > 0, name
for name in ["report.md", "factor_ic.png", "factor_quantiles.png", "portfolio_nav.png"]:
    assert (root / name).stat().st_size > 0, name
print("artifacts ok")
PY
```

Expected: `artifacts ok`。

- [ ] **Step 5: 运行最终验证**

Run:

```bash
.venv/bin/python -m pytest alpha101/tests strategies/tests intraday/tests -q
.venv/bin/python -m compileall -q alpha101 strategies intraday
git diff --check
git status --short
```

Expected: pytest 零失败；compileall 和 diff check 退出码 0；状态只包含预期报告
产物，不包含 `.env`、token、分钟 Parquet 或临时文件。

- [ ] **Step 6: 提交可复现的小型研究产物**

```bash
git add output/intraday_6m README.md
git commit -m "research: validate A-share intraday factors over six months"
```

提交前人工确认报告如实写明每个阈值是否达到，不因结果好坏修改固定参数。
