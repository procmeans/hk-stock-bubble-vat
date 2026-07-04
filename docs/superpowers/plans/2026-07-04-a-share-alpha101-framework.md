# A股 101-Alpha 基础多因子框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在全 A 股上实现《101 Formulaic Alphas》的 82 个纯价量因子,提供因子回测评估(IC/分层多空)与每日选股清单。

**Architecture:** 纯 pandas 管线,不引入 qlib。数据组织为"字段面板"(`dict[str, DataFrame[日期×股票代码]]`)。`operators.py` 实现论文全部算子;`alphas.py` 用算子拼出 82 个因子;`backtest.py`/`compose.py`/`select.py` 做评估与选股。所有代码在新目录 `alpha101/`,不改动现有可视化文件。

**Tech Stack:** Python 3 + pandas + numpy + akshare + matplotlib + pyarrow;pytest 测试;本地 venv;parquet 缓存。

## Global Constraints

- 所有新代码位于 `alpha101/`,**禁止修改**现有文件(`fetch_*.py`、`index.html`、`data/`、根 `requirements.txt` 等)。
- 依赖装在 `alpha101/.venv`(系统 Python 为 PEP-668 锁定,不能全局 pip)。所有 python/pytest 命令用 `alpha101/.venv/bin/python` 与 `alpha101/.venv/bin/pytest`。
- 因子实现的**纯价量子集 = 82 个**:1–47, 49–55, 57, 60–62, 64–66, 68, 71–75, 77–78, 81, 83–86, 88, 92, 94–96, 98–99, 101(排除用 `IndNeutralize` 的 18 个与用 `cap` 的 #56)。
- **字段面板**约定:`dict[str, pd.DataFrame]`,每个 DataFrame `index` 为升序 `DatetimeIndex`(交易日),`columns` 为股票代码(str)。字段键:`open,high,low,close,volume,amount,vwap,returns`。
- **算子约定**:所有算子输入/输出均为 `DataFrame[日期×股票]`;截面算子按行(单日,`axis=1`)运算,时序算子按列(单股)滚动。窗口天数 `d` 若为浮点一律 `int(np.floor(d))`。
- **前复权**价计算因子;`vwap = amount / volume`(A 股日线无原生 vwap)。
- **防未来函数**:T 日因子预测 T+1 收益(默认收盘对齐,`fwd_ret = close.shift(-1)/close - 1`)。

---

### Task 1: 项目骨架 + venv + operators 基础算子(截面 + 逐元素)

**Files:**
- Create: `alpha101/__init__.py`
- Create: `alpha101/requirements.txt`
- Create: `alpha101/operators.py`
- Create: `alpha101/tests/__init__.py`
- Create: `alpha101/tests/test_operators.py`
- Create: `alpha101/pytest.ini`

**Interfaces:**
- Produces: 逐元素/截面算子,全部 `DataFrame -> DataFrame`:
  - `rank(x)` 截面百分位排名 `[0,1]`
  - `scale(x, a=1.0)` 截面缩放使每行 `sum(abs)=a`
  - `signedpower(x, a)` = `sign(x)*abs(x)**a`
  - `abs_(x)`, `log_(x)`, `sign_(x)` 逐元素
  - `rank_sub(x, y)`、`ew_min(x, y)`、`ew_max(x, y)` 逐元素二元(见 gotcha)

- [ ] **Step 1: 建 venv 并装依赖**

`alpha101/requirements.txt` 内容:
```
pandas
numpy
akshare
matplotlib
pyarrow
pytest
```

Run:
```bash
cd /Users/procmeans/Tools/hk-stock-bubble-vat
python3 -m venv alpha101/.venv
alpha101/.venv/bin/pip install --quiet -r alpha101/requirements.txt
alpha101/.venv/bin/python -c "import pandas,numpy,akshare,matplotlib,pyarrow,pytest;print('ok')"
```
Expected: 打印 `ok`

- [ ] **Step 2: 建空包文件与 pytest 配置**

`alpha101/__init__.py`:(空文件)
`alpha101/tests/__init__.py`:(空文件)
`alpha101/pytest.ini`:
```ini
[pytest]
testpaths = alpha101/tests
```

- [ ] **Step 3: 写失败测试** `alpha101/tests/test_operators.py`

```python
import numpy as np
import pandas as pd
import pytest
from alpha101 import operators as op


def _df(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx, columns=["A", "B", "C"])


def test_rank_is_cross_sectional_pct():
    df = _df([[1.0, 2.0, 3.0]])
    assert df.pipe(op.rank).iloc[0].tolist() == pytest.approx([1/3, 2/3, 1.0])


def test_scale_sum_abs_equals_a():
    df = _df([[1.0, -3.0, 0.0]])
    assert op.scale(df, a=1.0).iloc[0].abs().sum() == pytest.approx(1.0)


def test_signedpower_keeps_sign():
    df = _df([[-4.0, 9.0, 2.0]])
    out = op.signedpower(df, 0.5).iloc[0]
    assert out.tolist() == pytest.approx([-2.0, 3.0, np.sqrt(2)])


def test_ew_max_elementwise():
    x = _df([[1.0, 5.0, 3.0]])
    y = _df([[4.0, 2.0, 3.0]])
    assert op.ew_max(x, y).iloc[0].tolist() == [4.0, 5.0, 3.0]
```

- [ ] **Step 4: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_operators.py -x -q`
Expected: FAIL(`ModuleNotFoundError: No module named 'alpha101.operators'`)

- [ ] **Step 5: 实现** `alpha101/operators.py`(本任务部分)

```python
"""论文《101 Formulaic Alphas》算子实现。所有算子作用于 DataFrame[日期×股票]。"""
import numpy as np
import pandas as pd


def _d(d):
    """窗口天数:浮点向下取整为 int。"""
    return int(np.floor(d))


# ---- 逐元素 ----
def abs_(x):
    return x.abs()


def log_(x):
    return np.log(x)


def sign_(x):
    return np.sign(x)


def signedpower(x, a):
    return np.sign(x) * (x.abs() ** a)


# ---- 截面 ----
def rank(x):
    """截面百分位排名 [0,1](按行)。"""
    return x.rank(axis=1, pct=True)


def scale(x, a=1.0):
    """截面缩放,使每行 sum(abs)=a。"""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.mul(a).div(denom, axis=0)


# ---- 逐元素二元(见 alphas 里的 min/max/rank 相减)----
def rank_sub(x, y):
    return x - y


def ew_min(x, y):
    """逐元素最小(两个 DataFrame)。"""
    return pd.DataFrame(np.minimum(x.values, y.values), index=x.index, columns=x.columns)


def ew_max(x, y):
    return pd.DataFrame(np.maximum(x.values, y.values), index=x.index, columns=x.columns)
```

- [ ] **Step 6: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_operators.py -q`
Expected: PASS(4 passed)

- [ ] **Step 7: 提交**

```bash
git add alpha101/__init__.py alpha101/requirements.txt alpha101/operators.py alpha101/tests alpha101/pytest.ini
git commit -m "feat(alpha101): 项目骨架 + 逐元素/截面算子"
```

---

### Task 2: operators 时序算子

**Files:**
- Modify: `alpha101/operators.py`(追加时序算子)
- Modify: `alpha101/tests/test_operators.py`(追加测试)

**Interfaces:**
- Consumes: `_d`
- Produces(全部 `DataFrame -> DataFrame`,按列滚动窗口 `d`):
  - `delay(x, d)`、`delta(x, d)`
  - `ts_sum(x, d)`、`ts_product(x, d)`、`ts_stddev(x, d)`
  - `ts_min(x, d)`、`ts_max(x, d)`、`ts_argmin(x, d)`、`ts_argmax(x, d)`
  - `ts_rank(x, d)`
  - `correlation(x, y, d)`、`covariance(x, y, d)`
  - `decay_linear(x, d)`
  - 别名 `ts_min`→`min_`、`ts_max`→`max_`(供 alphas 里 `min(x,d)`/`max(x,d)` 用)

- [ ] **Step 1: 写失败测试**(追加到 test_operators.py)

```python
def test_delta():
    df = _df([[1.0], [3.0], [7.0]])[["A"]]
    assert op.delta(df, 1)["A"].tolist()[1:] == [2.0, 4.0]


def test_ts_argmax_position():
    # 窗口内最大值出现在"几天前":最新一天为 d-1,最早为 0
    s = pd.DataFrame({"A": [1.0, 9.0, 2.0, 3.0]},
                     index=pd.date_range("2020-01-01", periods=4))
    out = op.ts_argmax(s, 3)
    # 末窗口 [9,2,3] 最大在第 0 位 -> argmax 索引 0
    assert out["A"].iloc[-1] == 0


def test_ts_rank_last_value_rank():
    s = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]},
                     index=pd.date_range("2020-01-01", periods=4))
    out = op.ts_rank(s, 4)
    # 最后一个值是窗口内最大 -> pct rank = 1.0
    assert out["A"].iloc[-1] == pytest.approx(1.0)


def test_decay_linear_weights():
    s = pd.DataFrame({"A": [1.0, 2.0, 3.0]},
                     index=pd.date_range("2020-01-01", periods=3))
    out = op.decay_linear(s, 3)
    # 权重 3,2,1 归一化 -> (1*1 + 2*2 + 3*3)/6 = 14/6
    assert out["A"].iloc[-1] == pytest.approx(14/6)


def test_correlation_range():
    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=pd.date_range("2020-01-01", periods=5))
    y = pd.DataFrame({"A": [2.0, 4, 6, 8, 10]}, index=pd.date_range("2020-01-01", periods=5))
    assert op.correlation(x, y, 5)["A"].iloc[-1] == pytest.approx(1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_operators.py -q`
Expected: FAIL(`AttributeError: module has no attribute 'delta'` 等)

- [ ] **Step 3: 实现时序算子**(追加到 operators.py)

```python
def delay(x, d):
    return x.shift(_d(d))


def delta(x, d):
    return x - x.shift(_d(d))


def ts_sum(x, d):
    return x.rolling(_d(d)).sum()


def ts_product(x, d):
    return x.rolling(_d(d)).apply(np.prod, raw=True)


def ts_stddev(x, d):
    return x.rolling(_d(d)).std()


def ts_min(x, d):
    return x.rolling(_d(d)).min()


def ts_max(x, d):
    return x.rolling(_d(d)).max()


def ts_argmin(x, d):
    return x.rolling(_d(d)).apply(np.argmin, raw=True)


def ts_argmax(x, d):
    return x.rolling(_d(d)).apply(np.argmax, raw=True)


def ts_rank(x, d):
    def _r(w):
        return pd.Series(w).rank(pct=True).iloc[-1]
    return x.rolling(_d(d)).apply(_r, raw=True)


def decay_linear(x, d):
    d = _d(d)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d).apply(lambda a: np.dot(a, w), raw=True)


def correlation(x, y, d):
    return x.rolling(_d(d)).corr(y)


def covariance(x, y, d):
    return x.rolling(_d(d)).cov(y)


# 别名:论文中 min(x,d)/max(x,d) 即时序 min/max
min_ = ts_min
max_ = ts_max
```

- [ ] **Step 4: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_operators.py -q`
Expected: PASS(全部通过)

- [ ] **Step 5: 提交**

```bash
git add alpha101/operators.py alpha101/tests/test_operators.py
git commit -m "feat(alpha101): 时序算子(ts_*, correlation, decay_linear 等)"
```

---

### Task 3: 数据层 data.py(拉取/缓存/构建字段面板)

**Files:**
- Create: `alpha101/data.py`
- Create: `alpha101/tests/test_data.py`

**Interfaces:**
- Produces:
  - `build_panel(raw: pd.DataFrame) -> dict[str, pd.DataFrame]` — 纯函数,把长表(列:`code,date,open,high,low,close,volume,amount`)转成字段面板,并派生 `vwap=amount/volume`、`returns=close.pct_change()`。
  - `adv(panel, d) -> pd.DataFrame` — 过去 d 日 `amount` 均值。
  - `fetch_all(years=5, cache="alpha101/cache/panel.parquet") -> dict[str, pd.DataFrame]` — 用 akshare 逐只拉前复权日线,存/读 parquet。
  - `load_panel(cache="alpha101/cache/panel.parquet") -> dict[str, pd.DataFrame]` — 读缓存。

- [ ] **Step 1: 写失败测试** `alpha101/tests/test_data.py`(只测纯函数,不联网)

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_data.py -q`
Expected: FAIL(`ModuleNotFoundError`/`AttributeError`)

- [ ] **Step 3: 实现** `alpha101/data.py`

```python
"""A股前复权日线数据:拉取、缓存、构建字段面板。"""
import os
import time
import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
CACHE = "alpha101/cache/panel.parquet"


def build_panel(raw):
    """长表(code,date,open,high,low,close,volume,amount) -> 字段面板 dict。"""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    panel = {}
    for f in FIELDS:
        panel[f] = raw.pivot(index="date", columns="code", values=f).sort_index()
    panel["vwap"] = panel["amount"] / panel["volume"].replace(0, np.nan)
    panel["returns"] = panel["close"].pct_change()
    return panel


def adv(panel, d):
    return panel["amount"].rolling(int(d)).mean()


def _fetch_one(code, start, end):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close", "成交量": "volume",
                            "成交额": "amount"})
    df["code"] = code
    return df[["code", "date"] + FIELDS]


def fetch_all(years=5, cache=CACHE, sleep=0.05):
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    codes = ak.stock_zh_a_spot_em()["代码"].astype(str).tolist()
    frames = []
    for i, code in enumerate(codes):
        try:
            one = _fetch_one(code, start, end)
            if one is not None:
                frames.append(one)
        except Exception as e:
            print(f"skip {code}: {e}", flush=True)
        if i % 200 == 0:
            print(f"{i}/{len(codes)}", flush=True)
        time.sleep(sleep)
    raw = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    raw.to_parquet(cache)
    return build_panel(raw)


def load_panel(cache=CACHE):
    raw = pd.read_parquet(cache)
    return build_panel(raw)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_data.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add alpha101/data.py alpha101/tests/test_data.py
git commit -m "feat(alpha101): 数据层 build_panel/adv/fetch_all(前复权日线缓存)"
```

---

### Task 4: 股票池过滤 universe.py

**Files:**
- Create: `alpha101/universe.py`
- Create: `alpha101/tests/test_universe.py`

**Interfaces:**
- Consumes: 字段面板。
- Produces:
  - `liquidity_mask(panel, min_days=60, adv_window=20, drop_pct=0.20) -> pd.DataFrame[bool]` — 每日有效票 mask:剔上市 < `min_days`(该股在面板中前 `min_days` 行置 False)、当日停牌(`volume` 为 0 或 NaN)、过去 `adv_window` 日均额截面后 `drop_pct` 分位。
  - `apply_mask(factor, mask) -> pd.DataFrame` — 把 mask 外的因子值置 NaN。
- 说明:ST 过滤需名称数据,当前 data.py 不含名称;本任务 ST 过滤先以 `st_codes: set[str]` 参数注入(默认空),留 `run.py` 传入;不在此处联网。

- [ ] **Step 1: 写失败测试** `alpha101/tests/test_universe.py`

```python
import numpy as np
import pandas as pd
from alpha101 import universe


def _panel():
    idx = pd.date_range("2020-01-01", periods=5)
    cols = ["A", "B"]
    vol = pd.DataFrame([[100, 100], [100, 0], [100, 100], [100, 100], [100, 100]],
                       index=idx, columns=cols, dtype=float)
    amt = pd.DataFrame(1e8, index=idx, columns=cols)
    amt["B"] = 1.0  # B 流动性极低
    return {"volume": vol, "amount": amt}


def test_suspended_day_is_false():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=0, adv_window=1, drop_pct=0.0)
    assert m.loc["2020-01-02", "B"] == False  # volume==0 当日停牌


def test_new_listing_masked():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=3, adv_window=1, drop_pct=0.0)
    assert m.iloc[:3]["A"].any() == False  # 前3行次新


def test_low_liquidity_dropped():
    p = _panel()
    m = universe.liquidity_mask(p, min_days=0, adv_window=1, drop_pct=0.5)
    assert m.loc["2020-01-03", "B"] == False  # B 均额在后 50%
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_universe.py -q`
Expected: FAIL

- [ ] **Step 3: 实现** `alpha101/universe.py`

```python
"""每日有效股票池 mask。"""
import numpy as np
import pandas as pd


def liquidity_mask(panel, min_days=60, adv_window=20, drop_pct=0.20, st_codes=None):
    vol = panel["volume"]
    amt = panel["amount"]
    mask = pd.DataFrame(True, index=vol.index, columns=vol.columns)

    # 当日停牌:volume 为 0 或 NaN
    mask &= vol.fillna(0) > 0

    # 次新:每只股票前 min_days 个有效交易日置 False
    if min_days > 0:
        traded = vol.fillna(0) > 0
        age = traded.cumsum()
        mask &= age > min_days

    # 流动性:过去 adv_window 日均额,截面后 drop_pct 分位剔除
    if drop_pct > 0:
        advw = amt.rolling(adv_window, min_periods=1).mean()
        thresh = advw.quantile(drop_pct, axis=1)
        mask &= advw.ge(thresh, axis=0)

    # ST
    if st_codes:
        for c in st_codes:
            if c in mask.columns:
                mask[c] = False

    return mask


def apply_mask(factor, mask):
    return factor.where(mask.reindex_like(factor).fillna(False))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_universe.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add alpha101/universe.py alpha101/tests/test_universe.py
git commit -m "feat(alpha101): 每日票池过滤(停牌/次新/流动性/ST)"
```

---

### Task 5: alphas.py — 因子公式(翻译约定 + 全部 82 个 + 冒烟测试)

**Files:**
- Create: `alpha101/alphas.py`
- Create: `alpha101/tests/test_alphas.py`

**Interfaces:**
- Consumes: `operators` 全部算子;`data.adv`。
- Produces:
  - 每个 `alpha_<N>(P) -> pd.DataFrame`,`P` 为字段面板 dict。仅纯计算,无 I/O。
  - `ALPHAS: dict[int, callable]` — 编号→函数,含全部 82 个。
  - `compute_all(P) -> dict[int, pd.DataFrame]`。

**翻译约定(把论文公式逐字转成本模块代码):**
- `rank(x)` → `op.rank(x)`;`Ts_Rank(x,d)`/`ts_rank` → `op.ts_rank(x,d)`。
- `delay/delta/correlation/covariance/scale/decay_linear/signedpower/sum/product/stddev/ts_min/ts_max/ts_argmin/ts_argmax` → `op.` 同名(`sum`→`op.ts_sum`,`stddev`→`op.ts_stddev`,`product`→`op.ts_product`)。
- `abs/log/sign` → `op.abs_/op.log_/op.sign_`。
- 字段 `open/high/low/close/volume/vwap/returns` → `P["..."]`;`adv{n}` → `data.adv(P, n)`。
- 常量天数浮点(如 `decay_linear(x, 2.65461)`)照传,`op._d` 会 `floor`。
- **min/max 的二义性(关键 gotcha)**:论文把 `min(x,d)/max(x,d)` 定义为时序算子;但当两个参数都是"序列表达式"(如 `max(rank(...), Ts_Rank(...))`)时是**逐元素**二元 max/min → 用 `op.ew_max/op.ew_min`。判据:第二参数是**整数/浮点常量** → 时序 `op.ts_max/op.ts_min`;第二参数是**表达式** → `op.ew_max/op.ew_min`。
- 布尔比较(`<`,`>`,`==`)在算术里当 0/1:用 `(a < b).astype(float)`。
- `x^y`(如 Alpha#94 的 `rank(...)^Ts_Rank(...)`)→ `op.signedpower` 不适用(指数是序列),用 `a ** b`(pandas 逐元素幂)。

- [ ] **Step 1: 写失败测试(5 个代表因子手工核对 + 全量冒烟)** `alpha101/tests/test_alphas.py`

```python
import numpy as np
import pandas as pd
import pytest
from alpha101 import alphas


def _panel(n=60, m=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = [f"{i:06d}" for i in range(m)]
    def rand_pos():
        return pd.DataFrame(rng.uniform(5, 50, size=(n, m)), index=idx, columns=cols)
    close = rand_pos()
    P = {"open": rand_pos(), "high": rand_pos(), "low": rand_pos(),
         "close": close, "volume": rand_pos() * 1000, "amount": rand_pos() * 1e6}
    P["vwap"] = P["amount"] / P["volume"]
    P["returns"] = P["close"].pct_change()
    return P


def test_alpha101_formula():
    # Alpha#101 = (close - open) / ((high - low) + .001)
    P = _panel()
    out = alphas.alpha_101(P)
    expected = (P["close"] - P["open"]) / ((P["high"] - P["low"]) + 0.001)
    pd.testing.assert_frame_equal(out, expected)


def test_alpha1_shape_and_range():
    # Alpha#1 是 rank 结果,范围应在 (0,1]
    P = _panel()
    out = alphas.alpha_1(P)
    vals = out.dropna(how="all").stack()
    assert vals.between(0, 1).all()


@pytest.mark.parametrize("n", sorted(alphas.ALPHAS))
def test_alpha_smoke_not_all_nan(n):
    # 每个因子在充足历史后应有非 NaN、非常数输出
    P = _panel(n=80, m=12, seed=n)
    out = alphas.ALPHAS[n](P)
    assert out.shape == P["close"].shape
    tail = out.iloc[-1].dropna()
    assert len(tail) > 0, f"alpha_{n} 末行全 NaN"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_alphas.py -q`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 alphas.py — 先搭结构 + 5 个已核对样例**

```python
"""82 个纯价量 alpha。逐字翻译自 Kakushadze (2015) 附录 A。"""
import numpy as np
import pandas as pd
from alpha101 import operators as op
from alpha101 import data


def alpha_1(P):
    # rank(Ts_ArgMax(SignedPower((returns<0 ? stddev(returns,20) : close), 2.), 5)) - 0.5
    inner = P["returns"].where(P["returns"] < 0, P["close"])
    cond = op.ts_stddev(P["returns"], 20).where(P["returns"] < 0, P["close"])
    x = op.signedpower(cond, 2.0)
    return op.rank(op.ts_argmax(x, 5)) - 0.5


def alpha_41(P):
    # (((high * low)^0.5) - vwap)
    return (P["high"] * P["low"]) ** 0.5 - P["vwap"]


def alpha_54(P):
    # ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    num = -1 * ((P["low"] - P["close"]) * (P["open"] ** 5))
    den = (P["low"] - P["high"]) * (P["close"] ** 5)
    return num / den


def alpha_101(P):
    # ((close - open) / ((high - low) + .001))
    return (P["close"] - P["open"]) / ((P["high"] - P["low"]) + 0.001)


def alpha_13(P):
    # (-1 * rank(covariance(rank(close), rank(volume), 5)))
    return -1 * op.rank(op.covariance(op.rank(P["close"]), op.rank(P["volume"]), 5))
```

- [ ] **Step 4: 跑样例测试确认这 5 个通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_alphas.py -k "101 or alpha1_shape" -q`
Expected: PASS(alpha_101 与 alpha_1 通过)

- [ ] **Step 5: 逐个补齐剩余 77 个 alpha + 注册 ALPHAS/compute_all**

按"翻译约定"把论文附录 A 中编号为 2–47, 49–55, 57, 60–62, 64–66, 68, 71–75, 77–78, 81, 83–86, 88, 92, 94–96, 98–99 的公式逐个实现为 `alpha_<N>(P)`。每个函数上方用注释抄录论文原公式。示例(Alpha#4、#6、#12,展示典型模式):

```python
def alpha_4(P):
    # (-1 * Ts_Rank(rank(low), 9))
    return -1 * op.ts_rank(op.rank(P["low"]), 9)


def alpha_6(P):
    # (-1 * correlation(open, volume, 10))
    return -1 * op.correlation(P["open"], P["volume"], 10)


def alpha_12(P):
    # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    return op.sign_(op.delta(P["volume"], 1)) * (-1 * op.delta(P["close"], 1))
```

文件末尾注册:

```python
ALPHAS = {int(name.split("_")[1]): fn
          for name, fn in list(globals().items())
          if name.startswith("alpha_") and callable(fn)}


def compute_all(P):
    out = {}
    for n, fn in sorted(ALPHAS.items()):
        try:
            out[n] = fn(P)
        except Exception as e:
            print(f"alpha_{n} 失败: {e}", flush=True)
    return out
```

实现每一个后,立即跑该编号的冒烟测试:
Run: `alpha101/.venv/bin/pytest alpha101/tests/test_alphas.py -k "smoke and [<N>]" -q`

- [ ] **Step 6: 跑全量冒烟测试**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_alphas.py -q`
Expected: PASS(82 个 smoke + 2 个核对 + shape 测试全过);若某编号末行全 NaN,回看该公式窗口长度是否超过测试数据行数,或翻译是否有误,修正后重跑。

- [ ] **Step 7: 校验数量为 82**

Run: `alpha101/.venv/bin/python -c "from alpha101.alphas import ALPHAS; print(len(ALPHAS)); assert len(ALPHAS)==82"`
Expected: 打印 `82`

- [ ] **Step 8: 提交**

```bash
git add alpha101/alphas.py alpha101/tests/test_alphas.py
git commit -m "feat(alpha101): 82 个纯价量 alpha 公式 + 冒烟测试"
```

---

### Task 6: 回测评估 backtest.py(IC / 分层多空 / 换手)

**Files:**
- Create: `alpha101/backtest.py`
- Create: `alpha101/tests/test_backtest.py`

**Interfaces:**
- Consumes: 因子 `DataFrame`、面板 `close`。
- Produces:
  - `forward_return(close) -> pd.DataFrame` = `close.shift(-1)/close - 1`。
  - `ic_series(factor, fwd, method="spearman") -> pd.Series` — 每日截面(Rank)IC。
  - `ic_stats(ic: pd.Series) -> dict` — `{"ic_mean","ic_std","icir"}`(icir = mean/std)。
  - `quantile_returns(factor, fwd, q=5) -> pd.DataFrame` — 每日分 q 层等权收益,列 `0..q-1`。
  - `long_short_stats(qret, q=5) -> dict` — top 层减 bottom 层的年化收益/夏普(按 252)。
  - `turnover(factor, q=5) -> float` — top 层日均换手(相邻日成分变动比例)。
  - `evaluate(factor, close, q=5) -> dict` — 汇总上述指标 + 保留 `ic`、`qret` 供画图。

- [ ] **Step 1: 写失败测试** `alpha101/tests/test_backtest.py`

```python
import numpy as np
import pandas as pd
import pytest
from alpha101 import backtest as bt


def _factor_and_close():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    cols = list("ABCDE")
    # 让因子完全预测次日收益:因子越大,次日涨越多
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    factor = pd.DataFrame(np.tile(np.arange(5), (6, 1)), index=idx, columns=cols, dtype=float)
    # 构造 fwd return 与 factor 同序
    for i in range(len(idx) - 1):
        close.iloc[i + 1] = close.iloc[i] * (1 + 0.01 * np.arange(5))
    return factor, close


def test_forward_return_is_next_day():
    _, close = _factor_and_close()
    fwd = bt.forward_return(close)
    assert fwd.iloc[-1].isna().all()  # 末行无未来
    assert fwd.iloc[0, 1] == pytest.approx(0.01)


def test_ic_positive_when_aligned():
    factor, close = _factor_and_close()
    fwd = bt.forward_return(close)
    ic = bt.ic_series(factor, fwd)
    assert ic.dropna().mean() == pytest.approx(1.0)  # 完美单调 -> rank IC = 1


def test_quantile_returns_monotonic():
    factor, close = _factor_and_close()
    fwd = bt.forward_return(close)
    qret = bt.quantile_returns(factor, fwd, q=5)
    means = qret.mean()
    assert means.iloc[-1] > means.iloc[0]  # top 层收益 > bottom 层
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_backtest.py -q`
Expected: FAIL

- [ ] **Step 3: 实现** `alpha101/backtest.py`

```python
"""因子回测评估:IC、分层多空、换手。"""
import numpy as np
import pandas as pd


def forward_return(close):
    return close.shift(-1) / close - 1.0


def ic_series(factor, fwd, method="spearman"):
    f = factor.reindex_like(fwd)
    out = {}
    for day in f.index:
        a, b = f.loc[day], fwd.loc[day]
        m = a.notna() & b.notna()
        if m.sum() >= 5:
            out[day] = a[m].corr(b[m], method=method)
    return pd.Series(out).sort_index()


def ic_stats(ic):
    ic = ic.dropna()
    mean, std = ic.mean(), ic.std()
    return {"ic_mean": mean, "ic_std": std,
            "icir": mean / std if std else np.nan}


def quantile_returns(factor, fwd, q=5):
    f = factor.reindex_like(fwd)
    rows = {}
    for day in f.index:
        a, b = f.loc[day], fwd.loc[day]
        m = a.notna() & b.notna()
        if m.sum() < q:
            continue
        labels = pd.qcut(a[m].rank(method="first"), q, labels=False)
        rows[day] = b[m].groupby(labels).mean()
    return pd.DataFrame(rows).T.sort_index()


def long_short_stats(qret, q=5):
    ls = qret[q - 1] - qret[0]
    ls = ls.dropna()
    ann = ls.mean() * 252
    sharpe = (ls.mean() / ls.std() * np.sqrt(252)) if ls.std() else np.nan
    return {"ls_annual": ann, "ls_sharpe": sharpe}


def turnover(factor, q=5):
    top_sets = []
    for day in factor.index:
        a = factor.loc[day].dropna()
        if len(a) < q:
            top_sets.append(set())
            continue
        n = max(1, len(a) // q)
        top_sets.append(set(a.nlargest(n).index))
    tos = []
    for i in range(1, len(top_sets)):
        prev, cur = top_sets[i - 1], top_sets[i]
        if cur:
            tos.append(len(cur - prev) / len(cur))
    return float(np.mean(tos)) if tos else np.nan


def evaluate(factor, close, q=5):
    fwd = forward_return(close)
    ic = ic_series(factor, fwd)
    qret = quantile_returns(factor, fwd, q=q)
    return {**ic_stats(ic), **long_short_stats(qret, q=q),
            "turnover": turnover(factor, q=q), "ic": ic, "qret": qret}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_backtest.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add alpha101/backtest.py alpha101/tests/test_backtest.py
git commit -m "feat(alpha101): 回测评估(IC/ICIR/分层多空/换手)"
```

---

### Task 7: 合成与选股 compose.py + select.py

**Files:**
- Create: `alpha101/compose.py`
- Create: `alpha101/select.py`
- Create: `alpha101/tests/test_compose.py`

**Interfaces:**
- Consumes: 因子 dict、mask。
- Produces:
  - `winsorize_zscore(factor, n=3) -> pd.DataFrame` — 截面 MAD 去极值(±n 倍 MAD)后 z-score。
  - `composite(factors: dict[int, pd.DataFrame], mask=None) -> pd.DataFrame` — 各因子标准化后等权相加(mask 外置 NaN)。
  - `pick(score, date, names=None, top_n=50) -> pd.DataFrame` — 某日 top_n 清单,列 `code,name,score,rank`。

- [ ] **Step 1: 写失败测试** `alpha101/tests/test_compose.py`

```python
import numpy as np
import pandas as pd
import pytest
from alpha101 import compose


def _f(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=list("ABCD"))


def test_zscore_zero_mean_unit_std():
    z = compose.winsorize_zscore(_f([[1.0, 2, 3, 4]]))
    assert z.iloc[0].mean() == pytest.approx(0, abs=1e-9)
    assert z.iloc[0].std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_composite_equal_weight():
    f1 = _f([[1.0, 2, 3, 4]])
    f2 = _f([[4.0, 3, 2, 1]])
    c = compose.composite({1: f1, 2: f2})
    # 两个相反因子等权相加 -> 截面近似相等
    assert c.iloc[0].std(ddof=0) == pytest.approx(0, abs=1e-9)


def test_pick_top_n():
    score = _f([[1.0, 4, 3, 2]])
    picks = compose_pick_helper(score)
    assert list(picks["code"]) == ["B", "C"]


def compose_pick_helper(score):
    from alpha101 import select
    return select.pick(score, score.index[0], top_n=2)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_compose.py -q`
Expected: FAIL

- [ ] **Step 3: 实现** `alpha101/compose.py`

```python
"""因子合成:去极值 -> z-score -> 等权。"""
import numpy as np
import pandas as pd


def winsorize_zscore(factor, n=3):
    med = factor.median(axis=1)
    mad = (factor.sub(med, axis=0)).abs().median(axis=1)
    lo = med - n * 1.4826 * mad
    hi = med + n * 1.4826 * mad
    clipped = factor.clip(lower=lo, upper=hi, axis=0)
    mean = clipped.mean(axis=1)
    std = clipped.std(axis=1, ddof=0).replace(0, np.nan)
    return clipped.sub(mean, axis=0).div(std, axis=0)


def composite(factors, mask=None):
    total = None
    for f in factors.values():
        z = winsorize_zscore(f)
        total = z if total is None else total.add(z, fill_value=0)
    if mask is not None:
        total = total.where(mask.reindex_like(total).fillna(False))
    return total
```

- [ ] **Step 4: 实现** `alpha101/select.py`

```python
"""每日选股清单。"""
import pandas as pd


def pick(score, date, names=None, top_n=50):
    date = pd.Timestamp(date)
    row = score.loc[date].dropna().sort_values(ascending=False).head(top_n)
    df = pd.DataFrame({"code": row.index, "score": row.values})
    df["rank"] = range(1, len(df) + 1)
    df["name"] = df["code"].map(names) if names else ""
    return df[["code", "name", "score", "rank"]]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `alpha101/.venv/bin/pytest alpha101/tests/test_compose.py -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add alpha101/compose.py alpha101/select.py alpha101/tests/test_compose.py
git commit -m "feat(alpha101): 因子合成(去极值/z-score/等权)+ 选股清单"
```

---

### Task 8: 报告 report.py + CLI run.py + 端到端联调

**Files:**
- Create: `alpha101/report.py`
- Create: `alpha101/run.py`
- Create: `alpha101/README.md`

**Interfaces:**
- Consumes: 全部模块。
- Produces:
  - `report.factor_figure(name, ev, out_dir) -> str` — 画 IC 时序 + 分层净值 + 多空净值,存 PNG。
  - `report.summary_table(results: dict[int, dict], out_path) -> pd.DataFrame` — 每因子一行汇总 CSV。
  - CLI:`python -m alpha101.run fetch|eval|select`。

- [ ] **Step 1: 实现** `alpha101/report.py`

```python
"""评估图与汇总表。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def factor_figure(name, ev, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ev["ic"].cumsum().plot(ax=ax[0], title=f"{name} 累计IC")
    qcum = (1 + ev["qret"]).cumprod()
    qcum.plot(ax=ax[1], title=f"{name} 分层净值")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def summary_table(results, out_path):
    rows = []
    for n, ev in sorted(results.items()):
        rows.append({"alpha": n, "ic_mean": ev["ic_mean"], "icir": ev["icir"],
                     "ls_annual": ev["ls_annual"], "ls_sharpe": ev["ls_sharpe"],
                     "turnover": ev["turnover"]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
```

- [ ] **Step 2: 实现** `alpha101/run.py`

```python
"""命令行入口:fetch / eval / select。"""
import sys
import pandas as pd
from alpha101 import data, universe, alphas, backtest, compose, select, report


def cmd_fetch():
    data.fetch_all(years=5)
    print("数据缓存完成")


def cmd_eval():
    P = data.load_panel()
    mask = universe.liquidity_mask(P)
    facs = alphas.compute_all(P)
    results = {}
    for n, f in facs.items():
        fm = universe.apply_mask(f, mask)
        ev = backtest.evaluate(fm, P["close"])
        results[n] = ev
        report.factor_figure(f"alpha_{n}", ev, "output/factor_eval")
    df = report.summary_table(results, "output/factor_eval/factor_summary.csv")
    print(df.sort_values("icir", ascending=False).head(15).to_string(index=False))


def cmd_select(top_n=50):
    P = data.load_panel()
    mask = universe.liquidity_mask(P)
    facs = alphas.compute_all(P)
    score = compose.composite(facs, mask=mask)
    last = score.dropna(how="all").index[-1]
    picks = select.pick(score, last, top_n=top_n)
    out = f"output/picks/{last.date()}.csv"
    import os
    os.makedirs("output/picks", exist_ok=True)
    picks.to_csv(out, index=False)
    print(f"已写 {out}")
    print(picks.head(10).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"fetch": cmd_fetch, "eval": cmd_eval, "select": cmd_select}[cmd]()
```

- [ ] **Step 3: 写** `alpha101/README.md`(用法说明)

```markdown
# alpha101 — A股 101-Alpha 基础多因子框架

基于 Kakushadze (2015)《101 Formulaic Alphas》纯价量子集(82 个)。

## 用法
    python3 -m venv alpha101/.venv
    alpha101/.venv/bin/pip install -r alpha101/requirements.txt
    alpha101/.venv/bin/python -m alpha101.run fetch    # 拉取缓存5年日线(20-40分钟)
    alpha101/.venv/bin/python -m alpha101.run eval     # 82因子评估 -> output/factor_eval/
    alpha101/.venv/bin/python -m alpha101.run select   # 当日top50清单 -> output/picks/

数据仅供研究,不构成投资建议。
```

- [ ] **Step 4: 端到端联调(用小样本真实数据验证跑通)**

先只抓少量股票验证管线(避免等 40 分钟):
```bash
alpha101/.venv/bin/python -c "
from alpha101 import data, universe, alphas, backtest, compose, select, report
import akshare as ak, pandas as pd
codes = ak.stock_zh_a_spot_em()['代码'].astype(str).tolist()[:30]
frames=[]
start=(pd.Timestamp.today()-pd.DateOffset(years=1)).strftime('%Y%m%d')
end=pd.Timestamp.today().strftime('%Y%m%d')
for c in codes:
    o=data._fetch_one(c,start,end)
    if o is not None: frames.append(o)
raw=pd.concat(frames); P=data.build_panel(raw)
mask=universe.liquidity_mask(P)
facs=alphas.compute_all(P)
ev=backtest.evaluate(universe.apply_mask(facs[1],mask),P['close'])
print('alpha_1 ICIR=',ev['icir'])
score=compose.composite(facs,mask=mask)
last=score.dropna(how=\"all\").index[-1]
print(select.pick(score,last,top_n=10).to_string(index=False))
"
```
Expected: 打印 alpha_1 的 ICIR 数值(非 NaN)与 10 只清单;无异常。

- [ ] **Step 5: 提交**

```bash
git add alpha101/report.py alpha101/run.py alpha101/README.md
git commit -m "feat(alpha101): 报告+CLI(fetch/eval/select)+端到端联调"
```

- [ ] **Step 6: 忽略缓存与产出(不入库)**

创建 `alpha101/.gitignore`:
```
.venv/
cache/
```
以及在仓库根忽略产出目录——创建 `output/.gitignore`:
```
*
!.gitignore
```
提交:
```bash
git add alpha101/.gitignore output/.gitignore
git commit -m "chore(alpha101): 忽略 venv/缓存/产出"
```

---

## 附:82 个 alpha 的论文出处

实现 Task 5 时,逐个从论文附录 A 抄录公式(PDF 第 8–15 页)。已在计划中给出完整实现的:#1, #4, #6, #12, #13, #41, #54, #101。其余 74 个按"翻译约定"逐字转换,每个实现后立即用冒烟测试(末行非全 NaN)把关,并在函数注释里保留论文原式,便于后续人工复核。
