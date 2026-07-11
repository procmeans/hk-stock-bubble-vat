# 经典量化策略 MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建顶层 `strategies/` 包:6 个经典策略(双均线/均值回归/动量/市场中性/配对套利/机器学习)+ 分层回测框架(data/backtest/metrics/run),多市场日线缓存通用。

**Architecture:** 策略 = 纯函数 `signal(panel) -> date×code 权重表`;`backtest.run` 统一执行模拟(信号 shift(1) 次日生效 + 换手成本);`metrics.summary` 统一绩效;分层概念对齐 zipline/pyfolio(见 spec `docs/plans/2026-07-11-strategies-mvp-design.md`)。

**Tech Stack:** pandas / numpy / scikit-learn(新增)/ pytest;Python 用仓库根 `.venv`。

## Global Constraints

- 严禁未来函数:任何策略权重经 `backtest.run` 内部 `shift(1)` 后才生效;ml 训练集止于 `t - horizon`。
- 面板口径:`dict[str, DataFrame]`,date×code,键含 `open/high/low/close/volume/amount/vwap/returns`。
- 权重约定:多头正、空头负,行绝对值和 ≤ 1。
- 测试全部离线合成面板,不联网;运行命令一律 `.venv/bin/python -m pytest strategies/tests -q`。
- 不引入 zipline/pyfolio/TA-Lib/QuantLib/statsmodels。

---

### Task 1: 包骨架 + backtest.run

**Files:**
- Create: `strategies/__init__.py`(空文件)
- Create: `strategies/backtest.py`
- Create: `strategies/tests/conftest.py`
- Test: `strategies/tests/test_backtest.py`

**Interfaces:**
- Produces: `backtest.run(weights: DataFrame, panel: dict, cost_bps=20, slippage_bps=0) -> DataFrame`,返回列 `gross/net/turnover/equity`,index 与面板日期一致。
- Produces(测试侧): `conftest.make_panel(prices: dict[str, list[float]]) -> dict` 合成面板。

- [ ] **Step 1: 写 conftest 与失败测试**

`strategies/tests/conftest.py`:

```python
import pandas as pd
import pytest


@pytest.fixture
def make_panel():
    def _make(prices: dict, volume: float = 1000.0) -> dict:
        n = len(next(iter(prices.values())))
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.DataFrame(prices, index=idx, dtype=float)
        return {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": close * 0 + volume,
            "amount": close * volume, "vwap": close,
            "returns": close.pct_change(),
        }
    return _make
```

`strategies/tests/test_backtest.py`:

```python
import pandas as pd
import pytest

from strategies import backtest


def test_signal_takes_effect_next_day_with_costs(make_panel):
    panel = make_panel({"a": [100, 110, 121], "b": [100, 100, 100]})
    weights = pd.DataFrame(
        {"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}, index=panel["close"].index
    )

    result = backtest.run(weights, panel, cost_bps=20)

    assert result["gross"].iloc[0] == 0.0            # T 日信号当日不产生收益
    assert result["gross"].iloc[1] == pytest.approx(0.10)
    assert result["turnover"].iloc[1] == pytest.approx(1.0)   # 次日建仓计换手
    assert result["net"].iloc[1] == pytest.approx(0.10 - 1.0 * 20 / 1e4)
    assert result["equity"].iloc[2] == pytest.approx(
        (1 + result["net"].iloc[1]) * (1 + result["net"].iloc[2])
    )


def test_slippage_adds_to_cost(make_panel):
    panel = make_panel({"a": [100, 100, 100]})
    weights = pd.DataFrame({"a": [1.0, 1.0, 1.0]}, index=panel["close"].index)

    result = backtest.run(weights, panel, cost_bps=20, slippage_bps=10)

    assert result["net"].iloc[1] == pytest.approx(-1.0 * 30 / 1e4)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_backtest.py -q`
Expected: FAIL/ERROR(`No module named strategies.backtest`)

- [ ] **Step 3: 实现 backtest.py**

```python
"""执行模拟层:信号次日生效 + 换手成本(对应 zipline 的 commission/slippage 模型)。"""
import pandas as pd


def run(weights, panel, cost_bps=20, slippage_bps=0):
    """weights: date×code 目标权重(T 日收盘信号,T+1 日起持有)。"""
    rets = panel["close"].pct_change()
    w = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps + slippage_bps) / 1e4
    equity = (1.0 + net).cumprod()
    return pd.DataFrame(
        {"gross": gross, "net": net, "turnover": turnover, "equity": equity}
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_backtest.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/ && git commit -m "feat(strategies): 回测执行层(信号次日生效+换手成本)"
```

---

### Task 2: metrics.summary

**Files:**
- Create: `strategies/metrics.py`
- Test: `strategies/tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 1 `backtest.run` 的返回 DataFrame(列 `gross/net/turnover/equity`)。
- Produces: `metrics.summary(result: DataFrame) -> dict`,键 `total_return/annual_return/annual_vol/sharpe/max_drawdown/annual_turnover`。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from strategies import metrics


def test_summary_hand_computed():
    net = pd.Series([0.0, 0.01, -0.02, 0.01])
    result = pd.DataFrame({
        "gross": net, "net": net,
        "turnover": pd.Series([0.0, 1.0, 0.0, 0.5]),
        "equity": (1 + net).cumprod(),
    })

    stats = metrics.summary(result)

    equity = (1 + net).cumprod()
    total = equity.iloc[-1] - 1
    assert stats["total_return"] == pytest.approx(total)
    assert stats["annual_return"] == pytest.approx((1 + total) ** (252 / 4) - 1)
    assert stats["annual_vol"] == pytest.approx(net.std(ddof=0) * np.sqrt(252))
    # 最大回撤:峰值 1.01 -> 谷底 1.01*0.98
    assert stats["max_drawdown"] == pytest.approx(-0.02)
    assert stats["annual_turnover"] == pytest.approx(1.5 * 252 / 4)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_metrics.py -q`
Expected: FAIL(`No module named strategies.metrics`)

- [ ] **Step 3: 实现 metrics.py**

```python
"""绩效指标层(对应 pyfolio 的 tear sheet,MVP 只保留核心指标)。"""
import numpy as np

TRADING_DAYS = 252


def summary(result) -> dict:
    net, equity = result["net"], result["equity"]
    n = max(len(net), 1)
    total = equity.iloc[-1] - 1.0
    annual = (1.0 + total) ** (TRADING_DAYS / n) - 1.0
    vol = net.std(ddof=0) * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": annual / vol if vol > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "annual_turnover": result["turnover"].sum() * TRADING_DAYS / n,
    }
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_metrics.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/metrics.py strategies/tests/test_metrics.py
git commit -m "feat(strategies): 绩效指标 summary"
```

---

### Task 3: data.load_panel

**Files:**
- Create: `strategies/data.py`
- Test: `strategies/tests/test_data.py`

**Interfaces:**
- Consumes: `alpha101.yf_history.build_panel(raw)` / `alpha101.ths_history.build_panel(raw)`(raw 为长表 `code,date,open,high,low,close,volume[,amount]`)。
- Produces: `data.load_panel(market: str, top: int | None = None, cache=None) -> dict` 面板。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_data.py`:

```python
import pandas as pd
import pytest

from strategies import data


def _raw(codes, days=3):
    rows = []
    for i, code in enumerate(codes):
        for d in range(days):
            price = 10.0 * (i + 1)
            rows.append({
                "code": code, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "open": price, "high": price, "low": price, "close": price,
                "volume": 100.0 * (i + 1),
            })
    return pd.DataFrame(rows)


def test_load_panel_us_from_cache(tmp_path):
    cache = tmp_path / "yf_panel_us.pkl"
    _raw(["NVDA", "AAPL"]).to_pickle(cache)

    panel = data.load_panel("us", cache=cache)

    assert set(panel["close"].columns) == {"NVDA", "AAPL"}
    assert "vwap" in panel and "amount" in panel


def test_load_panel_top_filters_by_amount(tmp_path):
    cache = tmp_path / "yf_panel_us.pkl"
    _raw(["SMALL", "BIG"]).to_pickle(cache)

    panel = data.load_panel("us", top=1, cache=cache)

    assert list(panel["close"].columns) == ["BIG"]   # 成交额更大的留下


def test_load_panel_missing_cache_hints_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="yf_history fetch --market hk"):
        data.load_panel("hk", cache=tmp_path / "nope.pkl")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_data.py -q`
Expected: FAIL(`No module named strategies.data`)

- [ ] **Step 3: 实现 data.py**

```python
"""行情装载层(对应 zipline 的 data bundle):统一面板口径。"""
from pathlib import Path

import pandas as pd

CACHES = {
    "us": Path("alpha101/cache/yf_panel_us.pkl"),
    "hk": Path("alpha101/cache/yf_panel_hk.pkl"),
    "a": Path("alpha101/cache/ths_panel.pkl"),
}
FETCH_HINTS = {
    "us": "python -m alpha101.yf_history fetch --market us",
    "hk": "python -m alpha101.yf_history fetch --market hk",
    "a": "python -m alpha101.ths_history fetch",
}


def load_panel(market: str, top: int | None = None, cache=None) -> dict:
    cache = Path(cache) if cache else CACHES[market]
    if not cache.exists():
        raise FileNotFoundError(f"缓存不存在 {cache},先运行: {FETCH_HINTS[market]}")
    raw = (
        pd.read_parquet(cache)
        if cache.suffix.lower() == ".parquet"
        else pd.read_pickle(cache)
    )
    if market == "a":
        from alpha101.ths_history import build_panel
    else:
        from alpha101.yf_history import build_panel
    panel = build_panel(raw)
    if top:
        keep = panel["amount"].tail(60).mean().nlargest(top).index
        panel = {
            key: value[keep] if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
    return panel
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_data.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/data.py strategies/tests/test_data.py
git commit -m "feat(strategies): 多市场面板装载(us/hk/a 缓存)"
```

---

### Task 4: ma_cross 双均线

**Files:**
- Create: `strategies/ma_cross.py`
- Test: `strategies/tests/test_ma_cross.py`

**Interfaces:**
- Produces: `ma_cross.signal(panel, fast=20, slow=60) -> DataFrame`(date×code 权重,信号股等权)。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_ma_cross.py`:

```python
def test_holds_uptrend_not_downtrend(make_panel):
    up = [100 + i for i in range(10)]
    down = [100 - i for i in range(10)]
    panel = make_panel({"up": up, "down": down})

    from strategies.ma_cross import signal
    w = signal(panel, fast=2, slow=4)

    assert w["up"].iloc[-1] == 1.0      # 快线在慢线上方,独占权重
    assert w["down"].iloc[-1] == 0.0
    assert (w.iloc[:3] == 0).all().all()  # 慢线未形成前空仓
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_ma_cross.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 ma_cross.py**

```python
"""双均线:快线上穿慢线持有、下穿离场,信号股等权。"""
import numpy as np


def signal(panel, fast=20, slow=60):
    close = panel["close"]
    hold = close.rolling(fast).mean() > close.rolling(slow).mean()
    counts = hold.sum(axis=1)
    return hold.div(counts.replace(0, np.nan), axis=0).fillna(0.0)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_ma_cross.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/ma_cross.py strategies/tests/test_ma_cross.py
git commit -m "feat(strategies): 双均线策略"
```

---

### Task 5: mean_reversion 均值回归

**Files:**
- Create: `strategies/mean_reversion.py`
- Test: `strategies/tests/test_mean_reversion.py`

**Interfaces:**
- Produces: `mean_reversion.signal(panel, window=20, entry=-2.0, exit_=0.0) -> DataFrame`。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_mean_reversion.py`:

```python
def test_enters_on_dip_exits_on_recovery(make_panel):
    # 10 天横盘 -> 暴跌(z<-2 入场)-> 收复(z>=0 出场)
    prices = [100.0] * 10 + [90.0, 90.0, 101.0, 101.0]
    panel = make_panel({"a": prices, "flat": [100.0] * len(prices)})

    from strategies.mean_reversion import signal
    w = signal(panel, window=5, entry=-2.0, exit_=0.0)

    assert w["a"].iloc[10] == 1.0    # 暴跌日入场
    assert w["a"].iloc[11] == 1.0    # 未回归前继续持有
    assert w["a"].iloc[-1] == 0.0    # 收复后离场
    assert (w["flat"] == 0).all()    # 无偏离不交易
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_mean_reversion.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 mean_reversion.py**

```python
"""均值回归:z-score 低于 entry 买入,回到 exit_ 以上离场,持仓等权。"""
import numpy as np
import pandas as pd


def signal(panel, window=20, entry=-2.0, exit_=0.0):
    close = panel["close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std
    state = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    state[z >= exit_] = 0.0
    state[z < entry] = 1.0
    hold = state.ffill().fillna(0.0) > 0
    counts = hold.sum(axis=1)
    return hold.div(counts.replace(0, np.nan), axis=0).fillna(0.0)
```

注意赋值顺序:先 exit 后 entry,同日既触发 entry 又触发 exit 时以 entry 为准(不会发生,但顺序决定优先级)。横盘股 std=0 → z 为 NaN → 两个条件都不触发 → 全程 0。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_mean_reversion.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/mean_reversion.py strategies/tests/test_mean_reversion.py
git commit -m "feat(strategies): 均值回归策略"
```

---

### Task 6: momentum 横截面动量

**Files:**
- Create: `strategies/momentum.py`
- Test: `strategies/tests/test_momentum.py`

**Interfaces:**
- Produces: `momentum.signal(panel, top_n=20, lookback=252, skip=21, rebalance=21) -> DataFrame`。
- Produces(内部复用): `momentum.score(close, lookback, skip) -> DataFrame`,Task 7 复用。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_momentum.py`:

```python
def test_picks_strongest_and_rebalances(make_panel):
    n = 20
    strong = [100 * 1.05 ** i for i in range(n)]
    weak = [100 * 0.98 ** i for i in range(n)]
    flat = [100.0] * n
    panel = make_panel({"strong": strong, "weak": weak, "flat": flat})

    from strategies.momentum import signal
    w = signal(panel, top_n=1, lookback=10, skip=2, rebalance=5)

    assert w["strong"].iloc[-1] == 1.0
    assert w["weak"].iloc[-1] == 0.0
    assert (w.iloc[:10].sum(axis=1) == 0).all()       # lookback 未满前空仓
    assert (w.iloc[10] == w.iloc[12]).all()           # 调仓间隔内权重不变
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_momentum.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 momentum.py**

```python
"""横截面动量:12-1 月收益 top N 等权,固定间隔调仓。"""
import numpy as np


def score(close, lookback=252, skip=21):
    """跳过最近 skip 日的 lookback 区间收益(避开短期反转)。"""
    return close.shift(skip) / close.shift(lookback) - 1.0


def signal(panel, top_n=20, lookback=252, skip=21, rebalance=21):
    close = panel["close"]
    mom = score(close, lookback, skip)
    keep = np.zeros(len(close), dtype=bool)
    keep[::rebalance] = True
    mom[~keep] = np.nan                     # 非调仓日不出新信号
    top = mom.rank(axis=1, ascending=False) <= top_n
    counts = top.sum(axis=1)
    w = top.div(counts.replace(0, np.nan), axis=0)
    w[~keep] = np.nan
    return w.ffill().fillna(0.0)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_momentum.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/momentum.py strategies/tests/test_momentum.py
git commit -m "feat(strategies): 横截面动量策略"
```

---

### Task 7: market_neutral 市场中性

**Files:**
- Create: `strategies/market_neutral.py`
- Test: `strategies/tests/test_market_neutral.py`

**Interfaces:**
- Consumes: `momentum.score(close, lookback, skip)`(Task 6)。
- Produces: `market_neutral.signal(panel, top_n=20, lookback=252, skip=21, rebalance=21) -> DataFrame`(净敞口 0、总敞口 1)。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_market_neutral.py`:

```python
import pytest


def test_long_short_neutral(make_panel):
    n = 20
    panel = make_panel({
        "strong": [100 * 1.05 ** i for i in range(n)],
        "weak": [100 * 0.95 ** i for i in range(n)],
        "flat": [100.0] * n,
        "flat2": [100.0 + 0.01 * i for i in range(n)],
    })

    from strategies.market_neutral import signal
    w = signal(panel, top_n=1, lookback=10, skip=2, rebalance=5)

    last = w.iloc[-1]
    assert last["strong"] == pytest.approx(0.5)     # 多最强
    assert last["weak"] == pytest.approx(-0.5)      # 空最弱
    assert last.sum() == pytest.approx(0.0)         # 净敞口 0
    assert last.abs().sum() == pytest.approx(1.0)   # 总敞口 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_market_neutral.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 market_neutral.py**

```python
"""市场中性:动量 top N 等权做多、bottom N 等权做空,净敞口 0、总敞口 1。

A 股无法融券做空全名单,回测结果视为纸面模拟(run.py 会标注)。
"""
import numpy as np

from strategies.momentum import score


def signal(panel, top_n=20, lookback=252, skip=21, rebalance=21):
    close = panel["close"]
    mom = score(close, lookback, skip)
    keep = np.zeros(len(close), dtype=bool)
    keep[::rebalance] = True
    mom[~keep] = np.nan                     # 非调仓日不出新信号
    long_ = mom.rank(axis=1, ascending=False) <= top_n
    short = (mom.rank(axis=1, ascending=True) <= top_n) & ~long_
    w = (
        long_.div((long_.sum(axis=1) * 2).replace(0, np.nan), axis=0)
        - short.div((short.sum(axis=1) * 2).replace(0, np.nan), axis=0)
    )
    w[~keep] = np.nan
    return w.ffill().fillna(0.0)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_market_neutral.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/market_neutral.py strategies/tests/test_market_neutral.py
git commit -m "feat(strategies): 市场中性多空策略"
```

---

### Task 8: pairs 配对统计套利

**Files:**
- Create: `strategies/pairs.py`
- Test: `strategies/tests/test_pairs.py`

**Interfaces:**
- Produces: `pairs.signal(panel, n_pairs=5, train=252, window=20, entry=2.0, exit_=0.5) -> DataFrame`;`pairs.top_pairs(close, train, n_pairs) -> list[tuple[str, str]]`。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_pairs.py`:

```python
def _pair_prices(n=60, diverge_at=40):
    # a、b 高度相关(带确定性小噪声);diverge_at 后 a 跳涨脱离 b
    a, b = [], []
    for i in range(n):
        base = 100 + (i % 5)
        a.append(base * (1.25 if i >= diverge_at else 1.0))
        b.append(base + 0.5 * (-1) ** i)
    return a, b


def test_shorts_spread_after_divergence(make_panel):
    a, b = _pair_prices()
    noise = [100.0 + (i % 7) for i in range(len(a))]   # 第三只:相关性低
    panel = make_panel({"a": a, "b": b, "noise": noise})

    from strategies.pairs import signal, top_pairs
    assert top_pairs(panel["close"], train=40, n_pairs=1) == [("a", "b")]

    w = signal(panel, n_pairs=1, train=40, window=10, entry=2.0, exit_=0.5)

    assert (w.iloc[:40] == 0).all().all()       # 训练窗内不交易
    later = w.iloc[45]
    assert later["a"] < 0 and later["b"] > 0    # 价差过高:空 a 多 b
    assert abs(later["a"]) + abs(later["b"]) <= 1.0 + 1e-9
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_pairs.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 pairs.py**

```python
"""配对统计套利:训练窗选高相关对,对数价差 z-score 开平仓。

MVP 用相关性 + 静态对冲比替代协整检验(statsmodels ADF 留作升级)。
"""
import numpy as np
import pandas as pd


def top_pairs(close, train=252, n_pairs=5):
    """训练窗内日收益相关性最高、互不重叠的股票对。"""
    window = close.iloc[:train]
    valid = window.columns[window.notna().all()]
    corr = window[valid].pct_change().corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
    ranked = upper.stack().sort_values(ascending=False)
    result, used = [], set()
    for (left, right), _ in ranked.items():
        if left in used or right in used:
            continue
        result.append((left, right))
        used.update((left, right))
        if len(result) == n_pairs:
            break
    return result


def signal(panel, n_pairs=5, train=252, window=20, entry=2.0, exit_=0.5):
    close = panel["close"]
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for left, right in top_pairs(close, train, n_pairs):
        la, lb = np.log(close[left]), np.log(close[right])
        beta = np.polyfit(lb.iloc[:train], la.iloc[:train], 1)[0]
        spread = la - beta * lb
        z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        state = pd.Series(np.nan, index=close.index)
        state[z.abs() < exit_] = 0.0
        state[z > entry] = -1.0    # 价差过高:空 left 多 right
        state[z < -entry] = 1.0    # 价差过低:多 left 空 right
        state.iloc[:train] = 0.0   # 训练窗内不交易
        state = state.ffill().fillna(0.0)
        weights[left] += state * 0.5 / n_pairs
        weights[right] -= state * 0.5 / n_pairs
    return weights
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_pairs.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/pairs.py strategies/tests/test_pairs.py
git commit -m "feat(strategies): 配对统计套利策略"
```

---

### Task 9: ml 机器学习

**Files:**
- Create: `strategies/ml.py`
- Test: `strategies/tests/test_ml.py`
- Modify: 根 `.venv` 安装 scikit-learn(`.venv/bin/pip install scikit-learn`)

**Interfaces:**
- Produces: `ml.signal(panel, top_n=20, train=504, retrain=21, horizon=21, feat_windows=(21, 63, 126)) -> DataFrame`。

- [ ] **Step 1: 装依赖 + 写失败测试**

```bash
.venv/bin/pip install -q scikit-learn
```

`strategies/tests/test_ml.py`:

```python
import pytest


def test_walk_forward_picks_persistent_winner(make_panel):
    n = 60
    panel = make_panel({
        "winner": [100 * 1.02 ** i for i in range(n)],
        "loser": [100 * 0.99 ** i for i in range(n)],
        "flat": [100 + (i % 3) for i in range(n)],
        "flat2": [100 + ((i + 1) % 3) for i in range(n)],
    })

    from strategies.ml import signal
    w = signal(panel, top_n=1, train=30, retrain=5, horizon=3,
               feat_windows=(3, 5, 8))

    assert (w.iloc[:30].sum(axis=1) == 0).all()          # 训练期空仓
    assert w["winner"].iloc[-1] == pytest.approx(1.0)    # 学到持续上涨者
    row_sums = w.abs().sum(axis=1)
    assert ((row_sums == 0) | (row_sums == pytest.approx(1.0))).all()
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_ml.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 ml.py**

```python
"""机器学习:滚动窗口逻辑回归预测下期截面强弱(walk-forward,无未来函数)。"""
import numpy as np
import pandas as pd


def _features(panel, feat_windows):
    close, volume = panel["close"], panel["volume"]
    short = max(feat_windows[0], 2)
    feats = {f"ret{win}": close.pct_change(win) for win in feat_windows}
    feats["vol"] = close.pct_change().rolling(short * 2).std()
    feats["volu"] = (
        volume.rolling(short).mean() / volume.rolling(short * 3).mean()
    )
    return feats


def signal(panel, top_n=20, train=504, retrain=21, horizon=21,
           feat_windows=(21, 63, 126)):
    from sklearn.linear_model import LogisticRegression

    close = panel["close"]
    feats = _features(panel, feat_windows)
    forward = close.shift(-horizon) / close - 1.0
    label = forward.gt(forward.median(axis=1), axis=0)

    X = np.stack([f.values for f in feats.values()], axis=-1)  # (T, N, F)
    y = label.values
    y_known = forward.notna().values
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in range(train, len(close), retrain):
        # 训练集止于 t - horizon:标签在 t 日已完全实现,无泄漏
        train_x = X[max(0, t - train): t - horizon].reshape(-1, X.shape[-1])
        train_y = y[max(0, t - train): t - horizon].reshape(-1)
        known = y_known[max(0, t - train): t - horizon].reshape(-1)
        ok = ~np.isnan(train_x).any(axis=1) & known
        if ok.sum() < 30 or len(np.unique(train_y[ok])) < 2:
            continue
        model = LogisticRegression(max_iter=200)
        model.fit(train_x[ok], train_y[ok])

        now = X[t]
        ok_now = ~np.isnan(now).any(axis=1)
        if not ok_now.any():
            continue
        prob = pd.Series(np.nan, index=close.columns)
        prob[ok_now] = model.predict_proba(now[ok_now])[:, 1]
        top = prob.nlargest(min(top_n, int(ok_now.sum()))).index
        row = pd.Series(0.0, index=close.columns)
        row[top] = 1.0 / len(top)
        weights.iloc[t] = row
    return weights.ffill().fillna(0.0)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest strategies/tests/test_ml.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add strategies/ml.py strategies/tests/test_ml.py
git commit -m "feat(strategies): 滚动逻辑回归 ML 策略"
```

---

### Task 10: run CLI + requirements + README + 真实数据冒烟

**Files:**
- Create: `strategies/run.py`
- Create: `strategies/requirements.txt`
- Create: `strategies/README.md`
- Test: `strategies/tests/test_run.py`

**Interfaces:**
- Consumes: 全部前置任务;`REGISTRY: dict[str, signal_fn]`。
- Produces: CLI `python -m strategies.run --market us|hk|a (--strategy NAME | --all) [--top N] [--cost-bps X]`;输出 `output/strategies/<market>_<strategy>_equity.csv`、`--all` 时另存 `<market>_compare.csv`。

- [ ] **Step 1: 写失败测试**

`strategies/tests/test_run.py`:

```python
import pandas as pd


def test_run_one_writes_equity_and_stats(make_panel, tmp_path, monkeypatch):
    from strategies import run as run_mod
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", tmp_path)
    panel = make_panel({
        "a": [100 + i for i in range(30)],
        "b": [100 - i for i in range(30)],
    })

    stats = run_mod.run_one("ma_cross", panel, "us")

    assert "sharpe" in stats and "max_drawdown" in stats
    saved = pd.read_csv(tmp_path / "us_ma_cross_equity.csv")
    assert "equity" in saved.columns


def test_market_neutral_on_a_share_is_annotated(make_panel, tmp_path, monkeypatch):
    from strategies import run as run_mod
    monkeypatch.setattr(run_mod, "OUTPUT_DIR", tmp_path)
    panel = make_panel({
        "s": [100 * 1.03 ** i for i in range(40)],
        "w": [100 * 0.97 ** i for i in range(40)],
        "f": [100 + (i % 2) for i in range(40)],
    })

    stats = run_mod.run_one("market_neutral", panel, "a")

    assert stats["note"] == "A股做空为纸面模拟"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest strategies/tests/test_run.py -q`
Expected: FAIL(module 不存在)

- [ ] **Step 3: 实现 run.py + requirements + README**

`strategies/run.py`:

```python
"""CLI:单策略回测或六策略对比。"""
import argparse
from pathlib import Path

import pandas as pd

from strategies import backtest, data, metrics
from strategies import ma_cross, market_neutral, mean_reversion, ml, momentum, pairs

REGISTRY = {
    "ma_cross": ma_cross.signal,
    "mean_reversion": mean_reversion.signal,
    "momentum": momentum.signal,
    "market_neutral": market_neutral.signal,
    "pairs": pairs.signal,
    "ml": ml.signal,
}
OUTPUT_DIR = Path("output/strategies")


def run_one(name, panel, market, cost_bps=20.0):
    result = backtest.run(REGISTRY[name](panel), panel, cost_bps=cost_bps)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / f"{market}_{name}_equity.csv")
    stats = metrics.summary(result)
    if name == "market_neutral" and market == "a":
        stats["note"] = "A股做空为纸面模拟"
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "hk", "a"], required=True)
    parser.add_argument("--strategy", choices=sorted(REGISTRY))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    args = parser.parse_args()
    if not args.all and not args.strategy:
        parser.error("需要 --strategy 或 --all")

    panel = data.load_panel(args.market, top=args.top)
    rows = []
    for name in sorted(REGISTRY) if args.all else [args.strategy]:
        stats = run_one(name, panel, args.market, cost_bps=args.cost_bps)
        rows.append({"strategy": name, **stats})
        print(f"{name}: 完成", flush=True)
    table = pd.DataFrame(rows).set_index("strategy")
    print(table.round(4).to_string())
    if args.all:
        table.to_csv(OUTPUT_DIR / f"{args.market}_compare.csv")


if __name__ == "__main__":
    main()
```

`strategies/requirements.txt`:

```
pandas
numpy
scikit-learn
pytest
```

`strategies/README.md`:

```markdown
# strategies — 经典量化策略 MVP

6 个经典策略的最小实现,统一回测口径,教学/研究用途。
设计文档:`docs/plans/2026-07-11-strategies-mvp-design.md`
(分层概念对齐 zipline/pyfolio:data=bundle,signal=algorithm,
backtest=execution+commission/slippage,metrics=tear sheet)。

## 用法

    # 先准备任一市场缓存
    python -m alpha101.yf_history fetch --market us
    # 单策略 / 全部对比
    python -m strategies.run --market us --strategy momentum
    python -m strategies.run --market us --all

策略:ma_cross(双均线)、mean_reversion(均值回归)、momentum(动量)、
market_neutral(市场中性多空)、pairs(配对套利)、ml(滚动逻辑回归)。

口径:T 日收盘信号次日生效;换手计单边成本(默认 20bp);
A 股做空为纸面模拟。研究用途,不构成投资建议。
```

- [ ] **Step 4: 全量测试确认通过**

Run: `.venv/bin/python -m pytest strategies/tests -q`
Expected: 全部通过(≥12 个)
Run: `.venv/bin/python -m pytest -q`
Expected: 仓库全部测试通过(alpha101 145 个 + strategies)

- [ ] **Step 5: 真实数据冒烟(需联网抓过缓存)**

```bash
# 若无缓存,先小规模抓取(或全市场,20-40 分钟)
.venv/bin/python -m alpha101.yf_history fetch --market us
.venv/bin/python -m strategies.run --market us --all --top 100
```

Expected: 打印六策略指标对比表,`output/strategies/us_compare.csv` 生成。

- [ ] **Step 6: Commit**

```bash
git add strategies/ && git commit -m "feat(strategies): CLI、依赖与文档,六策略齐备"
```
