# A 股 alpha101 模拟盘实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `a_alpha101` 模拟盘账户(alpha101 全因子组合,A 股,iFinD 数据),paper.py 多账户多市场化,页面账户切换,独立 Actions 工作流。

**Architecture:** 策略适配层进 REGISTRY 复用 validate;paper.step 面板化 + 按 state.strategy 分发目标权重;run 按 state.market 分发数据源;accounts.json 驱动页面。

**Tech Stack:** pandas/numpy/scipy(alpha101)/requests(iFinD);测试合成数据 + monkeypatch。

## Global Constraints

- 参数:alpha101 组合 top_n=50、rebalance=5;成本单边 20bp;虚拟资金 ¥100,000。
- 重计算只在调仓日(alpha101 全因子较重,每日只记 NAV)。
- 兼容既有 us_momentum:state 缺 strategy/market/params 时按 momentum/us/PARAMS。
- A 股纸面简化:忽略整手与涨跌停约束(文档与页面注明)。

---

### Task 1: alpha101_composite 适配层 + 注册 + A 股验证

**Files:**
- Create: `strategies/alpha101_composite.py`
- Modify: `strategies/run.py`(REGISTRY 注册)
- Test: `strategies/tests/test_alpha101_composite.py`;`strategies/tests/test_run.py`(registry 断言更新)

**Interfaces:**
- Produces: `signal(panel, top_n=50, rebalance=5) -> DataFrame`;
  `targets(panel, top_n=50, **_) -> dict[str, float]`(paper 调仓日用)。

- [ ] **Step 1: 写失败测试**

```python
import pandas as pd
import pytest


def _stub_alpha101(monkeypatch, panel):
    score = panel["close"].rank(axis=1, pct=True)      # 收盘价高者组合分高
    monkeypatch.setattr("alpha101.alphas.compute_all", lambda p: {"f": score})
    monkeypatch.setattr("alpha101.universe.liquidity_mask", lambda p: score.notna())
    monkeypatch.setattr("alpha101.compose.composite",
                        lambda factors, mask=None: factors["f"])


def test_signal_picks_top_composite_on_rebalance_grid(make_panel, monkeypatch):
    panel = make_panel({"hi": [200.0] * 12, "lo": [100.0] * 12})
    _stub_alpha101(monkeypatch, panel)
    from strategies.alpha101_composite import signal

    w = signal(panel, top_n=1, rebalance=5)

    assert w["hi"].iloc[-1] == 1.0 and w["lo"].iloc[-1] == 0.0
    assert (w.iloc[0] == w.iloc[4]).all()              # 调仓间隔内不变


def test_targets_latest_cross_section(make_panel, monkeypatch):
    panel = make_panel({"hi": [200.0] * 6, "lo": [100.0] * 6})
    _stub_alpha101(monkeypatch, panel)
    from strategies.alpha101_composite import targets

    assert targets(panel, top_n=1) == {"hi": 1.0}
```

- [ ] **Step 2: 运行确认失败**
  `.venv/bin/python -m pytest strategies/tests/test_alpha101_composite.py -q` → module 不存在
- [ ] **Step 3: 实现**

```python
"""alpha101 全因子组合适配层:滚动 IC 加权合成 -> 调仓日 top N 等权。

面板无 ind 时行业中性化因子按 alpha101 既有行为跳过。计算较重,
不进 optimize.GRIDS;paper 只在调仓日调用 targets。
"""
import numpy as np


def _score(panel):
    from alpha101 import alphas, compose, universe
    factors = alphas.compute_all(panel)
    mask = universe.liquidity_mask(panel)
    return compose.composite(factors, mask=mask)


def signal(panel, top_n=50, rebalance=5):
    score = _score(panel)
    keep = np.zeros(len(score), dtype=bool)
    keep[::rebalance] = True
    score[~keep] = np.nan
    top = score.rank(axis=1, ascending=False) <= top_n
    counts = top.sum(axis=1)
    weights = top.div(counts.replace(0, np.nan), axis=0)
    weights[~keep] = np.nan
    return weights.ffill().fillna(0.0)


def targets(panel, top_n=50, **_):
    """paper 调仓日用:最新截面 top N 等权。"""
    latest = _score(panel).iloc[-1].dropna()
    top = latest.nlargest(top_n)
    if top.empty:
        return {}
    return {code: 1.0 / len(top) for code in top.index}
```

run.py:REGISTRY 增加 `"alpha101_composite": alpha101_composite.signal`
(import 行加 alpha101_composite);test_run.py 的 registry 集合断言加该名。

- [ ] **Step 4: 全部 strategies 测试通过**
- [ ] **Step 5: 真实验证(本地 iFinD 缓存)并记录结果**

```bash
.venv/bin/python -m strategies.validate --market a --strategy alpha101_composite --top 500
```

Expected: 打印 verdict 与分年表(结果好坏都继续,报告给用户)。

- [ ] **Step 6: Commit** `feat(strategies): alpha101 全因子组合适配层(注册+验证)`

---

### Task 2: paper.py 多账户多市场化

**Files:**
- Modify: `strategies/paper.py`
- Test: `strategies/tests/test_paper.py`(step 面板化改造 + 新增分发/市场测试)

**Interfaces:**
- Consumes: Task 1 `alpha101_composite.targets(panel, **params)`。
- Produces:
  - `A101_PARAMS = {"top_n": 50, "rebalance": 5}`
  - `step(state, panel: dict, params=None, cost_bps=COST_BPS)`(panel 至少含 close;
    params 缺省取 `state["params"]` 再退 PARAMS;策略取 `state.get("strategy","momentum")`)
  - `compute_targets(strategy, panel, params) -> dict`
  - `init(account, capital, strategy="momentum", market="us", params=None, title=None)`
    (state 增 strategy/market/params;维护 `paper/accounts.json`
    `[{"account","title","currency"}]`,us→"$",a→"¥")
  - `a_universe_tickers(snapshot_dir=Path("data")) -> list`(manifest_a 最新快照市值前 FETCH_SIZE)
  - `fetch_a_panel(codes, window_days=WINDOW_DAYS) -> dict`(iFinD 批 25 →
    normalize_history_frame → build_panel)
  - `run(account, fetch=None)`:market=="a" 走 fetch_a_panel(ADV 池取
    panel["amount"].tail(60).mean() 前 500 ∪ 持仓,全面板切列);否则原 yfinance 路径
    (panel={"close": pool})。

- [ ] **Step 1: 改造既有测试 + 新增失败测试**

既有 7 个 step/run 测试把第二参数 `close` 改为 `{"close": close}`
(`test_run_appends_nav_and_saves_state` 的 fetch 注入不变,run 内部 us 路径构面板)。新增:

```python
def test_step_dispatches_alpha101_targets(monkeypatch):
    import strategies.alpha101_composite as ac
    seen = {}

    def fake_targets(panel, **kw):
        seen["kw"] = kw
        return {"A": 1.0}

    monkeypatch.setattr(ac, "targets", fake_targets)
    state = _state()
    state["strategy"] = "alpha101"
    state["params"] = {"top_n": 1, "rebalance": 5}

    state, nav_row, orders = paper.step(state, {"close": _close()})

    assert state["pending_targets"] == {"A": 1.0}
    assert seen["kw"] == {"top_n": 1, "rebalance": 5}


def test_init_registers_account_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("a_alpha101", 100000.0, strategy="alpha101", market="a",
               title="A股 Alpha101")
    entries = json.loads((tmp_path / "accounts.json").read_text())
    assert entries[0]["account"] == "a_alpha101"
    assert entries[0]["currency"] == "¥"
    state = paper.load_state("a_alpha101")
    assert state["strategy"] == "alpha101" and state["market"] == "a"
    assert state["params"] == paper.A101_PARAMS


def test_run_a_market_uses_amount_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    paper.init("acct", 100000.0, strategy="momentum", market="a",
               params={"top_n": 1, "lookback": 10, "skip": 2, "rebalance": 5})
    close = _close()
    panel = {"close": close, "volume": close * 0 + 1000.0,
             "amount": close * 1000.0, "returns": close.pct_change()}
    monkeypatch.setattr(paper, "a_universe_tickers", lambda: ["A", "B"])
    paper.run("acct", fetch=lambda codes, window_days=0: panel)
    assert paper.load_state("acct")["last_run"] == \
        close.index[-1].strftime("%Y-%m-%d")
```

- [ ] **Step 2: 确认失败;Step 3: 实现(见 Interfaces,代码遵设计);Step 4: 全量测试通过**
- [ ] **Step 5: Commit** `feat(strategies): paper 多账户多市场化(策略分发+iFinD A股数据)`

---

### Task 3: 迁移 us_momentum + 建 a_alpha101 + 真实冒烟

**Files:**
- Modify: `paper/us_momentum/state.json`(补 strategy/market/params 字段)
- Create: `paper/accounts.json`、`paper/a_alpha101/`

- [ ] **Step 1: 迁移脚本 + 建账**

```bash
.venv/bin/python - <<'EOF'
import json
p = 'paper/us_momentum/state.json'
s = json.load(open(p))
s.setdefault('strategy', 'momentum'); s.setdefault('market', 'us')
s.setdefault('params', {"top_n": 40, "lookback": 126, "skip": 21, "rebalance": 21})
json.dump(s, open(p, 'w'), ensure_ascii=False, indent=1)
EOF
.venv/bin/python -c "
from strategies import paper
paper.register_account('us_momentum', title='美股动量', currency='\$')"
.venv/bin/python -m strategies.paper init --account a_alpha101 \
  --strategy alpha101 --market a --capital 100000 --title 'A股 Alpha101 全因子'
```

(register_account(account, title, currency) 为 Task 2 的 init 内部函数,公开导出。)

- [ ] **Step 2: 真实冒烟(本地 iFinD,source .env)**

```bash
set -a; source .env; set +a
.venv/bin/python -m strategies.paper run --account a_alpha101
.venv/bin/python -m strategies.paper status --account a_alpha101
```

Expected: 首日 NAV=100000,挂单 50 只。

- [ ] **Step 3: Commit** `feat(paper): a_alpha101 账户上线(首日挂单)`

---

### Task 4: paper.html 账户切换 + paper-a.yml + Secret + 触发验证

**Files:**
- Modify: `paper.html`(账户按钮行;fmt$ 用账户 currency;标题/副标题随账户)
- Create: `.github/workflows/paper-a.yml`
- Modify: `strategies/README.md`(多账户说明一段)

- [ ] **Step 1: paper.html 切换逻辑**

boot() 改为:fetch `paper/accounts.json`(404 时退回 `[{"account":"us_momentum",
"title":"美股动量","currency":"$"}]`)→ 渲染按钮行(样式同站点 #mkt 的
on/off 态)→ `load(entry)`:按 `paper/<account>/` 读三个文件渲染,货币符号用
entry.currency,`<h1>`/副标题随 entry.title/desc 更新。图表/tiles/orders 渲染函数不变。

- [ ] **Step 2: paper-a.yml**

```yaml
name: paper-trading-a

on:
  schedule:
    - cron: "30 7 * * 1-5"   # UTC,A股收盘后
  workflow_dispatch: {}

permissions:
  contents: write

concurrency:
  group: paper-trading-a
  cancel-in-progress: false

jobs:
  step:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      THS_HTTP_REFRESH_TOKEN: ${{ secrets.THS_HTTP_REFRESH_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install deps
        run: pip install pandas numpy scipy requests
      - name: Run paper trading step (A-share alpha101)
        run: python -m strategies.paper run --account a_alpha101
      - name: Commit & push state
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git pull --rebase
          git add paper/
          git diff --cached --quiet || git commit -m "paper: a_alpha101 daily step $(date -u +%F)"
          git push
```

- [ ] **Step 3: 设置 Secret 并全量收尾**

```bash
set -a; source .env; set +a
gh secret set THS_HTTP_REFRESH_TOKEN --body "$THS_HTTP_REFRESH_TOKEN"
.venv/bin/python -m pytest -q          # 全量通过
```

- [ ] **Step 4: 合并推送 + 手动触发工作流验证**

```bash
git checkout main && git merge feature/a-alpha101-paper --no-edit
git branch -d feature/a-alpha101-paper && git push
gh workflow run paper-trading-a && sleep 5
gh run watch $(gh run list --workflow=paper-trading-a --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```

Expected: 工作流绿色;首日已本地跑过则日志显示"已运行过,跳过"(幂等)。

- [ ] **Step 5: Commit(页面与文档改动随 Task 4 一并提交)**
  `feat: 模拟盘账户切换页面与 A股工作流`
