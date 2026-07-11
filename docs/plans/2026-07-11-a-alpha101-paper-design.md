# A 股 alpha101 全因子模拟盘设计(a_alpha101)

日期:2026-07-11
状态:已确认

## 目标

新增第二个模拟盘账户 `a_alpha101`:alpha101 全因子滚动 IC 加权组合,
A 股,top 50 等权,每 5 个交易日调仓,单边 20bp,虚拟资金 ¥100,000。
上盘前先用 validate 补一次 4 年历史检验(结果好坏都上盘,但必须知道底细)。

## 组件

### 1. strategies/alpha101_composite.py(策略适配层)

把 alpha101 组合适配成标准 `signal(panel, top_n=50, rebalance=5)`:
`alphas.compute_all(panel)` → `universe.liquidity_mask` → `compose.composite`
→ 调仓日截面 top N 等权(keep/ffill 模式同 momentum)。注册进 REGISTRY 后
自动获得 validate 能力;不进 optimize.GRIDS(全因子计算太重,网格不经济)。
面板无 `ind` 时行业中性化因子按 alpha101 的既有行为跳过,组合用其余因子。

另供 paper 用的 `targets(panel, top_n=50, **_) -> dict`:只算最新截面。

### 2. paper.py 多账户多市场化

- state.json 增加 `strategy` / `market` / `params`(缺省 momentum/us/PARAMS,
  兼容既有 us_momentum;同时给其 state.json 补上这三个字段)。
- `step(state, panel, ...)`:第二参数从 close 改为 panel dict(至少含 close);
  调仓日按 `state["strategy"]` 分发目标权重函数:
  momentum → close 截面动量;alpha101 → alpha101_composite.targets(全面板)。
  重计算只发生在调仓日(每日 NAV 便宜,周频重算组合)。
- `run(account)`:按 market 分发数据源——
  us:现有 yfinance 路径;
  a:`data/a-*.json` 市值前 800 → iFinD `history_quotation`(批 25,窗口 400
  自然日)→ `ths_history.normalize_history_frame/build_panel` → ADV 前 500 ∪ 持仓。
- `init` 增加 `--strategy --market` 参数;维护 `paper/accounts.json` 清单
  (含 account/title/desc,供页面渲染)。

### 3. paper.html 账户切换

读 `paper/accounts.json` 渲染顶部账户切换(两个按钮),选中账户加载其
nav/orders/state;版式、配色、交互不变(用户既定惯例)。

### 4. .github/workflows/paper-a.yml

cron `30 7 * * 1-5`(UTC,A 股收盘后)+ 手动触发;
`pip install pandas numpy scipy requests`;
`THS_HTTP_REFRESH_TOKEN` 来自 GitHub Secrets(用 `gh secret set` 写入,
用户已确认);运行 `python -m strategies.paper run --account a_alpha101`,
提交 paper/。iFinD 每日消耗约 32 个请求的配额。

## 测试

- alpha101_composite.signal:monkeypatch alpha101 三件套(compute_all/
  liquidity_mask/composite)验证调仓网格与等权(alpha101 自身已有 145 测试);
- paper.step 面板化后既有 7 测试改造为传 `{"close": ...}`;
- 策略分发:stub 策略字段选择正确目标函数;
- run 的 a 市场路径:注入 fetch 返回合成面板,验证 ADV 池与步进落盘;
- 真实冒烟:本地 iFinD 跑 `paper run --account a_alpha101` 首日。

## 免责

A 股整手(100 股)约束在纸面模拟中忽略(碎股);涨跌停无法成交的约束
忽略——两者都会让纸面结果略偏乐观,文档与页面注明。
