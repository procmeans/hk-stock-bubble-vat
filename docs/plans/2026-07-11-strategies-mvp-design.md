# 经典量化策略 MVP 设计(strategies/ 模块)

日期:2026-07-11
状态:已确认

## 目标

为 6 种经典量化策略各实现一个最小可用 MVP,统一回测口径、统一绩效指标,
支持美股/港股/A股三个市场的现有日线缓存,便于对照学习策略思想。

高频与事件驱动暂不做:高频需要 tick 级数据与低延迟执行链路(iFinD 只有分钟级
`high_frequency` 与 3 秒 `snap_shot`,且需另开权限);事件驱动需要新闻/财报数据流。
以后可用 iFinD 分钟线做准高频演示。

## 定位

独立顶层 `strategies/` 包。不依赖、不修改 `feature/quant-trading-system` 分支
(那是横截面多因子选股系统,本模块是策略谱系教学实验)。

不引入 zipline/pyfolio/TA-Lib/QuantLib(停维护或依赖过重,MVP 用 pandas 手写);
statsmodels 暂不加,配对交易用价差 z-score 替代协整检验。唯一新增依赖:scikit-learn。

**但架构分层刻意对齐工业框架的概念**,便于理解这些工具的设计思想、以后平滑迁移:

| 本模块 | zipline 对应 | pyfolio 对应 | 职责 |
|---|---|---|---|
| `data.py` | data bundle / `DataPortal` | — | 行情装载,统一面板口径 |
| 策略 `signal()` | `TradingAlgorithm`(向量化版) | — | 信号 → 目标权重 |
| `backtest.py` | 执行模拟 + `commission`/`slippage` 模型 | — | 成交时序、成本、净值 |
| `metrics.py` | — | tear sheet | 绩效归因与风险指标 |

成本模型做成参数(`cost_bps` 佣金、`slippage_bps` 滑点),对应 zipline 的
可插拔 commission/slippage;以后要换成按成交量冲击的模型,只改 backtest 一处。

## 结构

```
strategies/
  data.py            # load_panel(market)
  backtest.py        # run(weights, panel, cost_bps, slippage_bps) -> 净值曲线
  metrics.py         # summary(equity, weights): 年化/夏普/回撤/换手 (pyfolio 风格)
  ma_cross.py        # 1 双均线
  mean_reversion.py  # 2 均值回归
  momentum.py        # 3 横截面动量
  market_neutral.py  # 4 市场中性多空
  pairs.py           # 5 统计套利(配对)
  ml.py              # 6 机器学习
  run.py             # CLI
  tests/
```

## 统一接口

- 面板:沿用 alpha101 口径,`dict[str, DataFrame]`,date × code,
  含 `open/high/low/close/volume/amount/vwap/returns`。
- 策略:纯函数 `signal(panel, **params) -> DataFrame`(date × code 目标权重,
  行和的绝对值 ≤ 1;多空策略多头 +、空头 -)。
- 回测:`backtest.run(weights, panel, cost_bps=20, slippage_bps=0)`:
  - T 日收盘信号 → T+1 收盘成交(权重整体 `shift(1)`),杜绝未来函数;
  - 收益 = 权重 × 次日 close-to-close 收益;
  - 成本 = 权重变化绝对值之和 × (cost_bps + slippage_bps) / 1e4;
  - 返回日收益序列与净值曲线;指标计算在 `metrics.summary`:
    累计/年化收益、年化波动、夏普、最大回撤、年均换手。

## 数据层

`load_panel(market)`:

- `us` / `hk`:读 `alpha101/cache/yf_panel_{market}.pkl`(yf_history 管线),
  用 `yf_history.build_panel` 构面板;
- `a`:读 `alpha101/cache/ths_panel.pkl`(iFinD)或 `panel.parquet`(akshare),
  用对应模块的 build_panel;
- 缓存不存在时报错并提示对应 fetch 命令。
- 可选 `top`:按期末成交额取前 N 只,加快实验。

## 六个策略(MVP 参数)

1. **ma_cross 双均线**:每只股票 20 日均线上穿 60 日均线持有、下穿离场;
   信号股等权,无信号日空仓。
2. **mean_reversion 均值回归**:20 日 z-score < -2 买入,z 回到 0 离场;持仓等权。
3. **momentum 横截面动量**:过去 252-21 日(12-1 月)收益率排名 top N(默认 20)
   等权持有,每 21 个交易日换仓。
4. **market_neutral 市场中性**:动量得分 top N 等权做多、bottom N 等权做空,
   净敞口 0、总敞口 1。A 股无法真实做空,输出标注"纸面模拟"。
5. **pairs 统计套利**:训练窗内相关性最高的 K 对(默认 5 对),价差
   z-score > 2 做空价差、< -2 做多价差、|z| < 0.5 平仓;每对资金等分。
6. **ml 机器学习**:滚动窗口(训练 504 日、每 21 日重训)逻辑回归,
   特征 = 过去 21/63/126 日收益、20 日波动、量比;预测下期上涨概率 top N 等权。
   仅用 scikit-learn 基础件,不调参。

## CLI

```
python -m strategies.run --market us --strategy momentum
python -m strategies.run --market hk --all          # 六策略对比表
```

输出:`output/strategies/<market>_<strategy>_equity.csv` + 终端指标表;
`--all` 另存 `output/strategies/<market>_compare.csv`。

## 测试

pytest + 小合成面板(构造确定性行情):

- backtest:shift 时序正确(信号次日才生效)、成本按换手扣、指标数值可手算验证;
- 每个策略:关键行为断言(金叉后有持仓、z<-2 触发买入、动量选中强势股、
  中性策略净敞口≈0、配对策略对内一多一空、ml 输出权重合法);
- 全部离线,不联网。

## 追加:BigQuant 三策略(2026-07-11 追加确认)

参照用户提供的三篇 BigQuant 文档,追加三个滚动训练的横截面因子策略,
与 ml 同族,共享 walk-forward 骨架(抽到 `rolling_reg.py`):

7. **elastic_net 弹性网络选股**(qjl2v0tXah):滚动窗口 ElasticNet 回归预测
   下期收益,top 30 等权,训练窗 240 日、每 20 日调仓,alpha=0.005、l1_ratio=0.5。
8. **icir_weight ICIR 定权多因子**(lSr5ySFNmn):每次调仓用过去一年各因子
   日度 rank-IC 的 ICIR(均值/标准差)作为因子权重,合成打分选 top 30,
   每 5 日调仓。MVP 用直接 ICIR 定权替代原文的梯度上升最大化(思想一致,
   与 alpha101/compose.py 同源)。
9. **lasso 滚动选股**(J4Bwgm4r4j):滚动 Lasso 回归预测未来 10 日收益,
   top 20 等权,每 10 日调仓,L1 自动筛因子。

与原文的差异(数据所限):原文因子含 PE/PB/ROE 等基本面,本仓面板无
point-in-time 基本面数据,统一用价量因子(多窗口动量、波动、量比)替代;
股票池用 `--top` 流动性截断替代沪深300/中证500 成分。

## 免责

研究与教学用途,不构成投资建议;A 股做空、成本与滑点均为简化假设。
