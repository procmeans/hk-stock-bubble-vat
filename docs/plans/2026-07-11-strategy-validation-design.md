# 策略有效性验证与优化设计(validate / optimize)

日期:2026-07-11
状态:已确认

## 系统定位(用户定型)

本仓库目标是一个**完整的量化交易系统**:

```
数据层(yfinance / iFinD / akshare)
  → 策略层(strategies/ 九策略;alpha101/ 多因子)
  → 检验层(本篇:validate 有效性 + optimize 防过拟合调参)   ← 当前阶段
  → 模拟交易(paper trading)                                  ← 下一里程碑
  → 实盘对接                                                  ← 长期目标
```

## 动机:现有冒烟结果不可信的五个原因

1. 无基准对比(牛市里绝对收益无意义,要看超额);
2. 幸存者偏差(用"今天的市值前100"回测过去);
3. 单市场单区间(可能是运气);
4. 无统计显著性(两年夏普的置信区间很宽);
5. 参数无敏感性分析(可能过拟合)。

## 数据侧修正

- 美股/港股改用**全市场** 4 年历史(2022-07 起,含 2022 熊市段),yfinance
  后台抓取;已退市股仍缺失,属**残余幸存者偏差**,文档与输出中注明。
- A 股可用 iFinD(token 已验证:A股✓ 港股✓ 美股✗);全市场历史抓取消耗
  API 配额,执行前需用户确认。

## 第一步:strategies/validate.py(策略有效吗)

接口:

- `benchmark_returns(panel) -> Series`:同面板等权持有(逐日等权,即截面
  日收益均值),代表"不选股、躺平持有这批股票"。
- `validate_one(name, panel, market, cost_bps=20) -> dict`:
  - 策略日收益取 `backtest.run(...)["net"]`;
  - **起算日 = 策略首次有持仓的日期**(此前的空仓预热期剔除,否则稀释统计);
  - 超额 = 策略日收益 − 基准日收益(起算日之后);
  - `t_stat = mean(超额) / std(超额, ddof=0) × √n`(简化 t 检验,iid 假设,
    文档注明未做 Newey-West 自相关修正);
  - 输出:策略/基准/超额年化、t 值、结论(t≥2 显著跑赢;t≤-2 显著跑输;
    其余"超额不显著")、最大回撤、起算日;
  - `yearly_table(net, bench) -> DataFrame`:分年收益对比(年 × 策略/基准/超额)。
- CLI:`python -m strategies.validate --market us|hk|a (--strategy X | --all)
  [--top N] [--cost-bps X]`:
  - 打印汇总表(策略 × 指标+结论),存 `output/strategies/<market>_validate.csv`;
  - 单策略模式额外打印分年表。

## 第二步:strategies/optimize.py(防过拟合调参)

- **切分**:按日期前 60% 训练段 / 后 40% 留出段。所有策略信号皆
  point-in-time(无未来函数),故**在全量面板上算一次权重**,再把日收益
  按日期切段评估——既避免留出段被预热期吃掉,又不泄漏。
- **网格**:`GRIDS` 字典,每策略 ≤ 9 个组合(如 ma_cross: fast×slow 且
  fast<slow;momentum: top_n×lookback;lasso/elastic_net: alpha×top_n 等)。
- **流程**:每组合 → 全面板权重 → 训练段日均收益年化夏普 → 排序;
  仅最优组合在留出段评估(**留出段只用一次**,避免二次挑选污染);
  留出段夏普 < 训练段一半或为负 → 打过拟合红牌。
- **敏感性表**:全组合的训练段夏普,好参数应呈"高原",尖峰 = 过拟合信号。
- `metrics.daily_sharpe(net) -> float`:日收益年化夏普(mean×252 /
  (std×√252)),供分段评估(总收益年化法对短段不稳)。
- CLI:`python -m strategies.optimize --market us --strategy ma_cross
  [--top N] [--ratio 0.6]`,存 `output/strategies/<market>_<strategy>_optimize.csv`。

## 测试(合成面板,离线)

- benchmark_returns = 截面均值(手算验证);
- t 值手算验证;预热期剔除(前 k 日空仓不进统计);
- yearly_table 分年数值正确(跨两个日历年构造);
- split 比例正确、段间无重叠;
- 网格约束(fast<slow 组合数);
- 过拟合红牌:构造训练段动量有效、留出段反转的面板,momentum 网格
  在留出段变差 → 亮牌。

## 免责

t 检验为 iid 简化;等权基准为逐日再平衡口径;残余幸存者偏差存在;
研究用途,不构成投资建议。
