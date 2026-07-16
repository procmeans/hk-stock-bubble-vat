# A-share intraday factor six-month validation

## Portfolio metrics

- annual_excess: 0.1871685043188751
- annual_turnover: 47.64982265200556
- benchmark_annual: 0.15658637782668605
- benchmark_total: 0.07111003955345163
- excess_nw_t: 1.0206753616645416
- information_ratio: 1.5817724911576043
- max_drawdown: -0.1472887765116685
- monthly_win_rate: 0.5714285714285714
- sharpe: 0.970842997890384
- strategy_annual: 0.34375488214556116
- strategy_total: 0.14972925346380728

## Limitations and disclosures

- 固定验证区间 2026-01-12 至 2026-07-10
- API 实际区间 2025-12-11 至 2026-07-10
- 预热起点 2025-12-11
- ST 状态最多滞后 4 个交易日
- 行业列可能不是严格时点数据，仅按可获得分类口径处理
- 六个月初步证据；样本 119 个交易日，有效综合 RankIC 样本日 113
- 参数固定：top=500，top_n=50，rebalance=5，min_count=400
- 单边实际成交成本 20.0 bp；同时报告毛值与净值
- 换手定义：双边总成交额换手（买入与卖出名义金额之和/组合净值）
- 剔除统计：分钟质量/无数据 1038 个股日；年龄 265 个股日，停牌 18 个股日，属性缺失/陈旧 0 个股日，ST 407 个股日，无效流通市值 0 个股日
- 分钟 ok率：152418/153456 = 99.32%
- 最终 pool/ranked 覆盖率：58810/59500 = 98.84%
- 固定阈值 综合 RankIC >= 0.03：未达到
- 固定阈值 五组单调性 >= 0.8：未达到
- 固定阈值 扣费后累计超额 > 0：达到
