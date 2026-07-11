# strategies — 经典量化策略 MVP

9 个经典策略的最小实现,统一回测口径,教学/研究用途。
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
market_neutral(市场中性多空)、pairs(配对套利)、ml(滚动逻辑回归)、
elastic_net(弹性网络选股)、icir_weight(ICIR 定权多因子)、
lasso(Lasso 滚动选股)——后三个参照 BigQuant 文档,基本面因子以
价量因子替代(面板无 point-in-time 基本面)。

## 有效性验证与调参

    python -m strategies.validate --market us --all                # 基准/超额/t检验
    python -m strategies.validate --market us --strategy momentum  # 加分年表
    python -m strategies.optimize --market us --strategy ma_cross  # 训练/留出调参

结论口径:t≥2 显著跑赢;|t|<2 超额不显著(不能排除运气);
优化的留出段只用一次,红牌参数勿采用。已退市股缺失 = 残余幸存者偏差。

口径:T 日收盘信号次日生效;换手计单边成本(默认 20bp);
A 股做空为纸面模拟。研究用途,不构成投资建议。
