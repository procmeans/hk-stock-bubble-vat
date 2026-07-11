# A 股等权账户设计(a_equal_weight,追加)

日期:2026-07-11
状态:已确认(用户:"加 把这个也上模拟盘")

## 背景

风格实验显示"流动性前 500 等权 + 周再平衡"是 A 股近 4 年事实冠军
(年化 ~35%,反转/低波/动量单因子均跑不过它)。上盘作为第三个账户,
与 alpha101、momentum 当面对质。已知风险随账户注明:小微盘拥挤、
2024-02 式雪崩(基准口径最大回撤 -30%+)、幸存者偏差使历史数字偏乐观。

## 组件

1. **strategies/equal_weight.py**:
   - `signal(panel, top_n=500, rebalance=5)`:调仓日按 60 日 ADV
     (panel["amount"] 滚动均值)取前 top_n 等权,间隔 ffill;
   - `targets(panel, top_n=500, **_)`:最新 60 日 ADV 前 top_n 等权(paper 用);
   - 注册进 REGISTRY;需要 amount 字段,当前仅 A 股 paper 路径提供全面板。
2. **paper.py**:
   - `compute_targets` 增 "equal_weight" 分支;`EW_PARAMS={"top_n":500,"rebalance":5}`;
     init 的 params 缺省按策略选(momentum/alpha101/equal_weight);
   - **run-market 命令**:同市场所有账户一次抓数共用面板
     (held 取各账户并集,池 = ADV 前 500 ∪ 并集),逐账户 step 落盘;
     `run --account` 保留单账户路径。
3. **paper-a.yml**:改为 `python -m strategies.paper run-market --market a`
   (跑 a_alpha101 + a_equal_weight,配额不随账户数增长)。
4. **paper.html**:DESC 增 a_equal_weight 文案(含风险注记);账户按钮自动出现
   (读 accounts.json)。

## 测试

- equal_weight signal/targets(合成面板带 amount);
- compute_targets 分发 equal_weight;
- run-market:注入 fetch 断言只调一次、两个账户 last_run 均推进。

## 免责

同 a_alpha101:整手/涨跌停忽略;500 只碎股持仓为纸面简化。
