# 业绩预告事件驱动策略设计(PEAD,盈余公告后漂移)

日期:2026-07-11
状态:已确认(用户选定事件类型:业绩预告/快报)

## 目标

A 股业绩预告事件研究 + 可验证策略:公告后正向漂移(PEAD)。
本轮范围:数据抓取 → 事件研究(CAR)→ validate 检验。**验证后再议是否上模拟盘。**

## 数据源

东方财富数据中心 `RPT_PUBLIC_OP_NEWPREDICT`(免费、无鉴权、已实测):
`SECURITY_CODE / NOTICE_DATE / REPORT_DATE / PREDICT_TYPE / ADD_AMP_LOWER/UPPER`。
按报告期(季度末)分页抓取 2022-06-30 至今,缓存 `alpha101/cache/events_yjyg.csv`
(cache 目录已 gitignore)。备用源:iFinD `report_query`(端点已确认存在)、巨潮资讯。

## 组件(strategies/events.py)

1. `fetch_events(start="2022-06-30") -> DataFrame(code, notice_date, report_date, type,
   amp_lower, amp_upper)`:分季分页抓取并写缓存;`load_events()` 读缓存。
2. `signal(panel, events=None, hold=20, positive=("预增", "扭亏")) -> weights`:
   - 事件对齐:入场信号日 = 首个 交易日 ≥ NOTICE_DATE(业绩预告多为盘后发布;
     回测层再 shift(1) 次日成交,无前视);
   - 公告后持有 hold 个交易日,活跃事件等权;无事件日空仓;
   - 注册 REGISTRY("pead"),events=None 时读缓存 → 直接可用 validate。
3. `car(panel, events=None, pre=5, post=20, groups=按 PREDICT_TYPE)`:
   事件研究——异常收益 = 个股日收益 − 等权基准;输出各类型事件的
   CAR 曲线(-pre..+post)、样本数、事件后 [1, post] 累计异常收益的 t 值
   (逐事件 CAR 的均值/标准误)。
4. CLI:`python -m strategies.events fetch|study`;检验走
   `python -m strategies.validate --market a --strategy pead`。

## 测试(离线)

- 事件对齐:周末公告落到下一交易日;持有窗口长度正确;两事件重叠等权;
- positive 过滤:非正向类型不进 signal;
- CAR 手算:构造确定性异常收益验证曲线与 t;
- fetch 用保存的 JSON fixture 测解析,不联网。

## 已知局限(诚实声明)

- NOTICE_DATE 无时分,统一按"信号日收盘后、次日成交"的保守口径;
- 未剔除同日多事件/一字板无法成交的情形(涨停开盘买不进 → 实盘更差);
- 事件样本存在幸存者偏差(与面板同源)。
