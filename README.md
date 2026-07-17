# 港股 / A股 / 美股 · 估值气泡缸

一个**股市数据可视化**项目。把整个市场塞进一个大玻璃缸里,每家公司是一颗气泡——直观地看市场的"贵 / 便宜""大 / 小"和每天的浮沉。支持**港股、A股、美股**三个市场,可单看也可全部倒进同一个缸里比较。

> 🌐 在线访问:**https://procmeans.github.io/hk-stock-bubble-vat/**

---

## 视觉编码

| 视觉编码 | 含义 |
|---|---|
| 🫧 **气泡大小** | 总市值,统一折算**美元**后跨市场可比(1 USD = 7.80 HKD = 7.15 CNY,固定汇率) |
| ⬆️ **垂直高度** | PE(TTM)—— 越往上越贵(成长 / 高估值),越往下越便宜(价值) |
| 🎨 **颜色** | 按 PE 冷→暖;刻度随当前市场的实际 PE 分布自适应(2%~98% 分位数) |
| 🪨 **缸底灰色层** | 亏损股(PE ≤ 0)沉淀在底部 |
| ⭕ **描边色**(同缸模式) | 港股蓝 · A股橙 · 美股紫 |

## 交互

- 🔀 底部**市场切换**:港股(默认)/ A股 / 美股 / 全部同缸;也可用链接直达:[`#a`](https://procmeans.github.io/hk-stock-bubble-vat/#a) [`#us`](https://procmeans.github.io/hk-stock-bubble-vat/#us) [`#all`](https://procmeans.github.io/hk-stock-bubble-vat/#all)
- 🖱️ 悬停任意气泡 → 名称、市值(美元 + 原币)、PE、营收、净利、较前一日涨跌
- 📅 **日期滑块**切换历史快照;▶️ **播放**让气泡随时间平滑移动

---

## 数据

| 市场 | 覆盖 | 来源 | 文件 |
|---|---|---|---|
| 港股 | 约 2700+ 家(剔除「－Ｒ」人民币双柜台) | 市值/PE:百度股市通;营收/净利:东财(年报);代码表:新浪 | `data/<日期>.json` |
| A股 | 全市场约 5500 家 | 东财数据中心估值报表(PE TTM/市值)+ 业绩报表(年报营收/净利) | `data/a-<日期>.json` |
| 美股 | 市值 ≥ 1 亿美元约 4200+ 家(NASDAQ/NYSE/AMEX) | 东财行情列表(PE TTM/市值,美元) | `data/us-<日期>.json` |

- **更新**:每个交易日**港时 20:00** 由 GitHub Actions 自动抓取,生成当日快照(`data/manifest*.json` 记录各市场可用日期)
- **口径**:港股、A股为当日收盘数据;美股因抓取时段尚未开盘,为其**前一交易日收盘**数据;营收/净利为最近一期年报、各公司原始币种;美股营收/净利暂缺

---

## 分析记录

- [2026-07-13 A 股涨停股动量分析](docs/analysis/2026-07-13-limit-up-momentum.md):基于同花顺涨停池接口,对 2026-07-10 涨停股做次交易日续板概率排序与盘中复盘。

---

## A 股分钟因子六个月验证

`intraday` 包提供可复现的三阶段命令行流程。默认验证区间固定为
`2026-01-12` 至 `2026-07-10`，日线预热披露起点为 `2025-12-11`；流动性池
取前 500，只在至少 400 只有效股票时计算，五日调仓、持有综合得分前 50，
单边成交成本 20 bp。测试中的小样本参数不代表研究参数。

先安装研究依赖并准备同花顺 HTTP API refresh token：

```bash
pip install -r alpha101/requirements.txt
export THS_HTTP_REFRESH_TOKEN='...'
```

先以固定的不复权口径生成日线前置缓存：

```bash
python -m alpha101.ths_history fetch --universe data/a-2026-07-07.json --start 2022-07-01 --end 2026-07-10 --cache alpha101/cache/ths_panel.pkl --batch-size 80
```

该请求显式使用 `CPS=1`（不复权）和 `Fill=Omit`。缓存必需 schema 为
`code,date,open,high,low,close,volume,amount`；本次复现缓存含 4,873,244 行、
5,203 个代码。输入股票池 `data/a-2026-07-07.json` 的 SHA256 为
`abc0256b0985eca70ef4b4afb88e2cc8934bfb0a7174ed7be35fcd22443ed583`，生成的
`alpha101/cache/ths_panel.pkl` SHA256 为
`783f2580d90347554111ff0b91ce0df4f5ce654ad62c28b01ac7f1f75a3adc84`。

按阶段运行便于在真实下载中安全续传：

```bash
python -m intraday.run prepare   # 只读日线缓存，原子写 plan 与日度股票池
python -m intraday.run fetch     # 单次获取 access token，续传属性/后复权/分钟缓存
python -m intraday.run validate  # 严格只读缓存，生成 CSV、Markdown 与 PNG
```

也可用 `python -m intraday.run all` 顺序执行三阶段。默认日线缓存为
`alpha101/cache/ths_panel.pkl`，研究缓存为 `intraday/cache`，结果写到
`output/intraday_6m`；各命令可用 `--daily-cache`、`--cache`、`--output`、
`--start`、`--end`、`--warmup`、`--top`、`--min-count`、`--top-n`、
`--rebalance` 和 `--cost-bps` 显式覆盖。缺失、损坏或覆盖不足的缓存会令命令
非零退出，不会被当成已完成数据。

分钟请求遇到 HTTP 504/超时时，可安全重跑
`python -m intraday.run fetch --batch-size 100`；已完成日期会跳过，较小批次不改变
研究 plan 或参数。

分钟数据使用普通同花顺高频行情，不需要 Level2。每个交易日一个 Parquet 分区
和一个完成 manifest；股票代码统一为六位基础代码，字段为本地时间 `time`、
`close`、`volume`、`amount`，仅接受 09:30–11:30 与 13:00–15:00。
请求口径固定为不复权 `CPS=no`、`Fill=Original`；质量门槛为至少 200 条分钟、
至少 30 个正成交量分钟，且分钟成交额与日线成交额误差不超过 2%。manifest
必须对计划中的每个 `(date, code)` 明确记录 `returned` 或 `no_data`，coverage
同时记录质量原因。开盘回测另用 `CPS=3`、`Fill=Omit` 的后复权日线。

真实六个月下载规模很大：候选并集通常超过任一日的 500 只，且每只股票每个
交易日最多约 241 行、3 个核心数值字段。`prepare` 会先打印候选并集、预计行数
和单元数；开始 `fetch` 前请检查 iFinD API 配额、磁盘空间与预计运行时间。

---

## 自动化

`GitHub Actions`(`.github/workflows/update.yml`)每天定时运行:

```
fetch_a.py   A股:数据中心批量翻页,~1 分钟
fetch_us.py  美股:行情列表批量翻页,~1 分钟
fetch_hk.py  港股:逐只并发(12 线程)抓取,~7 分钟
  → 异常保护(有效记录过少视为被限流,放弃写入)
  → 写入当日快照 + 更新 manifest
  → 提交并推送(自动触发 GitHub Pages 重新部署)
```

网页为纯静态单页(D3 + Canvas),数据按需从 `data/` 加载,无需后端。

分钟因子验证与模拟盘发布由 `.github/workflows/intraday-paper.yml` 负责:

```
python -m intraday.run all
python -m intraday.paper publish --input output/intraday_6m --paper-dir paper
  → 写入 `output/intraday_6m/` 与 `paper/a_intraday_6m/`
  → 提交并推送(自动触发 GitHub Pages 重新部署)
```

---

## 本地 / 二次开发

```bash
pip install -r requirements.txt
python fetch_a.py           # A股快照
python fetch_us.py          # 美股快照
python fetch_hk.py          # 港股快照(较慢)
# 网页需通过本地服务器打开以允许 fetch data/
python -m http.server 8000  # 然后访问 http://localhost:8000
```

## 路线图

- [x] **A 股 / 美股**版本与三市同缸比较
- [x] 市值统一美元折算、PE 刻度自适应
- [ ] 按**行业 / 板块**分缸或着色
- [ ] 横轴引入第二指标(如 ROE、营收增速、股息率)做二维分布
- [ ] 历史回放区间选择、个股轨迹高亮
- [ ] 估值分位、资金流向等更多维度

欢迎提 issue 建议想看的维度。

---

*数据仅供研究与可视化展示,不构成任何投资建议。*
