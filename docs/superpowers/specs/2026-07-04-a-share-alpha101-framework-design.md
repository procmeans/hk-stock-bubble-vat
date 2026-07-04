# A股 101-Alpha 基础多因子框架 — 设计文档

**日期**:2026-07-04
**论文**:Zura Kakushadze, *101 Formulaic Alphas* (arXiv:1601.00991v3, 2015)
**目标市场**:A 股(全市场,含过滤)

---

## 1. 背景与目标

基于论文《101 Formulaic Alphas》构建一个**基础多因子选股框架**,应用于 A 股。论文给出 101 个公式化 alpha,全部由一套算子作用于**价量数据**拼成,平均持仓 0.6–6.4 天,属**短周期截面选股**信号。

本框架分两步交付(用户选择 C = 两者都要):

1. **因子回测研究**:验证每个 alpha 在 A 股的有效性(IC / 分层多空)。
2. **每日选股清单**:将有效因子合成总分,输出打分最高的股票清单。

### 范围界定(纯价量子集)

论文 101 个 alpha 中:

- **18 个**用到 `IndNeutralize`(行业中性化,需行业分类数据):#48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100
- **1 个**用到 `cap`(市值):#56

本"基础"版本**排除这 19 个**,实现剩余 **82 个纯价量 alpha**(只依赖 OHLCV + vwap + adv)。行业中性化版本留作后续扩展。

**82 个保留清单**:1–47, 49–55, 57, 60–62, 64–66, 68, 71–75, 77–78, 81, 83–86, 88, 92, 94–96, 98–99, 101。

---

## 2. 数据

### 字段(论文 A.2 + 衍生)

- `returns` = 日 close-to-close 收益
- `open, high, low, close, volume` = 标准日线,**前复权**(处理分红送股)
- `vwap` = 日成交量加权均价;A 股日线无原生 vwap,用 `amount / volume`(成交额/成交量)近似
- `adv{d}` = 过去 d 日平均日成交额(dollar volume)

### 来源与缓存

- **来源**:akshare `stock_zh_a_hist`(前复权日线,含 amount),逐只拉取。
- **区间**:近 **5 年**日线(约 1200 交易日)。
- **缓存**:全量拉取一次约 20–40 分钟,存本地 **parquet**;后续增量更新只补最新交易日。
- **面板结构**:内存中组织为 `{field: DataFrame[index=日期, columns=股票代码]}` 的字段面板,便于逐日截面运算。

---

## 3. 股票池(universe)

用户选择 B = 全 A 股 + 过滤。每个交易日动态生成有效票池 mask:

- 剔除 **ST / \*ST**
- 剔除**上市 < 60 交易日**的次新股
- 剔除**当日停牌**(当日无成交)
- 剔除**流动性尾部**:过去 20 日日均成交额排名后 20% 的小票

停牌日的股票不参与当日截面 `rank`,不产生因子值。

---

## 4. 模块划分

新建独立目录 `alpha101/`,**不改动现有可视化文件**(fetch_*.py / index.html / data/ 等)。

```
alpha101/
  data.py        拉取+缓存全A股5年前复权日线 → parquet;计算 vwap、adv、returns;构建字段面板
  universe.py    生成每日有效票池 mask(ST/次新/停牌/流动性过滤)
  operators.py   论文全部算子:截面 rank / scale;时序 ts_min/max/argmin/argmax/rank/sum/product/stddev、
                 delay/delta/correlation/covariance/decay_linear;标量 abs/log/sign/signedpower 及比较
  alphas.py      82 个 alpha 公式,每个 = 函数(字段面板 → 因子值 DataFrame[日期×股票])
  backtest.py    因子评估:IC/RankIC 时间序列 + 均值/ICIR、分5层多空累计净值、换手率
  compose.py     去极值(MAD)→ 截面 z-score → 等权合成总分
  select.py      给定日期,总分 top N 选股清单
  report.py      每因子评估图(PNG)+ 汇总表(CSV);每日清单(CSV)
  run.py         命令行入口:fetch / eval / select
```

### 各模块契约

- **operators.py**:每个算子输入/输出均为 `DataFrame[日期×股票]`(或时序窗口参数),截面算子按行(单日)运算,时序算子按列(单股)滚动。这是全框架的基础,须优先保证正确。
- **alphas.py**:每个 `alpha_N(fields) -> DataFrame` 纯函数,只调用 operators,不含 I/O。
- **backtest.py**:输入单因子 `DataFrame` + 未来收益,输出评估指标 dict + 图。
- **compose.py**:输入多因子 dict,输出合成总分 `DataFrame`。

---

## 5. 数据流

```
data.py  (拉取/缓存 parquet)
   ↓  字段面板 {field: DataFrame[日期×股票]}
universe.py  →  每日有效 mask
   ↓
alphas.py (调用 operators.py)  →  逐日截面因子值
   ↓                    ↓
backtest.py          compose.py → select.py
(IC/分层评估)         (合成总分 → top N 清单)
   ↓                    ↓
report.py  →  factor_eval/*.png + factor_summary.csv  |  picks/<日期>.csv
```

---

## 6. 防未来函数(look-ahead)

- 因子仅使用 T 日**收盘后**可得数据,用于预测 **T+1** 收益。
- 收益对齐:默认用 **T+1 收盘 / T 收盘** 的收益评估因子(收盘对齐)。
- 时序算子严格只取过去 d 天;`delay/delta/correlation` 等不触及未来数据。
- 停牌日不参与截面排名,避免用陈旧价格污染 rank。

---

## 7. 回测评估(多空评估 + 清单只做多)

用户选择 A:回测用多空看因子有效性,选股清单只做多头。

- **IC / RankIC**:每日因子值与 T+1 收益的截面(Rank)相关系数,输出时间序列、均值、ICIR(均值/标准差)。
- **分层回测**:按因子值分 **5 层**,计算各层等权组合累计净值,及**多空(top层 − bottom层)**年化收益/夏普。
- **换手率**:相邻两日因子排名变化,衡量交易成本敏感度。
- **说明**:多空仅用于**评估因子强弱**,不代表 A 股可实盘做空。

---

## 8. 因子合成与选股(每日清单)

用户选择 A = 等权 z-score:

1. 每个因子当日截面 **MAD 去极值** → **z-score 标准化**。
2. 82 个因子(或回测筛出的有效子集)**等权相加** = 总分。
3. 取总分 **top N = 50**(默认)只股票,输出**只做多**清单 CSV。

---

## 9. 产出

- `output/factor_eval/alpha_<N>.png` — 每因子:IC 时序、分层净值、多空净值。
- `output/factor_eval/factor_summary.csv` — 每因子一行:IC 均值、ICIR、多空年化、多空夏普、换手率。
- `output/picks/<日期>.csv` — 某交易日总分 top 50 清单(代码、名称、总分、排名)。

---

## 10. 技术栈

Python + pandas + numpy + akshare + matplotlib;缓存 parquet(需 pyarrow);纯本地命令行运行,不接现有可视化前端。

---

## 11. 验证标准(Definition of Done)

1. `data.py` 跑通:parquet 缓存含 5000+ 只股 × ~1200 交易日的前复权 OHLCV + amount。
2. `operators.py` 有单元测试:用小型构造数据核对 `rank / ts_rank / decay_linear / correlation / delta` 等关键算子输出正确。
3. 至少 **5 个代表性 alpha**(#1、#41、#54、#101 及一个含 correlation 的如 #13)手工核对公式实现无误,IC 序列非退化(非全 0 / 非 NaN)。
4. `backtest.py` 对全部 82 个因子产出 `factor_summary.csv`。
5. `select.py` 对最新交易日产出 `picks/<日期>.csv`(top 50)。

---

## 12. 非目标(YAGNI)

- 不做行业中性化的 19 个 alpha(后续扩展)。
- 不做 IC 加权 / 动态因子筛选(基础版用等权)。
- 不做交易成本精细建模、滑点、涨跌停撮合。
- 不接入现有 D3/Canvas 前端可视化。
- 不做实盘下单 / 券商接口。
