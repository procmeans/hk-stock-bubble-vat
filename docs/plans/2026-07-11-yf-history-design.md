# yfinance 港股/美股历史日线管线设计

日期:2026-07-11
状态:已确认

## 目标

用 yfinance(免费、无 token)拉取港股/美股全市场历史日线 OHLCV,
接入现有 Alpha101 因子管线,补上 iFinD 海外行情权限缺失的缺口。

## 方案

新建单个模块 `alpha101/yf_history.py`,按 `--market hk|us` 参数化,
与 `ths_history.py` 平行:fetch(抓取+缓存断点续传)→ build_panel → run(因子+合成+选股)。
两个市场共用一套代码,但每次只跑单一市场,截面绝不混市场。

否决的替代方案:

- 在 `ths_history.py` 加 `--source yf` 分支:A 股 iFinD 与海外 yfinance 的
  代码格式、复权口径、amount 缺失处理全不同,会搅在一起。
- 分别写 `yf_hk.py` / `yf_us.py`:近乎全量重复。

## 细节

### 股票池 `load_universe(path)`

- 直接读快照 JSON:港股 `data/2026-07-10.json`(code 为 5 位数字串如 `00700`),
  美股 `data/us-2026-07-10.json`(code 为 ticker 如 `NVDA`)。
- 不复用 `ths_today.load_code_pool`(其含 A 股代码过滤,会滤掉全部海外代码)。
- 返回 `code`/`name` 列;行业取 `g` 字段(GICS 行业组),供 `IndNeutralize` 使用。
- 默认 universe 路径按 market 从 `data/manifest*.json` 取最新一天,可用 `--universe` 覆盖。

### 代码转换 `to_yf_ticker(code, market)`

- 港股:去前导零补到 4 位 + `.HK`,如 `00700` → `0700.HK`。
- 美股:`.` 换 `-`(`BRK.B` → `BRK-B`),其余原样。
- 面板列名保持原始快照 code,不用 yf ticker,保证与 name/行业映射一致。

### 抓取 `fetch_history`

- `yf.download` 批量下载,每批约 100 只,`auto_adjust=True`(前复权口径,与 A 股 qfq 一致)。
- 缓存断点续传,模式同 `ths_history`(复用其 `read_raw_cache`/`write_raw_cache`);
  缓存按市场分开:`alpha101/cache/yf_panel_hk.pkl` / `yf_panel_us.pkl`。
- 退市/无数据代码返回全 NaN,直接丢弃(与 ths_history 一致,下次 fetch 会重试)。
- 默认 `--start 2024-07-01`,`--end` 今天。

### amount 缺失的近似(与 iFinD 管线的关键差异)

yfinance 无成交额字段,而面板 `vwap` 与 `adv` 依赖它:

- `vwap = (high + low + close) / 3`(典型价)
- `amount = vwap × volume`

vwap 类因子略有失真,换来全部 101 个因子可算。

### run

复用 `alphas.compute_all` / `compose.composite` / `universe.liquidity_mask` / `select.pick`。
`liquidity_mask` 为截面分位剔除,货币无关,直接可用。
输出 `output/yf_hk_alpha101_picks.csv` / `yf_us_alpha101_picks.csv`。

### 依赖与测试

- `alpha101/requirements.txt` 增加 `yfinance`。
- `alpha101/tests/test_yf_history.py`:只测纯函数(ticker 转换、universe 加载、
  下载结果规整、面板构建),不联网。

## CLI

```
python -m alpha101.yf_history fetch --market hk
python -m alpha101.yf_history run   --market us
python -m alpha101.yf_history all   --market hk --start 2024-07-01
```
