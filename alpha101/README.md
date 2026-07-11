# alpha101 — A股 101-Alpha 基础多因子框架

基于 Kakushadze (2015)《101 Formulaic Alphas》纯价量子集(82 个)。

## 用法
    python3 -m venv alpha101/.venv
    alpha101/.venv/bin/pip install -r alpha101/requirements.txt
    alpha101/.venv/bin/python -m alpha101.run fetch    # 拉取缓存5年日线(20-40分钟)
    alpha101/.venv/bin/python -m alpha101.run eval     # 82因子评估 -> output/factor_eval/
    alpha101/.venv/bin/python -m alpha101.run select   # 当日top50清单 -> output/picks/

## 同花顺今日行情评分

使用 iFinD HTTP API 的实时行情 `open/high/low/latest` 跑 WorldQuant 第 101 号因子:

    export THS_HTTP_REFRESH_TOKEN='你的 refresh token'
    python3 -m alpha101.ths_today \
      --universe data/a-2026-07-07.json \
      --output output/ths_alpha101_today.csv \
      --top-n 101

因子公式:

    (close - open) / ((high - low) + 0.001)

## yfinance 港股/美股历史管线

用 yfinance(免费、无 token)拉取港股/美股全市场前复权日线,跑完整 Alpha101 合成选股。
universe 默认取 `data/manifest*.json` 里最新一天的快照;yfinance 无成交额字段,
`vwap` 用典型价 `(high+low+close)/3` 近似,`amount = vwap * volume`:

    python3 -m alpha101.yf_history fetch --market hk   # 抓取缓存,断点续传
    python3 -m alpha101.yf_history run   --market hk   # -> output/yf_hk_alpha101_picks.csv
    python3 -m alpha101.yf_history all   --market us --start 2024-07-01

数据仅供研究,不构成投资建议。
