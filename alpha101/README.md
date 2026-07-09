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

数据仅供研究,不构成投资建议。
