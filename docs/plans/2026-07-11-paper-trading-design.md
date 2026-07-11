# 模拟交易设计(paper trading · 美股 momentum)

日期:2026-07-11
状态:已确认

## 定位

路线图第四站:让验证过的策略(美股 momentum,lookback=126/top_n=40,
唯一拿到显著性的)进入**前向纸面实盘**。`paper/<account>/nav.csv` 的逐日净值
就是策略的样本外终审。GitHub Actions 每日自动运行(沿用 update.yml 模式)。

## 账户模型(文件化,全部进 git)

```
paper/us_momentum/
  state.json    # cash、positions{ticker:股数}、pending_targets、
                # days_since_rebalance、bench_nav、last_run
  nav.csv       # date, nav, cash, positions_value, bench_nav
  orders.csv    # date, ticker, side, shares, price, value, cost
```

- 初始虚拟资金 $100,000;允许碎股(纸面简化);成本单边 20bp。
- 调仓节奏由 state 的交易日计数器管理(每 21 个交易日),
  不用 momentum.signal 内部日历(其调仓日随窗口漂移,不适合增量运行)。

## 每日流程(strategies/paper.py,本地与 Actions 同一命令)

`python -m strategies.paper run [--account us_momentum]`:

1. **数据**:市值前 800(读当日快照 data/us-*.json,Actions 已在维护)∪ 当前持仓,
   yf.download 最近 400 自然日 OHLCV;按 60 日均成交额(close×volume)取前 500
   为股票池(与验证口径一致)。Actions 上不依赖本地 300MB 缓存。
2. **执行昨日挂单**:pending_targets 以**今日收盘价**成交
   (T 日信号 → T+1 收盘成交,与回测时序一致);目标市值 = 权重 × 成交时 NAV,
   差额买卖,|成交额|×20bp 从现金扣。
3. **记净值**:持仓按今日收盘估值(停牌股用最近价 ffill);
   基准净值按股票池等权日收益复利,同 CSV 逐日对照。
4. **调仓判定**:交易日计数 ≥ 21 或从未建仓 → `momentum.score` 最新截面
   top 40 等权 → 写入 pending_targets(明日成交),计数清零。
5. 追加 nav.csv / orders.csv,写回 state.json。
6. **幂等**:state.last_run == 面板最新交易日 → 直接跳过(重复运行无副作用)。

## CLI

```
python -m strategies.paper init --capital 100000   # 建账户(已存在则拒绝)
python -m strategies.paper run                     # 每日步进
python -m strategies.paper status                  # 打印持仓与净值摘要
```

## GitHub Actions(.github/workflows/paper.yml)

- cron `0 22 * * 1-5`(UTC,美股收盘后)+ workflow_dispatch;
- 步骤:checkout → python 3.11 → `pip install pandas numpy yfinance`
  → `python -m strategies.paper run` → commit & push `paper/`;
- concurrency 防重入;yfinance 限流时 run 以非零退出,靠下次 cron 重试
  (幂等保证补跑安全,漏一天净值次日按最新价补记)。

## 测试(合成数据,离线)

- 会计恒等:现金 + 持仓市值 = NAV,逐步进保持;
- 挂单次日成交:下单日 NAV 不变,成交日按当日价、扣 20bp;
- 调仓节奏:第 1 天建仓,第 22 个交易日再调仓;
- 幂等:同面板重复 step 无变化;
- 停牌 ffill 估值;target_weights 取动量 top N 等权。
- `run` 的抓数函数注入(测试替身),不联网。

## 免责

纸面模拟:无滑点冲击、允许碎股、以收盘价成交;研究用途,不构成投资建议。
