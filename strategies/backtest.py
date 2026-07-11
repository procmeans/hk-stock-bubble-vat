"""执行模拟层:信号次日生效 + 换手成本(对应 zipline 的 commission/slippage 模型)。"""
import pandas as pd


def run(weights, panel, cost_bps=20, slippage_bps=0):
    """weights: date×code 目标权重(T 日收盘信号,T+1 日起持有)。"""
    rets = panel["close"].pct_change()
    w = weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps + slippage_bps) / 1e4
    equity = (1.0 + net).cumprod()
    return pd.DataFrame(
        {"gross": gross, "net": net, "turnover": turnover, "equity": equity}
    )
