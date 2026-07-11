"""绩效指标层(对应 pyfolio 的 tear sheet,MVP 只保留核心指标)。"""
import numpy as np

TRADING_DAYS = 252


def summary(result) -> dict:
    net, equity = result["net"], result["equity"]
    n = max(len(net), 1)
    total = equity.iloc[-1] - 1.0
    annual = (1.0 + total) ** (TRADING_DAYS / n) - 1.0
    vol = net.std(ddof=0) * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": annual / vol if vol > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "annual_turnover": result["turnover"].sum() * TRADING_DAYS / n,
    }
