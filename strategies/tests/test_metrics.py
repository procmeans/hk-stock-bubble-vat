import numpy as np
import pandas as pd
import pytest

from strategies import metrics


def test_summary_hand_computed():
    net = pd.Series([0.0, 0.01, -0.02, 0.01])
    result = pd.DataFrame({
        "gross": net, "net": net,
        "turnover": pd.Series([0.0, 1.0, 0.0, 0.5]),
        "equity": (1 + net).cumprod(),
    })

    stats = metrics.summary(result)

    equity = (1 + net).cumprod()
    total = equity.iloc[-1] - 1
    assert stats["total_return"] == pytest.approx(total)
    assert stats["annual_return"] == pytest.approx((1 + total) ** (252 / 4) - 1)
    assert stats["annual_vol"] == pytest.approx(net.std(ddof=0) * np.sqrt(252))
    # 最大回撤:峰值 1.01 -> 谷底 1.01*0.98
    assert stats["max_drawdown"] == pytest.approx(-0.02)
    assert stats["annual_turnover"] == pytest.approx(1.5 * 252 / 4)
