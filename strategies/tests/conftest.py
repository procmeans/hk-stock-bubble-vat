import pandas as pd
import pytest


@pytest.fixture
def make_panel():
    def _make(prices: dict, volume: float = 1000.0) -> dict:
        n = len(next(iter(prices.values())))
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.DataFrame(prices, index=idx, dtype=float)
        return {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": close * 0 + volume,
            "amount": close * volume, "vwap": close,
            "returns": close.pct_change(),
        }
    return _make
