"""行情装载层(对应 zipline 的 data bundle):统一面板口径。"""
from pathlib import Path

import pandas as pd

CACHES = {
    "us": Path("alpha101/cache/yf_panel_us.pkl"),
    "hk": Path("alpha101/cache/yf_panel_hk.pkl"),
    "a": Path("alpha101/cache/ths_panel.pkl"),
}
FETCH_HINTS = {
    "us": "python -m alpha101.yf_history fetch --market us",
    "hk": "python -m alpha101.yf_history fetch --market hk",
    "a": "python -m alpha101.ths_history fetch",
}


def load_panel(market: str, top: int | None = None, cache=None) -> dict:
    cache = Path(cache) if cache else CACHES[market]
    if not cache.exists():
        raise FileNotFoundError(f"缓存不存在 {cache},先运行: {FETCH_HINTS[market]}")
    raw = (
        pd.read_parquet(cache)
        if cache.suffix.lower() == ".parquet"
        else pd.read_pickle(cache)
    )
    if market == "a":
        from alpha101.ths_history import build_panel
    else:
        from alpha101.yf_history import build_panel
    panel = build_panel(raw)
    if top:
        keep = panel["amount"].tail(60).mean().nlargest(top).index
        panel = {
            key: value[keep] if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
    return panel
