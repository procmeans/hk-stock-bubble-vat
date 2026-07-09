import pandas as pd

from alpha101.ths_http import tables_to_dataframe
from alpha101.ths_today import normalize_ths_spot, score_today_alpha101


def test_tables_to_dataframe_flattens_ifind_payload():
    payload = {
        "tables": [
            {
                "thscode": "000001.SZ",
                "time": ["2026-07-08 15:00:00"],
                "table": {"open": [10.0], "latest": [10.5]},
            }
        ]
    }

    result = tables_to_dataframe(payload)

    assert result.loc[0, "thscode"] == "000001.SZ"
    assert result.loc[0, "time"] == "2026-07-08 15:00:00"
    assert result.loc[0, "latest"] == 10.5


def test_tables_to_dataframe_handles_none_indicator_values():
    payload = {
        "tables": [
            {
                "thscode": "000001.SZ",
                "time": ["2026-07-08"],
                "table": {"open": [10.0], "amount": None},
            }
        ]
    }

    result = tables_to_dataframe(payload)

    assert result.loc[0, "open"] == 10.0
    assert pd.isna(result.loc[0, "amount"])


def test_normalize_ths_spot_maps_today_quotation_columns():
    data = pd.DataFrame({
        "thscode": ["000001.SZ"],
        "open": [10.0],
        "high": [11.0],
        "low": [9.0],
        "latest": [10.5],
    })
    names = pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})

    result = normalize_ths_spot(data, names)

    assert result.loc[0, "code"] == "000001"
    assert result.loc[0, "name"] == "平安银行"
    assert result.loc[0, "open"] == 10.0
    assert result.loc[0, "close"] == 10.5


def test_score_today_alpha101_ranks_higher_factor_first():
    data = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "name": ["A", "B", "C"],
        "open": [10.0, 10.0, 10.0],
        "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0],
        "close": [9.5, 10.0, 10.5],
    })

    result = score_today_alpha101(data, top_n=2)

    assert result["code"].tolist() == ["000003", "000002"]
    assert result["rank"].tolist() == [1, 2]
    assert result["score"].tolist() == [100.0, 66.67]
