import pandas as pd


def test_model_registry_lists_initial_models():
    from alpha101 import models

    assert "alpha101_equal_weight" in models.available_models()
    assert "alpha101_single_101" in models.available_models()


def test_scores_to_long_frame_has_stable_schema():
    from alpha101.models import scores_to_long_frame

    score = pd.DataFrame(
        {"000001": [1.0], "000002": [2.0]},
        index=[pd.Timestamp("2026-07-08")],
    )
    names = {"000001": "平安银行", "000002": "万科A"}

    result = scores_to_long_frame(score, "demo", names)

    assert list(result.columns) == [
        "date",
        "code",
        "name",
        "model",
        "score",
        "rank",
        "eligible",
        "reason",
    ]
    assert result.iloc[0]["code"] == "000002"
    assert result.iloc[0]["rank"] == 1


def test_run_scores_writes_multiple_models(tmp_path):
    from alpha101.models import run_scores

    dates = pd.date_range("2026-07-07", periods=2)
    panel = {
        "open": pd.DataFrame({"000001": [10.0, 10.0], "000002": [20.0, 20.0]}, index=dates),
        "high": pd.DataFrame({"000001": [11.0, 11.0], "000002": [21.0, 21.0]}, index=dates),
        "low": pd.DataFrame({"000001": [9.0, 9.0], "000002": [19.0, 19.0]}, index=dates),
        "close": pd.DataFrame({"000001": [10.5, 10.6], "000002": [20.5, 20.4]}, index=dates),
        "volume": pd.DataFrame({"000001": [1000, 1100], "000002": [1000, 1100]}, index=dates),
        "amount": pd.DataFrame({"000001": [10500, 11660], "000002": [20500, 22440]}, index=dates),
    }
    panel["vwap"] = panel["amount"] / panel["volume"]
    panel["returns"] = panel["close"].pct_change()
    output = tmp_path / "scores.csv"

    result = run_scores(
        panel,
        ["alpha101_single_101", "alpha101_equal_weight"],
        names={"000001": "平安银行", "000002": "万科A"},
        output=output,
    )

    assert output.exists()
    assert set(result["model"]) == {"alpha101_single_101", "alpha101_equal_weight"}
