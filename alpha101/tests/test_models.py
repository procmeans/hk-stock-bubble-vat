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
