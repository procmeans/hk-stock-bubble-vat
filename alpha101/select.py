"""每日选股清单。"""
import pandas as pd


def pick(score, date, names=None, top_n=50):
    date = pd.Timestamp(date)
    row = score.loc[date].dropna().sort_values(ascending=False).head(top_n)
    df = pd.DataFrame({"code": row.index, "score": row.values})
    df["rank"] = range(1, len(df) + 1)
    df["name"] = df["code"].map(names) if names else ""
    return df[["code", "name", "score", "rank"]]
