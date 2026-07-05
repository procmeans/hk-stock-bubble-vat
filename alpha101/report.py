"""评估图与汇总表。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def factor_figure(name, ev, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ev["ic"].cumsum().plot(ax=ax[0], title=f"{name} 累计IC")
    qcum = (1 + ev["qret"]).cumprod()
    qcum.plot(ax=ax[1], title=f"{name} 分层净值")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def summary_table(results, out_path):
    rows = []
    for n, ev in sorted(results.items()):
        rows.append({"alpha": n, "ic_mean": ev["ic_mean"], "icir": ev["icir"],
                     "ls_annual": ev["ls_annual"], "ls_sharpe": ev["ls_sharpe"],
                     "turnover": ev["turnover"]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
