"""Deterministic CSV, Markdown, and headless figure outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TABLE_FILES = {
    "factor_summary": "factor_summary.csv",
    "daily_ic": "daily_ic.csv",
    "quantile_returns": "quantile_returns.csv",
    "portfolio_nav": "portfolio_nav.csv",
    "trades": "trades.csv",
    "data_coverage": "data_coverage.csv",
}


def _numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include="number")


def _ic_figure_data(results: dict) -> pd.DataFrame:
    frame = results["daily_ic"].copy()
    if "date" in frame:
        frame = frame.set_index("date")
    return _numeric_columns(frame).cumsum()


def _quantile_figure_data(results: dict) -> pd.DataFrame:
    frame = results["quantile_returns"].copy()
    required = {"factor", "date", "group", "return"}
    if required.issubset(frame.columns):
        frame = frame.pivot_table(
            index="date",
            columns=["factor", "group"],
            values="return",
            aggfunc="mean",
        )
    return _numeric_columns(frame)


def _nav_figure_data(results: dict) -> pd.DataFrame:
    frame = results["portfolio_nav"].copy()
    if "date" in frame:
        frame = frame.set_index("date")
    return _numeric_columns(frame)


def _save_figure(frame: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if frame.empty or frame.shape[1] == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        frame.plot(ax=ax)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_figures(results: dict, output: Path) -> list[Path]:
    figures = {
        "factor_ic.png": ("Cumulative factor IC", _ic_figure_data(results)),
        "factor_quantiles.png": (
            "Factor quantile returns",
            _quantile_figure_data(results),
        ),
        "portfolio_nav.png": ("Portfolio NAV", _nav_figure_data(results)),
    }
    paths = []
    for filename, (title, frame) in figures.items():
        path = output / filename
        _save_figure(frame, title, path)
        paths.append(path)
    return paths


def write_outputs(results: dict, output_dir) -> list[Path]:
    """Write the fixed ten-file validation-report contract."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, filename in TABLE_FILES.items():
        path = output / filename
        results[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)

    metric_lines = [
        f"- {key}: {value}"
        for key, value in sorted(results["portfolio_metrics"].items())
    ]
    disclosure_lines = [
        f"- {item}" for item in results.get("disclosures", [])
    ]
    markdown = (
        "# A-share intraday factor six-month validation\n\n"
        "## Portfolio metrics\n\n"
        + "\n".join(metric_lines)
        + "\n\n## Limitations and disclosures\n\n"
        + "\n".join(disclosure_lines)
        + "\n"
    )
    report_path = output / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    paths.append(report_path)
    paths.extend(_write_figures(results, output))
    return paths
