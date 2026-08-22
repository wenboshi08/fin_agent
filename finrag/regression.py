"""Experiment regression tracking: CSV + comparison chart."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_REGRESSION_CSV = config.RESULTS_DIR / "regression.csv"


def append_result(
    *,
    dataset: str,
    config_name: str,
    metric: str,
    value: float,
    extra: dict | None = None,
) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "config": config_name,
        "metric": metric,
        "value": value,
        **(extra or {}),
    }
    write_header = not _REGRESSION_CSV.exists() or _REGRESSION_CSV.stat().st_size == 0
    with _REGRESSION_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Regression recorded: %s/%s %s=%.4f", dataset, config_name, metric, value)


def compare_and_chart(*, datasets: list[str], metric: str = "ndcg@10", output: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not _REGRESSION_CSV.exists():
        raise FileNotFoundError(f"No regression CSV found: {_REGRESSION_CSV}")

    df = pd.read_csv(_REGRESSION_CSV)
    df = df[(df["dataset"].isin(datasets)) & (df["metric"] == metric)]
    if df.empty:
        raise ValueError(f"No rows for datasets={datasets}, metric={metric}")

    pivot = df.pivot_table(index="config", columns="dataset", values="value", aggfunc="last")
    pivot = pivot.reindex(columns=datasets)
    pivot = pivot.sort_values(by=datasets[0], ascending=False)

    out_path = Path(output or (config.RESULTS_DIR / f"{metric}_comparison.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ax = pivot.plot(kind="bar", figsize=(max(8, len(datasets) * 2), 6), title=f"{metric} by config")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Comparison chart saved -> %s", out_path)
    return out_path
