"""Karpathy-style autoresearch progress visualization.

Plots iteration-by-iteration performance improvement with hypothesis annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_progress_plot(
    experiments_dir: Path,
    output_path: Path,
    metric_key: str = "aggregate_rms",
) -> None:
    """Read the experiments/runs directory and plot progress."""
    runs_dir = experiments_dir / "runs"
    if not runs_dir.exists():
        logger.error("No runs directory found in %s", experiments_dir)
        return

    data = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"

        if not (manifest_path.exists() and metrics_path.exists()):
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            data.append({
                "iteration": manifest["iteration"],
                "metric": metrics.get(metric_key, np.nan),
                "verdict": manifest.get("verdict", "rejected"),
                "hypothesis": manifest.get("hypothesis", ""),
                "timestamp": manifest["timestamp_utc"],
            })
        except Exception as exc:
            logger.warning("Failed to parse run in %s: %s", run_dir, exc)

    if not data:
        logger.warning("No valid run data found.")
        return

    df = pd.DataFrame(data).sort_values("iteration")
    df = df[df["metric"].notna()]

    # Calculate running best
    df["running_best"] = df["metric"].cummin()

    # Identify kept (accepted) improvements
    accepted = df[df["verdict"] == "accepted"].copy()
    discarded = df[df["verdict"] != "accepted"].copy()

    # Plotting
    plt.figure(figsize=(15, 8), dpi=150)
    plt.grid(True, alpha=0.2)

    # 1. Discarded experiments (gray dots)
    plt.scatter(
        discarded["iteration"],
        discarded["metric"],
        color="gray",
        alpha=0.3,
        s=20,
        label="Discarded / Regression",
    )

    # 2. Running best line (green staircase)
    plt.step(
        df["iteration"],
        df["running_best"],
        where="post",
        color="#2ecc71",
        linewidth=2,
        alpha=0.8,
        label="Running best",
    )

    # 3. Accepted improvements (green circles)
    plt.scatter(
        accepted["iteration"],
        accepted["metric"],
        color="#2ecc71",
        edgecolor="#27ae60",
        s=80,
        zorder=5,
        label="Kept Improvement",
    )

    # 4. Annotations for accepted improvements
    for _, row in accepted.iterrows():
        # Shorten hypothesis for display
        short_h = row["hypothesis"]
        if len(short_h) > 50:
            short_h = short_h[:47] + "..."

        plt.annotate(
            short_h,
            (row["iteration"], row["metric"]),
            xytext=(5, 10),
            textcoords="offset points",
            rotation=30,
            fontsize=8,
            color="#27ae60",
            alpha=0.8,
        )

    num_kept = len(accepted)
    num_total = len(df)
    plt.title(f"Autoresearch Progress: {num_total} Experiments, {num_kept} Kept Improvements", fontsize=16)
    plt.xlabel("Iteration #", fontsize=12)
    plt.ylabel(f"Metric: {metric_key} (lower is better)", fontsize=12)
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved progress plot to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    repo_root = Path.cwd()
    generate_progress_plot(
        repo_root / "experiments",
        repo_root / "artifacts" / "autoresearch_progress.png",
    )
