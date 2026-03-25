"""Visualize the evolution of an autoresearch `results.tsv` file.

The script is intentionally self-contained so it can be run directly from the
repo root without depending on the harness internals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("commit", "aggregate_rms", "total_runtime_sec", "status", "description")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the evolution of an autoresearch results.tsv file.",
    )
    parser.add_argument(
        "tsv_path",
        nargs="?",
        default="autoresearch/results.tsv",
        help="Path to the tab-separated results file.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/results_tsv_progress.png",
        help="Path to the output image.",
    )
    parser.add_argument(
        "--metric",
        default="aggregate_rms",
        help="Numeric metric column to plot. Lower is treated as better.",
    )
    parser.add_argument(
        "--title",
        default="Autoresearch results.tsv progress",
        help="Plot title.",
    )
    parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="Also plot discard/crash rows instead of showing keep-only progress.",
    )
    return parser.parse_args()


def _load_results(tsv_path: Path) -> pd.DataFrame:
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing results file: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {tsv_path}: {', '.join(missing)}")

    df = df.copy()
    df["exp"] = np.arange(1, len(df) + 1, dtype=int)
    df["metric_num"] = pd.to_numeric(df["aggregate_rms"], errors="coerce")
    df["runtime_num"] = pd.to_numeric(df["total_runtime_sec"], errors="coerce")
    return df


def _format_commit(commit: str) -> str:
    commit = str(commit)
    return commit[:7] if len(commit) > 7 else commit


def _plot_results(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
    include_discarded: bool,
) -> None:
    if metric_column not in df.columns and metric_column != "aggregate_rms":
        raise ValueError(f"Metric column not found: {metric_column}")

    metric = df["metric_num"] if metric_column == "aggregate_rms" else pd.to_numeric(
        df[metric_column], errors="coerce"
    )
    plot_df = df.loc[metric.notna()].copy()
    plot_df["metric"] = metric.loc[metric.notna()]
    if plot_df.empty:
        raise ValueError("No numeric metric values found to plot.")

    if include_discarded:
        plot_df = plot_df.copy()
        plot_df["running_best"] = plot_df["metric"].cummin()
        status_order = ["keep", "discard", "crash"]
    else:
        plot_df = plot_df[plot_df["status"] == "keep"].copy()
        if plot_df.empty:
            raise ValueError("No keep rows found to plot.")
        plot_df["running_best"] = plot_df["metric"].cummin()
        status_order = ["keep"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.grid(True, alpha=0.2)

    # Plot the metric as a faint line so changes over time remain visible.
    ax.plot(
        plot_df["exp"],
        plot_df["metric"],
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.55,
        label=f"{metric_column} per keep" if not include_discarded else f"{metric_column} per run",
    )

    # Scatter points by status to mirror the bookkeeping when requested.
    for status in status_order:
        subset = plot_df[plot_df["status"] == status]
        if subset.empty:
            continue
        color = {"keep": "#2ecc71", "discard": "#e67e22", "crash": "#e74c3c"}[status]
        ax.scatter(
            subset["exp"],
            subset["metric"],
            s=42,
            color=color,
            alpha=0.9 if status == "keep" else 0.65,
            edgecolor="white",
            linewidth=0.5,
            label=status,
            zorder=4,
        )

    ax.step(
        plot_df["exp"],
        plot_df["running_best"],
        where="post",
        color="#1f77b4",
        linewidth=2.4,
        label="running best",
        zorder=3,
    )

    # Annotate kept runs with short commit hashes, which is usually what matters when scanning history.
    for _, row in plot_df[plot_df["status"] == "keep"].iterrows():
        ax.annotate(
            _format_commit(row["commit"]),
            (row["exp"], row["metric"]),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=8,
            color="#145a32",
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(metric_column)
    ax.legend(loc="best")
    ax.set_xlim(left=1)

    # Add a small runtime overlay so slow regressions are easy to spot.
    ax2 = ax.twinx()
    ax2.plot(
        plot_df["exp"],
        plot_df["runtime_num"],
        color="#8e44ad",
        linewidth=1.0,
        alpha=0.35,
        linestyle="--",
        label="runtime",
    )
    ax2.set_ylabel("total_runtime_sec")

    # Merge legends from both axes.
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    tsv_path = Path(args.tsv_path)
    output_path = Path(args.output)

    df = _load_results(tsv_path)
    _plot_results(df, args.metric, args.title, output_path, args.include_discarded)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
