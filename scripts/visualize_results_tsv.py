"""Visualize the evolution of an autoresearch `results.tsv` file.

The script is intentionally self-contained so it can be run directly from the
repo root without depending on the harness internals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


REQUIRED_COLUMNS = ("commit", "aggregate_rms", "total_runtime_sec", "status", "description")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src_to_path(repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


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
    parser.add_argument(
        "--compare-gls-robust",
        default="src/stitching/editable/gls_robust/baseline.py",
        help="Candidate file used for the GLS Robust error panel.",
    )
    parser.add_argument(
        "--compare-latest",
        default="src/stitching/editable/optimized_stitching_algo.py",
        help="Candidate file used for the latest-candidate error panel.",
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/s17_highres_circular.yaml",
        help="Scenario used to generate the error panels.",
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


def _friendly_label(row: pd.Series) -> str:
    description = str(row.get("description", "")).strip()
    commit = _format_commit(row.get("commit", ""))

    if description.endswith("baseline on s17_highres_circular"):
        return description.replace(" baseline on s17_highres_circular", "")
    if description == "baseline setup verification on s17_highres_circular":
        return "bb06"
    return commit


def detrend(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove global piston and tilt for visualization."""
    effective_mask = mask & ~np.isnan(data)
    if not np.any(effective_mask):
        return np.full_like(data, np.nan)

    yy, xx = np.indices(data.shape, dtype=float)
    y_vals = yy[effective_mask]
    x_vals = xx[effective_mask]
    z_vals = data[effective_mask]

    A = np.column_stack([x_vals, y_vals, np.ones_like(x_vals)])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
        y_all = yy[mask]
        x_all = xx[mask]
        A_all = np.column_stack([x_all, y_all, np.ones_like(x_all)])
        result = np.full_like(data, np.nan)
        result[mask] = data[mask] - (A_all @ coeff)
        return result
    except np.linalg.LinAlgError:
        return np.full_like(data, np.nan)


def _candidate_error_map(candidate_path: Path, scenario_path: Path) -> tuple[np.ndarray, float]:
    from stitching.harness.evaluator import evaluate_candidate_on_scenario, load_candidate_module

    candidate = load_candidate_module(candidate_path)
    report = evaluate_candidate_on_scenario(candidate, scenario_path)

    truth = report.truth
    recon = report.reconstruction
    mask = truth.valid_mask & recon.valid_mask
    z_truth_plot = detrend(truth.z, mask)
    z_recon_plot = detrend(recon.z, mask)
    z_diff = z_recon_plot - z_truth_plot

    rms = report.signal_metrics.get("rms_detrended", float("nan"))
    if np.isnan(rms) and np.any(~np.isnan(z_diff)):
        rms = float(np.sqrt(np.nanmean(z_diff**2)))
    return z_diff, float(rms)


def _plot_results(
    df: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
    include_discarded: bool,
    scenario_path: Path,
    compare_gls_robust: Path,
    compare_latest: Path,
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

    fig, ax = plt.subplots(figsize=(17, 8.5), dpi=150)
    ax.grid(True, alpha=0.2)

    ax.plot(
        plot_df["exp"],
        plot_df["metric"],
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.55,
        label=f"{metric_column} per keep" if not include_discarded else f"{metric_column} per run",
    )

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

    keep_rows = plot_df[plot_df["status"] == "keep"].reset_index(drop=True)
    label_offsets = [(4, 7), (4, -10), (4, 7), (4, -10)]
    for idx, (_, row) in enumerate(keep_rows.iterrows()):
        ax.annotate(
            _friendly_label(row),
            (row["exp"], row["metric"]),
            textcoords="offset points",
            xytext=label_offsets[idx % len(label_offsets)],
            fontsize=7,
            color="#145a32",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
        )

    ax.set_title(title)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(metric_column)
    ax.set_xlim(left=1)

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

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    try:
        repo_root = _repo_root()
        _add_src_to_path(repo_root)
        scenario_path = scenario_path if scenario_path.is_absolute() else repo_root / scenario_path

        gls_diff, gls_rms = _candidate_error_map(
            compare_gls_robust if compare_gls_robust.is_absolute() else repo_root / compare_gls_robust,
            scenario_path,
        )
        latest_diff, latest_rms = _candidate_error_map(
            compare_latest if compare_latest.is_absolute() else repo_root / compare_latest,
            scenario_path,
        )

        gls_vals = gls_diff[np.isfinite(gls_diff)]
        latest_vals = latest_diff[np.isfinite(latest_diff)]
        gls_absmax = float(np.max(np.abs(gls_vals))) if gls_vals.size else 1.0
        latest_absmax = float(np.max(np.abs(latest_vals))) if latest_vals.size else 1.0
        if not np.isfinite(gls_absmax) or gls_absmax <= 0.0:
            gls_absmax = 1.0
        if not np.isfinite(latest_absmax) or latest_absmax <= 0.0:
            latest_absmax = 1.0

        ax_gls = ax.inset_axes([5.0, 1.7, 3.2, 0.30], transform=ax.transData)
        ax_latest = ax.inset_axes([25, 1.3, 3.2, 0.30], transform=ax.transData)

        ax_gls.imshow(gls_diff, origin="lower", cmap="RdBu_r", vmin=-gls_absmax, vmax=gls_absmax)
        ax_latest.imshow(latest_diff, origin="lower", cmap="RdBu_r", vmin=-latest_absmax, vmax=latest_absmax)

        ax_gls.set_title(f"GLS Robust\n{gls_rms:.3f} nm RMS", fontsize=8)
        ax_latest.set_title(f"Latest\n{latest_rms:.3f} nm RMS", fontsize=8)
        for ax_map in (ax_gls, ax_latest):
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            for spine in ax_map.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(0.8)

    except Exception as exc:
        ax.text(
            0.5,
            0.08,
            f"Error panels unavailable: {exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e74c3c", alpha=0.85),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    tsv_path = Path(args.tsv_path)
    output_path = Path(args.output)

    df = _load_results(tsv_path)
    _plot_results(
        df,
        args.metric,
        args.title,
        output_path,
        args.include_discarded,
        Path(args.scenario),
        Path(args.compare_gls_robust),
        Path(args.compare_latest),
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
