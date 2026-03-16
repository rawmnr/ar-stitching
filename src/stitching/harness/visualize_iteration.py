"""Visualization for iteration-level performance and surface comparisons."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence

from stitching.contracts import EvalReport

def _masked_image(values: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    """Return a masked view that hides pixels outside the provided support."""
    return np.ma.masked_where(~np.asarray(valid_mask, dtype=bool), np.asarray(values, dtype=float))

def plot_iteration_report(
    reports: Sequence[EvalReport],
    output_path: Path,
    iteration: int,
    agent_name: str,
) -> None:
    """Generate a summary visualization for an iteration's performance."""
    num_scenarios = len(reports)
    if num_scenarios == 0:
        return

    # Create a grid of plots: 3 columns (Truth, Reconstructed, Diff) x num_scenarios rows
    fig, axes = plt.subplots(num_scenarios, 3, figsize=(15, 5 * num_scenarios), squeeze=False)
    plt.suptitle(f"Iteration {iteration} - Agent: {agent_name}", fontsize=20)

    masked_cmap = plt.get_cmap("viridis").copy()
    masked_cmap.set_bad(color="#f4f4f4")
    diff_cmap = plt.get_cmap("RdBu").copy()
    diff_cmap.set_bad(color="#f4f4f4")

    for i, report in enumerate(reports):
        truth = report.truth
        recon = report.reconstruction
        config = report.config

        # 1. Global Truth
        ax_truth = axes[i, 0]
        ax_truth.set_title(f"{config.scenario_id}: Global Truth")
        im_truth = ax_truth.imshow(_masked_image(truth.z, truth.valid_mask), cmap=masked_cmap, origin="lower")
        plt.colorbar(im_truth, ax=ax_truth)

        # 2. Reconstructed Surface
        ax_recon = axes[i, 1]
        ax_recon.set_title(f"Reconstructed (RMS: {report.signal_metrics['rms_on_valid_intersection']:.4f})")
        im_recon = ax_recon.imshow(_masked_image(recon.z, recon.valid_mask), cmap=masked_cmap, origin="lower")
        plt.colorbar(im_recon, ax=ax_recon)

        # 3. Difference (within valid intersection)
        ax_diff = axes[i, 2]
        ax_diff.set_title("Difference (Recon - Truth)")
        
        # Calculate diff only where both are valid
        common_mask = truth.valid_mask & recon.valid_mask
        diff = np.zeros_like(truth.z)
        diff[common_mask] = recon.z[common_mask] - truth.z[common_mask]
        
        # Normalize diff colorbar to be symmetric if there are values
        vmax = np.max(np.abs(diff[common_mask])) if np.any(common_mask) else 0.1
        im_diff = ax_diff.imshow(_masked_image(diff, common_mask), cmap=diff_cmap, origin="lower", vmin=-vmax, vmax=vmax)
        plt.colorbar(im_diff, ax=ax_diff)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
