"""Visualization utility for simulator and evaluation results."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from stitching.contracts import ScenarioConfig, SurfaceTruth, SubApertureObservation
from stitching.trusted.simulator.identity import simulate_identity_observations
from stitching.trusted.scan.transforms import extract_tile


def _masked_image(values: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    """Return a masked view that hides pixels outside the provided support."""

    return np.ma.masked_where(~np.asarray(valid_mask, dtype=bool), np.asarray(values, dtype=float))


def plot_scenario_report(
    config: ScenarioConfig,
    truth: SurfaceTruth,
    observations: tuple[SubApertureObservation, ...],
    output_path: str | Path | None = None,
) -> None:
    """Generate a comprehensive visual report of the simulation."""

    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"Scenario: {config.scenario_id} - {config.description}", fontsize=16)
    masked_cmap = plt.get_cmap("viridis").copy()
    masked_cmap.set_bad(color="#f4f4f4")

    # 1. Global Truth Surface + Scan Plan
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("Global Truth Surface & Scan Plan")
    im1 = ax1.imshow(_masked_image(truth.z, truth.valid_mask), cmap=masked_cmap, origin="lower")
    plt.colorbar(im1, ax=ax1, label="Height")
    
    # Overlay sub-aperture boxes/pupils
    for obs in observations:
        rows, cols = obs.tile_shape
        # Center_xy is (x, y) in global pixels
        cx, cy = obs.center_xy
        # Rectangle origin (lower left)
        # Note: center_xy is geometric center
        rect = plt.Rectangle(
            (cx - (cols-1)/2.0, cy - (rows-1)/2.0),
            cols, rows,
            fill=False, color="red", alpha=0.3, lw=1
        )
        ax1.add_patch(rect)
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    # 2. Zoom on a single observation (The first one)
    obs = observations[0]
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title(f"Observation 0 (Instrument view)")
    im2 = ax2.imshow(_masked_image(obs.z, obs.valid_mask), cmap=masked_cmap, origin="lower")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("Local X")
    ax2.set_ylabel("Local Y")

    # 3. Discrepancy Map (Obs 0 vs Truth)
    # Extract perfect local truth for comparison
    interpolation_order = int(config.metadata.get("interpolation_order", 3))
    local_truth_z, local_truth_mask = extract_tile(
        truth.z,
        truth.valid_mask,
        obs.tile_shape,
        obs.center_xy,
        rotation_deg=obs.rotation_deg,
        interpolation_order=interpolation_order,
    )
    
    # Difference (Nuisances + Noise + Drift + Interpolation)
    diff_mask = obs.valid_mask & local_truth_mask & np.isfinite(obs.z) & np.isfinite(local_truth_z)
    diff = np.zeros(obs.tile_shape, dtype=float)
    diff[diff_mask] = obs.z[diff_mask] - local_truth_z[diff_mask]
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("Diff: Obs 0 - Local Truth")
    im3 = ax3.imshow(_masked_image(diff, diff_mask), cmap=masked_cmap, origin="lower")
    plt.colorbar(im3, ax=ax3, label="Error")
    ax3.set_xlabel("Local X")

    # 4. Mismatch Map (Consistency between overlaps)
    from stitching.trusted.eval.mismatch import compute_mismatch_map
    std_map, count_z = compute_mismatch_map(observations)
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Mismatch Map (Local Std Dev)")
    im4 = ax4.imshow(_masked_image(std_map, count_z > 1), cmap=masked_cmap, origin="lower")
    plt.colorbar(im4, ax=ax4, label="Std Dev")
    ax4.set_xlabel("X [pixels]")

    # 5. Overlap Count
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Overlap Count")
    im5 = ax5.imshow(_masked_image(count_z, count_z > 0), cmap=masked_cmap, origin="lower")
    plt.colorbar(im5, ax=ax5, label="Count")
    ax5.set_xlabel("X [pixels]")

    # 6. Signal Profile (Cross section)
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Signal Cross-section (Y middle)")
    mid_y = truth.z.shape[0] // 2
    ax6.plot(truth.z[mid_y, :], label="Truth", lw=2, color="black")
    # Project some observations onto the profile if they cross mid_y
    # For simplicity, just show the truth
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Report saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Path to scenario YAML")
    parser.add_argument("--output", help="Path to save PNG", default="report.png")
    args = parser.parse_args()

    config = ScenarioConfig.from_yaml(args.scenario)
    truth, observations = simulate_identity_observations(config)
    plot_scenario_report(config, truth, observations, output_path=args.output)
