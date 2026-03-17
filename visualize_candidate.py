
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stitching.harness.evaluator import evaluate_candidate_on_scenario, load_candidate_module

def main():
    candidate_path = Path("src/stitching/editable/candidate_current.py")
    scenario_path = Path("scenarios/s17_highres_circular.yaml")
    
    print(f"Loading candidate: {candidate_path}")
    candidate = load_candidate_module(candidate_path)
    
    print(f"Evaluating scenario: {scenario_path}")
    report = evaluate_candidate_on_scenario(candidate, scenario_path)
    
    truth = report.truth
    recon = report.reconstruction
    
    # Common mask
    mask = truth.valid_mask & recon.valid_mask
    
    # Detrend for visualization (remove global piston and tilt to see stitching quality)
    def detrend(data, mask):
        yy, xx = np.indices(data.shape, dtype=float)
        y_vals = yy[mask]
        x_vals = xx[mask]
        z_vals = data[mask]
        A = np.column_stack([x_vals, y_vals, np.ones_like(x_vals)])
        coeff, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
        
        result = np.full_like(data, np.nan)
        result[mask] = z_vals - A @ coeff
        return result

    z_truth_plot = detrend(truth.z, mask)
    z_recon_plot = detrend(recon.z, mask)
    z_diff = z_recon_plot - z_truth_plot
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(z_truth_plot, origin='lower')
    axes[0].set_title(f"Truth (Detrended)\n{scenario_path.stem}")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(z_recon_plot, origin='lower')
    axes[1].set_title(f"Reconstruction (Detrended)\nRMS: {report.signal_metrics['rms_on_valid_intersection']:.4f}")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(z_diff, origin='lower', cmap='RdBu_r')
    axes[2].set_title(f"Difference (Recon - Truth)\nStitching Error")
    fig.colorbar(im2, ax=axes[2])
    
    output_path = Path("artifacts/manual_verification_s17.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
