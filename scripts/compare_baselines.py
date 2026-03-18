import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stitching.harness.evaluator import evaluate_candidate_on_scenario, load_candidate_module

def detrend(data, mask):
    """Remove global piston and tilt for visualization."""
    yy, xx = np.indices(data.shape, dtype=float)
    y_vals = yy[mask]
    x_vals = xx[mask]
    z_vals = data[mask]
    
    # Fit plane: z = a*x + b*y + c
    A = np.column_stack([x_vals, y_vals, np.ones_like(x_vals)])
    coeff, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
    
    result = np.full_like(data, np.nan)
    result[mask] = z_vals - (A @ coeff)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    args = parser.parse_args()
    
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path}")
        sys.exit(1)

    baselines = {
        "GLS Standard": "src/stitching/editable/gls/baseline.py",
        "GLS Robust (Huber)": "src/stitching/editable/gls_robust/baseline.py",
        "SCS (Calibration)": "src/stitching/editable/scs/baseline.py",
        "SIAC (Alternating)": "src/stitching/editable/siac/baseline.py",
        "PSO (Stochastic)": "src/stitching/editable/pso/baseline.py",
    }

    n_algos = len(baselines)
    # 1 row for GT, then 1 row per algorithm (Recon + Diff)
    fig, axes = plt.subplots(n_algos + 1, 3, figsize=(18, 5 * (n_algos + 1)))
    
    # First row: Ground Truth
    gt_done = False
    
    for row_idx, (name, path) in enumerate(baselines.items(), start=1):
        print(f"Processing {name}...")
        candidate_path = Path(path)
        if not candidate_path.exists():
            print(f"Skipping {name}, file not found: {path}")
            continue
            
        candidate = load_candidate_module(candidate_path)
        report = evaluate_candidate_on_scenario(candidate, scenario_path)
        
        truth = report.truth
        recon = report.reconstruction
        mask = truth.valid_mask & recon.valid_mask
        
        z_truth_plot = detrend(truth.z, mask)
        z_recon_plot = detrend(recon.z, mask)
        z_diff = z_recon_plot - z_truth_plot
        
        # Plot GT only once
        if not gt_done:
            im0 = axes[0, 0].imshow(z_truth_plot, origin='lower')
            axes[0, 0].set_title(f"Ground Truth (Detrended)\n{scenario_path.stem}")
            fig.colorbar(im0, ax=axes[0, 0])
            axes[0, 1].axis('off')
            axes[0, 2].axis('off')
            gt_done = True
            
        # Plot Reconstruction
        im1 = axes[row_idx, 0].imshow(z_recon_plot, origin='lower')
        axes[row_idx, 0].set_title(f"{name}\nReconstruction")
        fig.colorbar(im1, ax=axes[row_idx, 0])
        
        # Plot Difference
        rms = report.signal_metrics.get('rms_detrended', float('nan'))
        im2 = axes[row_idx, 1].imshow(z_diff, origin='lower', cmap='RdBu_r')
        axes[row_idx, 1].set_title(f"{name} Error (Diff to GT)\nRMS_detrend: {rms:.6f}")
        fig.colorbar(im2, ax=axes[row_idx, 1])
        
        # Plot mask/support (optional but useful for s17)
        im3 = axes[row_idx, 2].imshow(recon.valid_mask.astype(float), origin='lower', cmap='gray')
        axes[row_idx, 2].set_title(f"{name}\nValid Mask")
        
    output_path = Path("artifacts/comparison_all_algos_s17.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nComparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()
