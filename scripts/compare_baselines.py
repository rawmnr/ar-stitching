import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from stitching.harness.evaluator import evaluate_candidate_on_scenario, load_candidate_module
from stitching.contracts import ScenarioConfig
from stitching.trusted.instrument.bias import generate_reference_bias_field


def detrend(data, mask):
    """Remove global piston and tilt for visualization."""
    # Ensure mask only includes non-NaN data
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
        
        # Apply trend to all pixels in the original mask
        y_all = yy[mask]
        x_all = xx[mask]
        A_all = np.column_stack([x_all, y_all, np.ones_like(x_all)])
        
        result = np.full_like(data, np.nan)
        result[mask] = data[mask] - (A_all @ coeff)
        return result
    except np.linalg.LinAlgError:
        return np.full_like(data, np.nan)


def compute_3sigma_range(data, mask):
    """Compute color scale range as mean ± 3 sigma, ignoring NaN and outliers."""
    effective_data = data[mask & ~np.isnan(data)]
    if len(effective_data) == 0:
        return None, None
    mu = np.mean(effective_data)
    sigma = np.std(effective_data)
    return mu - 3 * sigma, mu + 3 * sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    args = parser.parse_args()
    
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path}")
        sys.exit(1)

    config = ScenarioConfig.from_yaml(scenario_path)
    tile_shape = config.tile_shape or (128, 128)
    ref_bias_coeffs = config.metadata.get("reference_bias_coefficients")
    
    radius_frac = None
    if config.metadata.get("detector_pupil") == "circular":
        radius_frac = float(config.metadata.get("detector_radius_fraction", 0.45))
        
    ref_bias_field = generate_reference_bias_field(
        tile_shape, 
        ref_bias_coeffs,
        radius_fraction=radius_frac
    )
    
    baselines = {
        "GLS Standard": "src/stitching/editable/gls/baseline.py",
        "GLS Robust (Huber)": "src/stitching/editable/gls_robust/baseline.py",
        "SCS (Calibration)": "src/stitching/editable/scs/baseline.py",
        "SIAC (Alternating)": "src/stitching/editable/siac/baseline.py",
        "PSO (Stochastic)": "src/stitching/editable/pso/baseline.py",
    }

    n_algos = len(baselines)
    fig, axes = plt.subplots(n_algos + 1, 4, figsize=(24, 5 * (n_algos + 1)))
    
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
        
        if not gt_done:
            gt_rms = np.sqrt(np.nanmean(z_truth_plot**2)) if np.any(~np.isnan(z_truth_plot)) else 0.0
            vmin, vmax = compute_3sigma_range(z_truth_plot, mask)
            im0 = axes[0, 0].imshow(z_truth_plot, origin='lower', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f"Ground Truth (Detrended)\nRMS: {gt_rms:.6f}\n{scenario_path.stem}")
            fig.colorbar(im0, ax=axes[0, 0])
            
            vmin_ref, vmax_ref = compute_3sigma_range(ref_bias_field, ~np.isnan(ref_bias_field))
            im_ref = axes[0, 1].imshow(ref_bias_field, origin='lower', cmap='RdBu_r', vmin=vmin_ref, vmax=vmax_ref)
            axes[0, 1].set_title("Reference Bias\n(Instrument Calibration - GT)")
            fig.colorbar(im_ref, ax=axes[0, 1])
            
            axes[0, 2].axis('off')
            axes[0, 3].axis('off')
            gt_done = True
        
        recon_rms = np.sqrt(np.nanmean(z_recon_plot**2)) if np.any(~np.isnan(z_recon_plot)) else 0.0
        vmin_r, vmax_r = compute_3sigma_range(z_recon_plot, mask)
        im1 = axes[row_idx, 0].imshow(z_recon_plot, origin='lower', vmin=vmin_r, vmax=vmax_r)
        axes[row_idx, 0].set_title(f"{name}\nReconstruction\nRMS: {recon_rms:.6f}")
        fig.colorbar(im1, ax=axes[row_idx, 0])
        
        # Use nan-aware RMS calculation if needed, or prefer the trusted metric
        rms = report.signal_metrics.get('rms_detrended', float('nan'))
        if np.isnan(rms) and np.any(~np.isnan(z_diff)):
             rms = np.sqrt(np.nanmean(z_diff**2))

        vmin_d, vmax_d = compute_3sigma_range(z_diff, mask)
        im2 = axes[row_idx, 1].imshow(z_diff, origin='lower', cmap='RdBu_r', vmin=vmin_d, vmax=vmax_d)
        axes[row_idx, 1].set_title(f"{name} Error (Diff to GT)\nRMS_detrend: {rms:.6f}")
        fig.colorbar(im2, ax=axes[row_idx, 1])
        
        cal_key = recon.metadata.get('instrument_calibration')
        if cal_key is None:
            cal_key = recon.metadata.get('calibration_map')
        
        if cal_key is not None and isinstance(cal_key, np.ndarray):
            vmin_c, vmax_c = compute_3sigma_range(cal_key, ~np.isnan(cal_key))
            im3 = axes[row_idx, 2].imshow(cal_key, origin='lower', cmap='RdBu_r', vmin=vmin_c, vmax=vmax_c)
            axes[row_idx, 2].set_title(f"{name}\nInstrument Calibration (Estimated)")
            fig.colorbar(im3, ax=axes[row_idx, 2])
            
            # Calibration Error (Estimated - Ground Truth)
            cal_mask = ~np.isnan(cal_key) & ~np.isnan(ref_bias_field)
            if np.any(cal_mask):
                cal_diff_raw = cal_key - ref_bias_field
                cal_diff = detrend(cal_diff_raw, cal_mask)
                cal_rms = np.sqrt(np.nanmean(cal_diff**2))
                vmin_e, vmax_e = compute_3sigma_range(cal_diff, cal_mask)
                im4 = axes[row_idx, 3].imshow(cal_diff, origin='lower', cmap='RdBu_r', vmin=vmin_e, vmax=vmax_e)
                axes[row_idx, 3].set_title(f"{name}\nCalib Error (Est - GT)\nRMS: {cal_rms:.6f}")
                fig.colorbar(im4, ax=axes[row_idx, 3])
            else:
                axes[row_idx, 3].text(0.5, 0.5, "Mask mismatch", ha='center', va='center', transform=axes[row_idx, 3].transAxes)
                axes[row_idx, 3].set_title(f"{name}\nCalib Error")
        else:
            axes[row_idx, 2].text(0.5, 0.5, "N/A\n(no calibration estimated)", 
                                  ha='center', va='center', transform=axes[row_idx, 2].transAxes,
                                  fontsize=12)
            axes[row_idx, 2].set_title(f"{name}\nInstrument Calibration")
            
            axes[row_idx, 3].text(0.5, 0.5, "N/A", ha='center', va='center', transform=axes[row_idx, 3].transAxes, fontsize=12)
            axes[row_idx, 3].set_title(f"{name}\nCalib Error")
    
    output_path = Path("artifacts/comparison_all_algos_s17.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nComparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()
