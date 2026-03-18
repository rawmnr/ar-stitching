import numpy as np
from pathlib import Path
from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario
from stitching.trusted.simulator.identity import simulate_identity_observations

def detrend(data, mask):
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

def debug_gls():
    scenario_path = Path("scenarios/s17_highres_circular.yaml")
    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)
    
    candidate = load_candidate_module(Path("src/stitching/editable/gls/baseline.py"))
    
    print("Running GLS reconstruction...")
    recon = candidate.reconstruct(observations, config)
    
    mask = truth.valid_mask & recon.valid_mask
    print(f"Truth valid_mask sum: {np.sum(truth.valid_mask)}")
    print(f"Recon valid_mask sum: {np.sum(recon.valid_mask)}")
    print(f"Intersection sum: {np.sum(mask)}")
    
    if np.sum(mask) > 0:
        print(f"Recon z NaNs in valid_mask: {np.sum(np.isnan(recon.z) & recon.valid_mask)}")
        z_recon_plot = detrend(recon.z, mask)
        print(f"Detrended Recon z NaNs in mask: {np.sum(np.isnan(z_recon_plot) & mask)}")
        print(f"Detrended Recon min z: {np.nanmin(z_recon_plot)}")
        print(f"Detrended Recon max z: {np.nanmax(z_recon_plot)}")
        
        # Check evaluator report
        from stitching.trusted.eval.metrics import build_eval_report
        report = build_eval_report(config, truth, recon, observations, 0.0)
        print(f"Eval Report RMS (detrended): {report.signal_metrics.get('rms_detrended')}")
        print(f"Eval Report Valid Intersection Penalized: {not np.any(truth.valid_mask & recon.valid_mask)}")

if __name__ == "__main__":
    debug_gls()
