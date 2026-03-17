
import argparse
import sys
from pathlib import Path
import numpy as np
from stitching.harness.evaluator import evaluate_candidate_on_suite, load_candidate_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default="src/stitching/editable/candidate_current.py")
    parser.add_argument("--scenario", type=str, default="scenarios/s17_highres_circular.yaml")
    args = parser.parse_args()
    
    candidate_path = Path(args.candidate)
    scenario_path = Path(args.scenario)
    
    if not candidate_path.exists():
        print(f"Candidate file not found: {candidate_path}")
        sys.exit(1)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path}")
        sys.exit(1)
        
    candidate = load_candidate_module(candidate_path)
    metrics, reports = evaluate_candidate_on_suite(candidate, [scenario_path])
    
    if not reports:
        print("No reports generated.")
        sys.exit(1)
        
    print(f"Scenario: {reports[0].scenario_id}")
    print(f"Aggregate RMS: {metrics.get('aggregate_rms'):.8f}")
    print(f"RMS on valid intersection: {reports[0].signal_metrics.get('rms_on_valid_intersection', float('nan')):.8f}")
    print(f"Accepted: {reports[0].accepted}")
    
    valid_intersection = reports[0].truth.valid_mask & reports[0].reconstruction.valid_mask
    ref_vals = reports[0].truth.z[valid_intersection]
    cand_vals = reports[0].reconstruction.z[valid_intersection]
    
    # Remove Piston
    ref_vals -= np.mean(ref_vals)
    cand_vals -= np.mean(cand_vals)
    
    # Remove Tilt (Simplified)
    yy, xx = np.indices(reports[0].truth.z.shape, dtype=float)
    y_vals = yy[valid_intersection]
    x_vals = xx[valid_intersection]
    
    A = np.column_stack([x_vals, y_vals, np.ones_like(x_vals)])
    coeff_ref, _, _, _ = np.linalg.lstsq(A, ref_vals, rcond=None)
    coeff_cand, _, _, _ = np.linalg.lstsq(A, cand_vals, rcond=None)
    
    ref_detrended = ref_vals - A @ coeff_ref
    cand_detrended = cand_vals - A @ coeff_cand
    
    rms_detrended = np.sqrt(np.mean((ref_detrended - cand_detrended)**2))
    print(f"RMS (Piston+Tilt removed): {rms_detrended:.8f}")

if __name__ == "__main__":
    main()
