
import sys
from pathlib import Path
import numpy as np
from stitching.harness.evaluator import evaluate_candidate_on_suite, load_candidate_module

def main():
    candidate_path = Path("src/stitching/editable/candidate_current.py")
    scenario_path = Path("scenarios/s17_highres_circular.yaml")
    
    candidate = load_candidate_module(candidate_path)
    metrics, reports = evaluate_candidate_on_suite(candidate, [scenario_path])
    
    print(f"Scenario: {reports[0].scenario_id}")
    print(f"Aggregate RMS: {metrics.get('aggregate_rms'):.8f}")
    print(f"RMS on valid intersection: {reports[0].signal_metrics.get('rms_on_valid_intersection'):.8f}")
    print(f"Accepted: {reports[0].accepted}")

if __name__ == "__main__":
    main()
