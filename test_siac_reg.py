import sys
from pathlib import Path

# Force flush
print("Starting...", flush=True)

from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario
print("Imports done", flush=True)

scenario_path = Path('scenarios/s17_highres_circular.yaml')
print(f"Loading candidate...", flush=True)
candidate = load_candidate_module(Path('src/stitching/editable/siac_reg/baseline.py'))
print(f"Evaluating...", flush=True)
report = evaluate_candidate_on_scenario(candidate, scenario_path)
print(f"Done!", flush=True)
rms = report.signal_metrics.get("rms_detrended")
print(f"RMS: {rms}", flush=True)
