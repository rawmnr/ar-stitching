import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario

import dataclasses

# Load a scenario and modify it to have pure noise, no ground truth surface, no reference bias
scenario_path = Path("scenarios/s17_highres_circular.yaml")
config_orig = ScenarioConfig.from_yaml(scenario_path)

# Zero out the true surface and instrument bias to strictly trace noise
new_metadata = config_orig.metadata.copy()
new_metadata["truth_coefficients"] = [0.0]
new_metadata["reference_bias_coefficients"] = []
new_metadata["geometric_retrace_error"] = 0.0
new_metadata["low_frequency_noise_std"] = 0.0
new_metadata["mid_spatial_ripple_std"] = 0.0
new_metadata["realized_pose_drift_std"] = 0.0
new_metadata["realized_pose_error_std"] = 0.0
new_metadata["alignment_random_coeff"] = 0.0

sigma_noise = 0.03
config = dataclasses.replace(
    config_orig,
    reference_bias=0.0,
    retrace_error=0.0,
    gaussian_noise_std=sigma_noise,
    metadata=new_metadata
)

print(f"Base single-frame noise: {sigma_noise:.4f}")

# Simulate observations
from stitching.trusted.simulator.identity import simulate_identity_observations
truth, observations = simulate_identity_observations(config)

# Run GLS (No calibration)
print("Running GLS...")
gls_module = load_candidate_module(Path("src/stitching/editable/gls/baseline.py"))
gls_recon = gls_module.reconstruct(observations, config)

# Run SIAC (With calibration)
print("Running SIAC...")
siac_module = load_candidate_module(Path("src/stitching/editable/siac/baseline.py"))
siac_recon = siac_module.reconstruct(observations, config)

mask = gls_recon.valid_mask & siac_recon.valid_mask

gls_noise_final = np.nanstd(gls_recon.z[mask])
siac_noise_final = np.nanstd(siac_recon.z[mask])

calib_map = siac_recon.metadata.get("instrument_calibration")
calib_mask = ~np.isnan(calib_map)
calib_noise = np.nanstd(calib_map[calib_mask])

print(f"GLS final noise  : {gls_noise_final:.4f} (Expected ~ {sigma_noise / np.sqrt(3):.4f})")
print(f"SIAC final noise : {siac_noise_final:.4f}")
print(f"SIAC Calib noise : {calib_noise:.4f}")
print(f"Noise penalty due to calibration estimation: {(siac_noise_final / gls_noise_final - 1.0) * 100:.1f}%")
