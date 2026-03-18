import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario
from stitching.trusted.instrument.bias import generate_reference_bias_field
from scipy import ndimage

scenario_path = Path("scenarios/s17_highres_circular.yaml")
config = ScenarioConfig.from_yaml(scenario_path)

tile_shape = config.tile_shape or (128, 128)
radius_frac = float(config.metadata.get("detector_radius_fraction", 0.45))

ref_bias_coeffs = np.zeros(36)
ref_bias_coeffs[30] = 0.5 

ref_bias_field = generate_reference_bias_field(
    tile_shape, 
    ref_bias_coeffs,
    radius_fraction=radius_frac
)

# Simulate observations
from stitching.trusted.simulator.identity import simulate_identity_observations

# Instead of rewriting the simulator, we will just use it normally, but we modify the config first
# Let's add a "scratch" metadata that the simulator will pick up.
# Actually, the simplest way to inject HF without touching the trusted code is to intercept the observations.

truth, observations = simulate_identity_observations(config)

# Inject high-frequency defect into instrument bias for all observations
yy, xx = np.indices(tile_shape)
# Create a sharp cross mark
scratch_mask = (np.abs(yy - 64) < 2) | (np.abs(xx - 64) < 2)
scratch_value = 2.0  # Large enough to see clearly

from stitching.trusted.surface.footprint import circular_pupil_mask
mask = circular_pupil_mask(tile_shape, radius_fraction=radius_frac)
scratch_mask = scratch_mask & mask

for obs in observations:
    # We add the scratch to the measured data
    # Because it's an instrument bias, it's added in the detector frame, which is exactly what obs.z is.
    obs.z[scratch_mask] += scratch_value

candidate = load_candidate_module(Path("src/stitching/editable/siac/baseline.py"))
recon = candidate.reconstruct(observations, config)

cal_est = recon.metadata.get('instrument_calibration')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
true_cal = np.zeros(tile_shape)
true_cal[scratch_mask] = scratch_value
plt.imshow(true_cal, origin='lower')
plt.title("Injected HF Defect (Cross)")
plt.colorbar()

plt.subplot(1, 2, 2)
if cal_est is not None:
    plt.imshow(cal_est, origin='lower', vmin=-1, vmax=3)
    plt.title("Estimated Instrument Calibration")
    plt.colorbar()

plt.savefig("artifacts/hf_capture_test.png")
print("HF test plot saved to artifacts/hf_capture_test.png")
