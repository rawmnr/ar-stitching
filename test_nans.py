import numpy as np
from stitching.contracts import ScenarioConfig
from stitching.trusted.simulator.identity import simulate_identity_observations

config = ScenarioConfig.from_yaml("scenarios/s17_highres_circular.yaml")
truth, observations = simulate_identity_observations(config)

for i, obs in enumerate(observations):
    nans_in_mask = np.sum(np.isnan(obs.z) & obs.valid_mask)
    if nans_in_mask > 0:
        print(f"Observation {i} has {nans_in_mask} NaNs inside valid_mask!")
        break
else:
    print("No NaNs found inside valid_mask for any observation.")
