from __future__ import annotations

from stitching.contracts import ScenarioConfig
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_zero_effect_hooks_preserve_identity_for_multiple_seeds() -> None:
    for seed in (0, 1, 7):
        config = ScenarioConfig(
            scenario_id=f"seed_{seed}",
            description="identity",
            grid_shape=(9, 9),
            pixel_size=1.0,
            scan_offsets=((0.0, 0.0),),
            seed=seed,
        )
        truth, observations = simulate_identity_observations(config)
        observation = observations[0]

        assert (observation.z == truth.z).all()
        assert (observation.valid_mask == truth.valid_mask).all()


def test_shifted_observation_keeps_value_and_mask_alignment_for_multiple_offsets() -> None:
    for offset in ((0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (2.0, 1.0)):
        config = ScenarioConfig(
            scenario_id="shift",
            description="shift",
            grid_shape=(11, 11),
            pixel_size=1.0,
            scan_offsets=(offset,),
            seed=0,
        )
        _, observations = simulate_identity_observations(config)
        observation = observations[0]

        assert observation.z.shape == observation.valid_mask.shape
        assert (observation.z[~observation.valid_mask] == 0.0).all()

