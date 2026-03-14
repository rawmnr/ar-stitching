from __future__ import annotations

import numpy as np
import pytest

from stitching.contracts import ScenarioConfig
from stitching.trusted.instrument.bias import reference_bias_for_observation
from stitching.trusted.noise.models import add_outliers, apply_retrace_error, outlier_magnitude_scale
from stitching.trusted.simulator.identity import simulate_identity_observations
from stitching.trusted.validation import validate_observation_alignment


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
        assert observation.nuisance_terms["subaperture_dc"] == 0.0


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
        assert observation.z.shape == observation.tile_shape
        assert (observation.z[~observation.valid_mask] == 0.0).all()
        validate_observation_alignment(observation)


def test_boundary_clipping_preserves_alignment_when_pupil_moves_outside_frame() -> None:
    config = ScenarioConfig(
        scenario_id="clipped",
        description="clipped",
        grid_shape=(11, 11),
        pixel_size=1.0,
        scan_offsets=((6.0, 0.0),),
        seed=0,
    )
    truth, observations = simulate_identity_observations(config)
    observation = observations[0]

    assert observation.valid_mask.sum() < truth.valid_mask.sum()
    assert observation.valid_mask.sum() > 0
    assert (observation.z[~observation.valid_mask] == 0.0).all()
    validate_observation_alignment(observation)


def test_validation_helper_rejects_nonzero_values_outside_mask() -> None:
    config = ScenarioConfig(
        scenario_id="bad",
        description="bad",
        grid_shape=(7, 7),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((3.0, 0.0),),
        seed=0,
    )
    _, observations = simulate_identity_observations(config)
    bad_observation = observations[0]
    bad_observation.z[0, 4] = 1.0

    with pytest.raises(ValueError):
        validate_observation_alignment(bad_observation)


def test_local_tile_observation_is_smaller_than_global_grid() -> None:
    config = ScenarioConfig(
        scenario_id="local_tile",
        description="local tile",
        grid_shape=(9, 9),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((2.0, 0.0),),
        seed=0,
    )
    _, observations = simulate_identity_observations(config)
    observation = observations[0]

    assert observation.tile_shape == (5, 5)
    assert observation.z.shape == (5, 5)
    assert observation.global_shape == (9, 9)


def test_explicit_nuisance_terms_are_preserved_without_fast_path_reset() -> None:
    config = ScenarioConfig(
        scenario_id="nuisance",
        description="nuisance",
        grid_shape=(9, 9),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0), (0.0, 0.0)),
        seed=0,
        metadata={"subaperture_dc_step": 0.25},
    )
    truth, observations = simulate_identity_observations(config)

    assert observations[0].nuisance_terms["subaperture_dc"] == 0.0
    assert observations[1].nuisance_terms["subaperture_dc"] == 0.25
    valid = observations[1].valid_mask
    assert np.allclose((observations[1].z - truth.z)[valid], 0.25)


def test_subaperture_dc_matches_mean_valid_difference_to_truth() -> None:
    config = ScenarioConfig(
        scenario_id="dc_match",
        description="dc match",
        grid_shape=(9, 9),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0), (0.0, 0.0)),
        seed=0,
        metadata={"subaperture_dc_values": [0.0, 0.4]},
    )
    truth, observations = simulate_identity_observations(config)

    observed = observations[1]
    valid = observed.valid_mask
    mean_delta = float(np.mean((observed.z - truth.z)[valid]))

    assert np.isclose(mean_delta, observed.nuisance_terms["subaperture_dc"])


def test_quarter_turn_rotation_is_applied_to_observation_pixels() -> None:
    config = ScenarioConfig(
        scenario_id="rotated",
        description="rotated",
        grid_shape=(9, 9),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((2.0, 0.0),),
        rotation_deg=(90.0,),
        seed=0,
    )
    unrotated_config = ScenarioConfig(
        scenario_id="unrotated",
        description="unrotated",
        grid_shape=(9, 9),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((2.0, 0.0),),
        rotation_deg=(0.0,),
        seed=0,
    )
    _, rotated_obs = simulate_identity_observations(config)
    _, unrotated_obs = simulate_identity_observations(unrotated_config)

    assert rotated_obs[0].rotation_deg == 90.0
    assert np.array_equal(rotated_obs[0].z, np.rot90(unrotated_obs[0].z, k=1))
    assert np.array_equal(rotated_obs[0].valid_mask, np.rot90(unrotated_obs[0].valid_mask, k=1))


def test_observation_preserves_rotation_from_config() -> None:
    config = ScenarioConfig(
        scenario_id="rotation_metadata",
        description="rotation metadata",
        grid_shape=(9, 9),
        tile_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0), (0.0, 0.0)),
        rotation_deg=(0.0, 270.0),
        seed=0,
    )
    _, observations = simulate_identity_observations(config)

    assert observations[0].rotation_deg == 0.0
    assert observations[1].rotation_deg == 270.0


def test_retrace_error_is_surface_dependent_not_uniform_piston() -> None:
    z = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)

    distorted = apply_retrace_error(z, magnitude=0.1)
    delta = distorted - z

    assert not np.allclose(delta, delta[0, 0])
    assert len(np.unique(np.round(delta, decimals=8))) > 1


def test_outlier_magnitude_scales_with_signal_amplitude() -> None:
    low = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    high = 100.0 * low

    assert outlier_magnitude_scale(high) > outlier_magnitude_scale(low)


def test_outliers_are_applied_only_on_valid_pixels() -> None:
    z = np.zeros((4, 4), dtype=float)
    z[1:3, 1:3] = 2.0
    valid_mask = np.zeros((4, 4), dtype=bool)
    valid_mask[1:3, 1:3] = True

    corrupted = add_outliers(z, fraction=0.5, magnitude=1.0, seed=0, valid_mask=valid_mask)

    assert np.all(corrupted[~valid_mask] == 0.0)


def test_outliers_dense_mask_sampling_preserves_count_scale() -> None:
    z = np.zeros((16, 16), dtype=float)

    corrupted = add_outliers(z, fraction=0.125, magnitude=1.0, seed=0)

    assert np.count_nonzero(corrupted) == int(round(z.size * 0.125))


def test_reference_bias_can_drift_over_observations() -> None:
    config = ScenarioConfig(
        scenario_id="bias_drift",
        description="bias drift",
        grid_shape=(9, 9),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        reference_bias=0.1,
        seed=0,
        metadata={"reference_bias_drift_step": 0.05},
    )
    truth, observations = simulate_identity_observations(config)

    assert np.allclose([observation.reference_bias for observation in observations], [0.1, 0.15, 0.2])
    valid = observations[2].valid_mask
    assert np.allclose((observations[2].z - truth.z)[valid], 0.2)


def test_reference_bias_sequence_loader_is_indexed_per_observation() -> None:
    values = [reference_bias_for_observation(0.1, index, {"reference_bias_values": [0.0, 0.2, -0.1]}) for index in range(3)]

    assert np.allclose(values, [0.1, 0.3, 0.0])
