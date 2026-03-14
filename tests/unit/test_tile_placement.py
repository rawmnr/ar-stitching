import numpy as np
import pytest

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.editable.baseline import baseline_integer_unshift_mean
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.scan.transforms import placement_slices
from stitching.trusted.simulator.identity import simulate_identity_observations


def test_even_parity_centering_places_tile_symmetrically() -> None:
    global_y, global_x, local_y, local_x = placement_slices(
        global_shape=(8, 8),
        tile_shape=(4, 4),
        center_xy=(3.5, 3.5),
    )

    assert (global_y.start, global_y.stop) == (2, 6)
    assert (global_x.start, global_x.stop) == (2, 6)
    assert (local_y.start, local_y.stop) == (0, 4)
    assert (local_x.start, local_x.stop) == (0, 4)


def test_even_tile_identity_reconstructs_without_half_pixel_bias() -> None:
    config = ScenarioConfig(
        scenario_id="even_identity",
        description="even tile",
        grid_shape=(8, 8),
        tile_shape=(4, 4),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
    )
    truth, observations = simulate_identity_observations(config)
    reconstruction = baseline_integer_unshift_mean(observations)
    report = build_eval_report(config, truth, reconstruction, runtime_sec=0.0)

    assert report.accepted is False
    assert report.signal_metrics["mae_on_valid_intersection"] == 0.0
    assert report.geometry_metrics["footprint_iou"] > 0.0


def test_out_of_bounds_tile_at_origin_is_clipped_consistently() -> None:
    config = ScenarioConfig(
        scenario_id="origin_tile",
        description="origin tile",
        grid_shape=(20, 20),
        tile_shape=(6, 6),
        pixel_size=1.0,
        scan_offsets=((-9.5, -9.5),),
        seed=0,
    )
    _, observations = simulate_identity_observations(config)
    observation = observations[0]

    assert observation.center_xy == (0.0, 0.0)
    assert observation.valid_mask.sum() > 0
    assert observation.valid_mask.sum() < observation.valid_mask.size


def test_baseline_rejects_conflicting_global_shapes() -> None:
    first = SubApertureObservation(
        observation_id="a",
        z=np.ones((3, 3), dtype=float),
        valid_mask=np.ones((3, 3), dtype=bool),
        tile_shape=(3, 3),
        center_xy=(2.0, 2.0),
        global_shape=(5, 5),
        rotation_deg=0.0,
    )
    second = SubApertureObservation(
        observation_id="b",
        z=np.ones((3, 3), dtype=float),
        valid_mask=np.ones((3, 3), dtype=bool),
        tile_shape=(3, 3),
        center_xy=(2.0, 2.0),
        global_shape=(6, 6),
        rotation_deg=0.0,
    )

    with pytest.raises(ValueError):
        baseline_integer_unshift_mean((first, second))


def test_evaluator_rejects_hallucinated_support() -> None:
    config = ScenarioConfig(
        scenario_id="hallucinated",
        description="hallucinated support",
        grid_shape=(5, 5),
        pixel_size=1.0,
        scan_offsets=((0.0, 0.0),),
        seed=0,
    )
    truth, _ = simulate_identity_observations(config)
    observed_support = np.zeros((5, 5), dtype=bool)
    observed_support[1:4, 1:4] = True
    candidate_mask = np.ones((5, 5), dtype=bool)
    candidate = ReconstructionSurface(
        z=np.array(truth.z, copy=True),
        valid_mask=candidate_mask,
        source_observation_ids=("obs",),
        observed_support_mask=observed_support,
    )

    report = build_eval_report(config, truth, candidate, runtime_sec=0.0)

    assert report.accepted is False
    assert "reconstruction_valid_mask_exceeds_observed_support" in report.notes
