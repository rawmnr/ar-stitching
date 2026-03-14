"""Small harness for running trusted identity evaluation on versioned scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stitching.editable.baseline import baseline_integer_unshift_mean, baseline_integer_unshift_median
from stitching.contracts import EvalReport, ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.trusted.eval.metrics import build_eval_report
from stitching.trusted.scan.transforms import placement_slices
from stitching.trusted.simulator.identity import simulate_identity_observations


def _resolve_baseline(baseline_name: str):
    """Resolve a named baseline implementation."""

    baselines = {
        "mean": baseline_integer_unshift_mean,
        "median": baseline_integer_unshift_median,
    }
    if baseline_name not in baselines:
        raise ValueError(f"Unsupported baseline '{baseline_name}'.")
    return baselines[baseline_name]


def _expected_observed_support(observations: tuple[SubApertureObservation, ...]) -> np.ndarray:
    """Compute the physical observed support implied by the provided detector tiles."""

    if not observations:
        raise ValueError("At least one observation is required to derive observed support.")
    support = np.zeros(observations[0].global_shape, dtype=bool)
    for observation in observations:
        global_y, global_x, local_y, local_x = placement_slices(
            observation.global_shape,
            observation.tile_shape,
            observation.center_xy,
        )
        local_mask = np.asarray(observation.valid_mask, dtype=bool)[local_y, local_x]
        support_view = support[global_y, global_x]
        support_view[local_mask] = True
    return support


def _with_verified_observed_support(
    reconstruction: ReconstructionSurface,
    observations: tuple[SubApertureObservation, ...],
) -> ReconstructionSurface:
    """Require the algorithm-declared observed support to match the physical tile union."""

    expected_support = _expected_observed_support(observations)
    if reconstruction.observed_support_mask is None:
        raise ValueError("ReconstructionSurface must provide observed_support_mask.")
    if reconstruction.observed_support_mask.shape != expected_support.shape:
        raise ValueError("Reconstruction observed_support_mask shape must match physical observed support.")
    if not np.array_equal(reconstruction.observed_support_mask, expected_support):
        raise ValueError("Reconstruction observed_support_mask must equal the union of physical observation support.")
    return ReconstructionSurface(
        z=reconstruction.z,
        valid_mask=reconstruction.valid_mask,
        source_observation_ids=reconstruction.source_observation_ids,
        observed_support_mask=expected_support,
        metadata=dict(reconstruction.metadata),
    )


def run_baseline_eval(scenario_path: str | Path, baseline_name: str | None = None) -> EvalReport:
    """Load a scenario, reconstruct with a selected baseline, and evaluate that reconstruction."""

    config = ScenarioConfig.from_yaml(scenario_path)
    truth, observations = simulate_identity_observations(config)
    selected_baseline = config.baseline_name if baseline_name is None else baseline_name
    reconstruction = _with_verified_observed_support(_resolve_baseline(selected_baseline)(observations), observations)
    return build_eval_report(config, truth, reconstruction, runtime_sec=0.0)


def run_median_baseline_eval(scenario_path: str | Path) -> EvalReport:
    """Convenience wrapper for the experimental median baseline."""

    return run_baseline_eval(scenario_path, baseline_name="median")


def run_identity_eval(scenario_path: str | Path) -> EvalReport:
    """Backward-compatible wrapper for the current baseline evaluation path."""

    return run_baseline_eval(scenario_path)
