"""Multi-scenario evaluator for autoresearch robustness testing."""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScenarioResult:
    """Result for a single scenario."""

    scenario_id: str
    rms_detrended: float = float("nan")
    rms_on_valid_intersection: float = float("nan")
    mae_detrended: float = float("nan")
    hf_retention: float = float("nan")
    accepted: bool = False
    runtime_sec: float = 0.0
    error_type: str = ""
    error_msg: str = ""
    zernike_residual_36: float = float("nan")
    num_pixels_valid: int = 0
    overlap_fraction: float = 0.0
    grid_shape: tuple[int, int] = (0, 0)
    tile_shape: tuple[int, int] = (0, 0)
    truth_pupil: str = ""
    detector_pupil: str = ""


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all scenarios."""

    aggregate_rms: float = float("nan")
    aggregate_mae: float = float("nan")
    max_rms: float = float("nan")
    min_rms: float = float("nan")
    std_rms: float = float("nan")
    num_accepted: int = 0
    num_scenarios: int = 0
    accepted_all: bool = False
    total_runtime_sec: float = 0.0
    worst_scenario: str = ""
    best_scenario: str = ""
    results: list[ScenarioResult] = field(default_factory=list)


def _zernike_residual_rms(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid_mask: np.ndarray,
    radius_fraction: float | None,
    num_terms: int = 36,
) -> float:
    """Fit Zernike modes and return residual RMS."""
    if not np.any(valid_mask):
        return float("nan")

    from stitching.trusted.bases.zernike import generate_zernike_surface

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = candidate - reference
    y = delta[valid_mask]
    finite = np.isfinite(y)
    if not np.any(finite):
        return float("nan")

    y = y[finite]
    fit_mask = np.asarray(valid_mask, dtype=bool)
    fit_mask_indices = np.flatnonzero(fit_mask.ravel())[finite]

    basis_columns: list[np.ndarray] = []
    for term_idx in range(num_terms):
        coeffs = np.zeros(num_terms, dtype=float)
        coeffs[term_idx] = 1.0
        basis = generate_zernike_surface(
            coeffs,
            reference.shape,
            indexing="noll",
            backend="internal",
            radius_fraction=radius_fraction,
            fill_value=np.nan,
        )
        basis_columns.append(np.asarray(basis, dtype=float).ravel()[fit_mask_indices])

    design = np.column_stack(basis_columns)
    finite_rows = np.all(np.isfinite(design), axis=1)
    if not np.any(finite_rows):
        return float("nan")

    design = design[finite_rows]
    y = y[finite_rows]

    coeffs_fit, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coeffs_fit
    return float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src_to_path(repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-scenario evaluator for autoresearch robustness testing.",
    )
    parser.add_argument(
        "--candidate",
        default="src/stitching/editable/optimized_stitching_algo.py",
        help="Candidate Python file to evaluate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "scenarios/s17_highres_circular.yaml",
            "scenarios/s17_lowres_circular.yaml",
            "scenarios/s17_highres_square.yaml",
            "scenarios/s17_highres_low_overlap.yaml",
            "scenarios/s17_highres_high_overlap.yaml",
        ],
        help="Scenario YAML files to evaluate on.",
    )
    parser.add_argument(
        "--scenario-dir",
        default="",
        help="Directory containing scenario YAML files (alternative to --scenarios).",
    )
    parser.add_argument(
        "--budget-sec",
        type=float,
        default=300.0,
        help="Per-scenario evaluation budget in seconds.",
    )
    parser.add_argument(
        "--weight-rms",
        action="store_true",
        help="Use RMS weighting for aggregate (inverse variance).",
    )
    parser.add_argument(
        "--min-accepted-ratio",
        type=float,
        default=1.0,
        help="Minimum fraction of scenarios that must be accepted (0.0-1.0).",
    )
    return parser.parse_args()


def _extract_scenario_metadata(scenario_path: Path) -> dict[str, object]:
    """Extract relevant metadata from scenario YAML without full parsing."""
    import yaml

    try:
        data = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
        return {
            "scenario_id": data.get("scenario_id", scenario_path.stem),
            "overlap_fraction": data.get("overlap_fraction", 0.25),
            "grid_shape": tuple(data.get("grid_shape", [0, 0])),
            "tile_shape": tuple(data.get("tile_shape", [0, 0])),
            "truth_pupil": data.get("truth_pupil", "unknown"),
            "detector_pupil": data.get("detector_pupil", "unknown"),
        }
    except Exception:
        return {
            "scenario_id": scenario_path.stem,
            "overlap_fraction": 0.25,
            "grid_shape": (0, 0),
            "tile_shape": (0, 0),
            "truth_pupil": "unknown",
            "detector_pupil": "unknown",
        }


def _evaluate_single_scenario(
    candidate_module,
    scenario_path: Path,
    budget_sec: float,
) -> ScenarioResult:
    """Evaluate a single scenario and return result."""
    from stitching.harness.evaluator import evaluate_candidate_on_suite

    result = ScenarioResult(scenario_id=scenario_path.stem)
    meta = _extract_scenario_metadata(scenario_path)
    result.overlap_fraction = float(meta["overlap_fraction"])
    result.grid_shape = tuple(meta["grid_shape"])
    result.tile_shape = tuple(meta["tile_shape"])
    result.truth_pupil = str(meta["truth_pupil"])
    result.detector_pupil = str(meta["detector_pupil"])

    try:
        import time

        t0 = time.time()
        _, reports = evaluate_candidate_on_suite(
            candidate_module,
            [scenario_path],
            eval_budget_sec=budget_sec,
        )
        result.runtime_sec = time.time() - t0

        if not reports:
            result.error_type = "NO_REPORT"
            result.error_msg = "Evaluator returned no reports"
            return result

        report = reports[0]
        result.scenario_id = report.scenario_id
        sig = report.signal_metrics
        result.rms_detrended = sig.get("rms_detrended", float("nan"))
        result.rms_on_valid_intersection = sig.get("rms_on_valid_intersection", float("nan"))
        result.mae_detrended = sig.get("mae_detrended", float("nan"))
        result.hf_retention = sig.get("hf_retention", float("nan"))
        result.accepted = report.accepted

        truth = report.truth
        reconstruction = report.reconstruction
        if truth is not None and reconstruction is not None:
            config = report.config
            radius_fraction = None
            if config is not None:
                radius_fraction = config.metadata.get("truth_radius_fraction")
                if radius_fraction is None:
                    radius_fraction = config.metadata.get("detector_radius_fraction")
            result.zernike_residual_36 = _zernike_residual_rms(
                truth.z,
                reconstruction.z,
                truth.valid_mask & reconstruction.valid_mask,
                radius_fraction=float(radius_fraction) if radius_fraction is not None else None,
                num_terms=36,
            )

        if truth is not None:
            result.num_pixels_valid = int(np.sum(truth.valid_mask))

    except Exception as exc:
        result.error_type = type(exc).__name__
        result.error_msg = str(exc)[:200]
        result.runtime_sec = 0.0

    return result


def _compute_aggregate(metrics: list[ScenarioResult], weight_rms: bool) -> AggregateMetrics:
    """Compute aggregate metrics from per-scenario results."""
    agg = AggregateMetrics()
    agg.results = metrics
    agg.num_scenarios = len(metrics)

    accepted_results = [m for m in metrics if m.accepted and np.isfinite(m.rms_detrended)]
    agg.num_accepted = len(accepted_results)

    if not accepted_results:
        agg.aggregate_rms = float("inf")
        agg.total_runtime_sec = float(np.sum([m.runtime_sec for m in metrics]))
        return agg

    rms_values = np.array([m.rms_detrended for m in accepted_results], dtype=float)
    mae_values = np.array([m.mae_detrended for m in accepted_results], dtype=float)

    if weight_rms:
        weights = 1.0 / (rms_values**2 + 1e-12)
        weights = weights / np.sum(weights)
        agg.aggregate_rms = float(np.sum(weights * rms_values))
        agg.aggregate_mae = float(np.sum(weights * mae_values))
    else:
        agg.aggregate_rms = float(np.sqrt(np.mean(rms_values**2)))
        agg.aggregate_mae = float(np.mean(mae_values))

    agg.max_rms = float(np.max(rms_values))
    agg.min_rms = float(np.min(rms_values))
    agg.std_rms = float(np.std(rms_values))
    agg.total_runtime_sec = float(np.sum([m.runtime_sec for m in metrics]))

    worst_idx = int(np.argmax(rms_values))
    best_idx = int(np.argmin(rms_values))
    agg.worst_scenario = accepted_results[worst_idx].scenario_id
    agg.best_scenario = accepted_results[best_idx].scenario_id
    agg.accepted_all = agg.num_accepted == agg.num_scenarios

    return agg


def _print_aggregate(agg: AggregateMetrics) -> None:
    """Print aggregate metrics."""
    print("---")
    print(f"aggregate_rms: {agg.aggregate_rms:.8f}")
    print(f"aggregate_mae: {agg.aggregate_mae:.8f}")
    print(f"max_rms: {agg.max_rms:.8f}")
    print(f"min_rms: {agg.min_rms:.8f}")
    print(f"std_rms: {agg.std_rms:.8f}")
    print(f"total_runtime_sec: {agg.total_runtime_sec:.2f}")
    print(f"num_accepted: {agg.num_accepted}")
    print(f"num_scenarios: {agg.num_scenarios}")
    print(f"accepted_all: {int(agg.accepted_all)}")
    print(f"worst_scenario: {agg.worst_scenario}")
    print(f"best_scenario: {agg.best_scenario}")


def _print_scenario_details(agg: AggregateMetrics) -> None:
    """Print per-scenario details."""
    print("\n=== SCENARIO_DETAILS ===")
    for i, res in enumerate(agg.results):
        status = "ACCEPT" if res.accepted else "REJECT"
        rms = res.rms_detrended if np.isfinite(res.rms_detrended) else -1.0
        print(f"[{i + 1:02d}] {res.scenario_id}")
        print(f"  status: {status}")
        print(f"  rms_detrended: {rms:.8f}")
        print(f"  rms_valid_intersection: {res.rms_on_valid_intersection:.8f}")
        print(f"  mae_detrended: {res.mae_detrended:.8f}")
        print(f"  hf_retention: {res.hf_retention:.8f}")
        print(f"  zernike_residual_36: {res.zernike_residual_36:.8f}")
        print(f"  runtime_sec: {res.runtime_sec:.2f}")
        print(f"  grid: {res.grid_shape[0]}x{res.grid_shape[1]}")
        print(f"  tile: {res.tile_shape[0]}x{res.tile_shape[1]}")
        print(f"  overlap: {res.overlap_fraction:.2f}")
        print(f"  truth_pupil: {res.truth_pupil}")
        print(f"  detector_pupil: {res.detector_pupil}")
        if res.error_type:
            print(f"  error: {res.error_type}: {res.error_msg}")
    print("=== END_SCENARIO_DETAILS ===\n")


def _print_summary_for_agent(agg: AggregateMetrics) -> None:
    """Print agent-friendly summary highlighting limiting factors."""
    print("\n=== AGENT_SUMMARY ===")

    good = [r for r in agg.results if r.accepted and r.rms_detrended < 1.0]
    medium = [r for r in agg.results if r.accepted and 1.0 <= r.rms_detrended < 2.0]
    poor = [r for r in agg.results if r.accepted and r.rms_detrended >= 2.0]
    failed = [r for r in agg.results if not r.accepted]

    print("performance_breakdown:")
    print(f"  good_rms_lt_1nm: {len(good)}/{agg.num_scenarios}")
    print(f"  medium_1_to_2nm: {len(medium)}/{agg.num_scenarios}")
    print(f"  poor_gt_2nm: {len(poor)}/{agg.num_scenarios}")
    print(f"  failed: {len(failed)}/{agg.num_scenarios}")

    print("limiting_factors:")

    by_resolution: dict[str, list[ScenarioResult]] = {}
    for result in agg.results:
        if result.grid_shape[0] > 0:
            key = f"{result.grid_shape[0]}x{result.grid_shape[1]}"
            by_resolution.setdefault(key, []).append(result)

    if len(by_resolution) > 1:
        print("  resolution_sensitivity:")
        for resolution, results in sorted(by_resolution.items()):
            accepted = [r for r in results if r.accepted]
            if accepted:
                avg_rms = np.mean([r.rms_detrended for r in accepted])
                print(f"    {resolution}: avg_rms={avg_rms:.4f}")

    by_pupil: dict[str, list[ScenarioResult]] = {}
    for result in agg.results:
        key = f"{result.truth_pupil}/{result.detector_pupil}"
        by_pupil.setdefault(key, []).append(result)

    if len(by_pupil) > 1:
        print("  pupil_type_sensitivity:")
        for pupil, results in sorted(by_pupil.items()):
            accepted = [r for r in results if r.accepted]
            if accepted:
                avg_rms = np.mean([r.rms_detrended for r in accepted])
                print(f"    {pupil}: avg_rms={avg_rms:.4f}")

    by_overlap: dict[str, list[ScenarioResult]] = {}
    for result in agg.results:
        overlap_bin = f"{result.overlap_fraction:.2f}"
        by_overlap.setdefault(overlap_bin, []).append(result)

    if len(by_overlap) > 1:
        print("  overlap_sensitivity:")
        for overlap, results in sorted(by_overlap.items()):
            accepted = [r for r in results if r.accepted]
            if accepted:
                avg_rms = np.mean([r.rms_detrended for r in accepted])
                print(f"    overlap={overlap}: avg_rms={avg_rms:.4f}")

    if poor or failed:
        print("  worst_cases:")
        problem_cases = sorted(
            [r for r in agg.results if not r.accepted or r.rms_detrended >= 1.5],
            key=lambda item: item.rms_detrended if np.isfinite(item.rms_detrended) else 999.0,
            reverse=True,
        )[:3]
        for result in problem_cases:
            print(
                f"    - {result.scenario_id}: rms={result.rms_detrended:.4f}, "
                f"grid={result.grid_shape[0]}x{result.grid_shape[1]}, "
                f"pupil={result.truth_pupil}"
            )

    print("=== END_AGENT_SUMMARY ===\n")


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    _add_src_to_path(repo_root)

    from stitching.harness.evaluator import load_candidate_module

    scenario_paths: list[Path] = []
    if args.scenario_dir:
        scenario_dir = _resolve_repo_path(repo_root, args.scenario_dir)
        if scenario_dir.is_dir():
            for yaml_file in sorted(scenario_dir.glob("*.yaml")):
                if yaml_file.name != "template_full.yaml":
                    scenario_paths.append(yaml_file)

    if not scenario_paths:
        for raw_path in args.scenarios:
            scenario_paths.append(_resolve_repo_path(repo_root, raw_path))

    for path in scenario_paths:
        if not path.exists():
            print(f"ERROR: Scenario not found: {path}", file=sys.stderr)
            return 1

    try:
        candidate_path = _resolve_repo_path(repo_root, args.candidate)
        if not candidate_path.exists():
            print(f"ERROR: Candidate file not found: {candidate_path}", file=sys.stderr)
            return 1

        candidate = load_candidate_module(candidate_path)

        print(f"Evaluating on {len(scenario_paths)} scenarios...", file=sys.stderr)

        results: list[ScenarioResult] = []
        for i, scenario_path in enumerate(scenario_paths):
            print(f"  [{i + 1}/{len(scenario_paths)}] {scenario_path.name}...", file=sys.stderr, end=" ")
            result = _evaluate_single_scenario(candidate, scenario_path, args.budget_sec)
            results.append(result)
            status = "OK" if result.accepted else "FAIL"
            rms = result.rms_detrended if np.isfinite(result.rms_detrended) else float("nan")
            print(f"[{status}] rms={rms:.4f}", file=sys.stderr)

        agg = _compute_aggregate(results, args.weight_rms)

        _print_aggregate(agg)
        _print_scenario_details(agg)
        _print_summary_for_agent(agg)

        min_accepted = int(args.min_accepted_ratio * agg.num_scenarios)
        if agg.num_accepted < min_accepted:
            return 2

        return 0

    except BaseException as exc:
        print("---", file=sys.stderr)
        print("aggregate_rms: CRASH", file=sys.stderr)
        print(f"error_type: {type(exc).__name__}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
