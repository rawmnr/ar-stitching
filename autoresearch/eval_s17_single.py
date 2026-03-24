from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _add_src_to_path(repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen single-scenario evaluator for autoresearch.",
    )
    parser.add_argument(
        "--candidate",
        default="src/stitching/editable/optimized_stitching_algo.py",
        help="Candidate Python file to evaluate.",
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/s17_highres_circular.yaml",
        help="Scenario YAML to evaluate on.",
    )
    parser.add_argument(
        "--budget-sec",
        type=float,
        default=300.0,
        help="Per-scenario evaluation budget in seconds.",
    )
    return parser.parse_args()


def _print_success(metrics: dict[str, float], scenario_report) -> None:
    sig = scenario_report.signal_metrics
    scenario_rms = sig.get("rms_detrended", sig["rms_on_valid_intersection"])
    accepted_all = int(
        metrics.get("num_accepted", 0) == metrics.get("num_scenarios", 0),
    )

    print("---")
    print(f"aggregate_rms: {metrics['aggregate_rms']:.8f}")
    print(f"total_runtime_sec: {metrics['total_runtime_sec']:.2f}")
    print(f"num_accepted: {int(metrics['num_accepted'])}")
    print(f"num_scenarios: {int(metrics['num_scenarios'])}")
    print(f"accepted_all: {accepted_all}")
    print(f"scenario_id: {scenario_report.scenario_id}")
    print(f"scenario_rms_detrended: {scenario_rms:.8f}")
    print(f"scenario_hf_retention: {sig.get('hf_retention', float('nan')):.8f}")
    print(f"scenario_mae_detrended: {sig.get('mae_detrended', float('nan')):.8f}")
    print(f"scenario_accepted: {int(scenario_report.accepted)}")


def _print_failure(exc: BaseException) -> None:
    print("---")
    print("aggregate_rms: CRASH")
    print("total_runtime_sec: 0.00")
    print("num_accepted: 0")
    print("num_scenarios: 1")
    print("accepted_all: 0")
    print(f"error_type: {type(exc).__name__}")
    print(f"error: {exc}")
    traceback.print_exc()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    _add_src_to_path(repo_root)

    try:
        from stitching.harness.evaluator import (
            evaluate_candidate_on_suite,
            load_candidate_module,
        )

        candidate_path = _resolve_repo_path(repo_root, args.candidate)
        scenario_path = _resolve_repo_path(repo_root, args.scenario)

        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        candidate = load_candidate_module(candidate_path)
        metrics, reports = evaluate_candidate_on_suite(
            candidate,
            [scenario_path],
            eval_budget_sec=args.budget_sec,
        )
        if not reports:
            raise RuntimeError("Trusted evaluator returned no reports.")

        _print_success(metrics, reports[0])
        accepted_all = metrics.get("num_accepted", 0) == metrics.get("num_scenarios", 0)
        return 0 if accepted_all else 2
    except BaseException as exc:
        _print_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
