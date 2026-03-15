"""Structured prompt construction for autoresearch iterations."""

from __future__ import annotations

from pathlib import Path

from stitching.harness.protocols import ExperimentContext

# Domain knowledge injected into every prompt
DOMAIN_NOTES = """\
## Optical Stitching Domain Notes

### Problem
Reconstruct a global surface S(x,y) from overlapping sub-aperture interferometric measurements.
Each measurement W_i = S + R + P_i + epsilon_i where:
- R: stationary reference bias (detector frame)
- P_i: rigid-body positioning error (piston, tip, tilt)
- epsilon_i: noise + outliers

### Key Algorithms
- GLS (Global Least Squares): simultaneous solve over all overlaps
- CS (Calibrate then Stitch): estimate R first, then stitch
- SC (Stitch then Calibrate): stitch first, then extract R from residuals
- Huber M-estimator: robust to outliers via IRLS
- Tikhonov regularization: stabilize ill-conditioned GLS

### Performance Drivers
- Vectorized sparse matrix operations (scipy.sparse)
- Overlap exploitation for redundancy
- Robust weighting to reject outliers
- Proper nuisance term estimation and removal

### Constraints
- Must return ReconstructionSurface with correct observed_support_mask
- Must not mask pixels to game the score
- Must not smooth aggressively (hf_retention monitored)
- Runtime < 300s per scenario
"""


def build_experiment_context(
    experiment_id: str,
    iteration: int,
    current_metrics: dict[str, float],
    best_metrics: dict[str, float],
    candidate_path: Path,
    previous_diff: str | None = None,
    previous_summary: str | None = None,
    scenario_ids: tuple[str, ...] = (),
    time_budget_sec: float = 300.0,
) -> ExperimentContext:
    """Assemble a complete ExperimentContext from raw inputs."""
    candidate_source = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else ""

    return ExperimentContext(
        experiment_id=experiment_id,
        iteration=iteration,
        current_metrics=current_metrics,
        best_metrics=best_metrics,
        previous_diff=previous_diff,
        previous_summary=previous_summary,
        candidate_source=candidate_source,
        editable_paths=(
            "src/stitching/editable/candidate_current.py",
        ),
        forbidden_paths=(
            "src/stitching/trusted/**",
            "src/stitching/harness/**",
            "scenarios/**",
        ),
        scenario_ids=scenario_ids,
        time_budget_sec=time_budget_sec,
        domain_notes=DOMAIN_NOTES,
    )
