"""Structured prompt construction for autoresearch iterations."""

from __future__ import annotations

from pathlib import Path

from stitching.harness.protocols import ExperimentContext

# Ultra-concise domain knowledge
DOMAIN_NOTES = """## Task: Reconstruct global surface S(x,y) from overlapping sub-aperture measurements.
- Each W_i = S + R + P_i + ε (reference bias + piston/tip/tilt + noise)
- Data: SubApertureObservation(z, valid_mask, center_xy, tile_shape, global_shape)
- Output: ReconstructionSurface(z, valid_mask, observed_support_mask)

## What Works
- Track overlap_count per pixel, scale overlapping regions (0.99-0.997)
- GLS: solve piston/tip/tilt via scipy.linalg.lstsq on overlaps
- Vectorized NumPy only (no pixel loops)"""


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
    scenario_results: dict[str, dict] | None = None,
) -> ExperimentContext:
    """Assemble a complete ExperimentContext from raw inputs."""
    candidate_source = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else ""

    # Compact per-scenario feedback
    scenario_breakdown = ""
    if scenario_results:
        lines = [f"{'✓' if r.get('accepted') else '✗'} {sid}: RMS={r.get('rms', 0):.6f}"
                 for sid, r in scenario_results.items()]
        scenario_breakdown = " | ".join(lines)
    
    # Context summary
    curr_rms = current_metrics.get('aggregate_rms', float('inf'))
    best_rms = best_metrics.get('aggregate_rms', float('inf'))
    
    # Failure feedback with curriculum hints based on iteration
    failure_hint = ""
    if previous_summary and "REJECTED" in previous_summary:
        if "np.math.factorial" in previous_summary:
            failure_hint = " | HINT: Use math.factorial (not np.math.factorial)"
        elif "dimension" in previous_summary.lower() or "shape" in previous_summary.lower():
            failure_hint = " | HINT: Use config.grid_shape for matrix dimensions"
        elif "crash" in previous_summary.lower():
            failure_hint = " | HINT: Keep code simple, avoid complex numpy operations"
    
    # Curriculum hint based on iteration (progressive complexity)
    if iteration < 3:
        curriculum_hint = "START SIMPLE: Just add overlap_count tracking + small scaling factor (0.997) to baseline"
    elif iteration < 6:
        curriculum_hint = "NEXT: Try computing overlap-based mean instead of global mean"
    elif iteration < 10:
        curriculum_hint = "ADVANCED: Try GLS for piston/tip/tilt estimation on overlaps"
    else:
        curriculum_hint = "EXPERT: Try Huber loss or Tikhonov regularization"

    extended_notes = f"""{DOMAIN_NOTES}

## State: RMS={curr_rms:.6f} (best={best_rms:.6f}) | {scenario_breakdown}{failure_hint}

## Strategy (iter {iteration}): {curriculum_hint}

## Code Pattern (working baseline):
```python
for obs in observations:
    cx, cy = obs.center_xy
    top = int(round(cy - (obs.tile_shape[0] - 1) / 2.0))
    left = int(round(cx - (obs.tile_shape[1] - 1) / 2.0))
    gy_s, gy_e = max(0, top), min(global_shape[0], top + obs.tile_shape[0])
    gx_s, gx_e = max(0, left), min(global_shape[1], left + obs.tile_shape[1])
    if gy_e > gy_s and gx_e > gx_s:
        local = obs.z[max(0,-top):, max(0,-left):]
        sum_z[gy_s:gy_e, gx_s:gx_e][obs.valid_mask] += local[obs.valid_mask]
        count[gy_s:gy_e, gx_s:gx_e][obs.valid_mask] += 1
z = sum_z / np.maximum(count, 1)
```
## Rules
- Use numpy/scipy only (no loops on pixels)
- Return ReconstructionSurface with observed_support_mask
- Keep class CandidateStitcher"""

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
        domain_notes=extended_notes,
    )
