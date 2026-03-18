"""Structured prompt construction for autoresearch iterations."""

from __future__ import annotations

from pathlib import Path

from stitching.harness.protocols import ExperimentContext


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
        items = []
        for sid, r in scenario_results.items():
            line = f"{'✓' if r.get('accepted') else '✗'} {sid}: RawRMS={r.get('rms', 0):.4f}"
            if r.get('rms_detrended') is not None:
                line += f" | DetrendRMS={r.get('rms_detrended'):.4f}"
            if r.get('tilt_piston_rms') is not None:
                line += f" | TiltMag={r.get('tilt_piston_rms'):.4f}"
            items.append(line)
        scenario_breakdown = "\n".join(items)
    
    # Context summary
    curr_rms = current_metrics.get('aggregate_rms', float('inf'))
    best_rms = best_metrics.get('aggregate_rms', float('inf'))

    # Ultra-simple strategy - ONE tiny change at a time
    strategy = ""
    if iteration == 0:
        strategy = "TINY CHANGE 1: Add z = z * 0.997 AFTER the line 'z = sum_z / np.maximum(count, 1)'. Do NOT change anything else."
    elif iteration < 5:
        strategy = f"Refine the global scale or implement a tiny per-tile piston correction."
    else:
        strategy = "Implement a tiny global tilt correction using scipy.linalg.lstsq if TiltMag is high."

    # Error hints
    error_hint = ""
    if previous_summary:
        if "REJECTED_CRASH" in previous_summary:
            error_hint = "\n⚠️ PREVIOUS CRASHED - check your matrix shapes!"
        elif "np.math" in previous_summary:
            error_hint = "\n⚠️ Use math.factorial NOT np.math.factorial"
        elif "copy=False" in previous_summary:
            error_hint = "\n⚠️ Remove all copy=False from np.array()"

    # Ultra-concise prompt - ~150 tokens
    extended_notes = f"""# TASK: Reduce RMS on {scenario_ids or 'scenarios'}

## Current: Aggregate RMS={curr_rms:.6f} | Best={best_rms:.6f}
{scenario_breakdown}{error_hint}

## STRATEGY (iter {iteration}):
{strategy}

## TEMPLATE (copy this pattern):
```python
import numpy as np
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
# import scipy.linalg # allowed for tilt/piston estimation

class CandidateStitcher:
    def reconstruct(self, observations, config):
        obs_list = list(observations)
        shape = obs_list[0].global_shape
        sum_z = np.zeros(shape)
        count = np.zeros(shape, dtype=int)
        support = np.zeros(shape, dtype=bool)
        
        for obs in obs_list:
            cx, cy = obs.center_xy
            r, c = obs.tile_shape
            top, left = int(round(cy - (r-1)/2)), int(round(cx - (c-1)/2))
            gs = max(0, top); ge = min(shape[0], top+r)
            ls = max(0, -top); le = ls + (ge-gs)
            if ge > gs:
                sum_z[gs:ge, max(0,left):min(shape[1],left+c)][obs.valid_mask] += obs.z[obs.valid_mask]
                count[gs:ge, max(0,left):min(shape[1],left+c)][obs.valid_mask] += 1
                support[gs:ge, max(0,left):min(shape[1],left+c)][obs.valid_mask] = True
        
        z = sum_z / np.maximum(count, 1)
        
        # === PASTE YOUR 1-LINE CHANGE HERE ===
        
        return ReconstructionSurface(z=z, valid_mask=count>0, source_observation_ids=tuple(o.observation_id for o in obs_list), observed_support_mask=support)
```

## RULES:
- Use scipy.linalg.lstsq ONLY if needed for tilt/piston
- NO copy=False in np.array()
- One small change at a time
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
            "src/stitching/harness**",
            "scenarios/**",
        ),
        scenario_ids=scenario_ids,
        time_budget_sec=time_budget_sec,
        domain_notes=extended_notes,
    )
