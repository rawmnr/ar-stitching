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
        lines = [f"{'✓' if r.get('accepted') else '✗'} {sid}: RMS={r.get('rms', 0):.6f}"
                 for sid, r in scenario_results.items()]
        scenario_breakdown = " | ".join(lines)
    
    # Context summary
    curr_rms = current_metrics.get('aggregate_rms', float('inf'))
    best_rms = best_metrics.get('aggregate_rms', float('inf'))

    # Ultra-simple strategy - no curriculum complexity
    strategy = ""
    if iteration == 0:
        strategy = "STEP 1: Add ONLY 'z = z * 0.997' after z computation. Nothing else."
    elif iteration == 1:
        strategy = "STEP 2: Try 'z = z * 0.995' or '0.999'. Find optimal."
    elif iteration == 2:
        strategy = "STEP 3: Track overlap_count, apply different scale to overlapping pixels."
    elif iteration < 6:
        strategy = f"STEP {iteration+1}: Try overlap-weighted mean or local averaging."
    else:
        strategy = "Try any approach that reduces RMS. Keep it simple."

    # Error hints
    error_hint = ""
    if previous_summary:
        if "REJECTED_CRASH" in previous_summary:
            error_hint = "\n⚠️ PREVIOUS CRASHED - keep code minimal!"
        elif "np.math" in previous_summary:
            error_hint = "\n⚠️ Use math.factorial NOT np.math.factorial"
        elif "copy=False" in previous_summary:
            error_hint = "\n⚠️ Remove all copy=False from np.array()"

    # Ultra-concise prompt - ~150 tokens
    extended_notes = f"""# TASK: Reduce RMS on {scenario_ids or 'scenarios'}

## Current: RMS={curr_rms:.6f} | Best={best_rms:.6f}
{scenario_breakdown}{error_hint}

## STRATEGY (iter {iteration}):
{strategy}

## TEMPLATE (copy this pattern):
```python
import numpy as np
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

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
        # ADD YOUR IMPROVEMENT HERE
        
        return ReconstructionSurface(z=z, valid_mask=count>0, source_observation_ids=tuple(o.observation_id for o in obs_list), observed_support_mask=support)
```

## RULES:
- NO scipy, NO lstsq, NO matrix operations
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
