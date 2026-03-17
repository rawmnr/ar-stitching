"""Simple mean-stitcher baseline for optical stitching."""
from __future__ import annotations

import numpy as np
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices

class CandidateStitcher:
    """Standard integer-unshift mean baseline as a class."""
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        observation_list = list(observations)
        global_shape = observation_list[0].global_shape
        
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=int)
        support = np.zeros(global_shape, dtype=bool)
        
        for obs in observation_list:
            cx, cy = obs.center_xy
            rows, cols = obs.tile_shape
            
            # Sub-pixel aware placement logic (same as simulator support calculation)
            top = int(round(cy - (rows - 1) / 2.0))
            left = int(round(cx - (cols - 1) / 2.0))
            bottom = top + rows
            right = left + cols
            
            gy_s, gy_e = max(0, top), min(global_shape[0], bottom)
            gx_s, gx_e = max(0, left), min(global_shape[1], right)
            
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e = ly_s + (gy_e - gy_s)
            lx_e = lx_s + (gx_e - gx_s)
            
            if gy_e > gy_s and gx_e > gx_s:
                local_z = np.array(obs.z, copy=False)[ly_s:ly_e, lx_s:lx_e]
                local_mask = np.array(obs.valid_mask, copy=False)[ly_s:ly_e, lx_s:lx_e]
                
                sum_z[gy_s:gy_e, gx_s:gx_e][local_mask] += local_z[local_mask]
                count[gy_s:gy_e, gx_s:gx_e][local_mask] += 1
                support[gy_s:gy_e, gx_s:gx_e][local_mask] = True
            
        valid_mask = count > 0
        z = np.zeros(global_shape, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observation_list),
            observed_support_mask=support,
            metadata={"baseline": "baseline_mean_stitcher"},
        )
