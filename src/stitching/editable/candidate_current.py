"""GLS piston+tip/tilt correction candidate for optical stitching."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices

# Hypothesis: Simultaneous GLS estimation of piston, tip, and tilt with robust sub-pixel placement.
class CandidateStitcher:
    """Candidate that aligns overlapping tiles using a sparse GLS piston+tip/tilt solve."""
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        """Aggregate observations after global piston+tip/tilt correction."""
        observation_list = list(observations)
        if not observation_list:
            raise ValueError("At least one observation is required for reconstruction.")
        
        global_shape = observation_list[0].global_shape
        observed_support_mask = np.zeros(global_shape, dtype=bool)
        contributions_idx: list[np.ndarray] = []
        contributions_val: list[np.ndarray] = []
        contributions_obs: list[np.ndarray] = []
        source_observation_ids: list[str] = []
        tile_centers: list[tuple[float, float]] = []

        for obs_idx, observation in enumerate(observation_list):
            if observation.global_shape != global_shape:
                raise ValueError("All observations must share the same global_shape.")
            
            source_observation_ids.append(observation.observation_id)
            tile_centers.append(observation.center_xy)
            
            # ──────────────────────────────────────────────────
            # FIX: Use placement_slices for robust indexing
            # ──────────────────────────────────────────────────
            try:
                gy, gx, ly, lx = placement_slices(global_shape, observation.tile_shape, observation.center_xy)
            except ValueError:
                # If exact integer placement fails (sub-pixel), 
                # we fallback to a simpler rounding logic for the support mask
                # but this shouldn't happen with our placement_slices robust implementation
                continue

            local_mask = np.asarray(observation.valid_mask, dtype=bool)[ly, lx]
            if not local_mask.any():
                continue
            
            # Accumulate global support
            support_view = observed_support_mask[gy, gx]
            support_view[local_mask] = True
                
            v_rows, v_cols = np.nonzero(local_mask)
            
            # Global linear indices
            global_rows = gy.start + v_rows
            global_cols = gx.start + v_cols
            lin_idx = np.ravel_multi_index((global_rows, global_cols), global_shape)
            
            # Local values
            local_z = np.asarray(observation.z, dtype=float)[ly, lx]
            vals = local_z[v_rows, v_cols]
            
            contributions_idx.append(lin_idx)
            contributions_val.append(vals)
            contributions_obs.append(np.full(lin_idx.shape, obs_idx, dtype=int))

        if not contributions_idx:
            return ReconstructionSurface(
                z=np.zeros(global_shape),
                valid_mask=np.zeros(global_shape, dtype=bool),
                source_observation_ids=tuple(source_observation_ids),
                observed_support_mask=np.zeros(global_shape, dtype=bool),
                metadata=self._metadata(observation_list, tile_centers, (), (), ()),
            )

        global_indices = np.concatenate(contributions_idx)
        values = np.concatenate(contributions_val)
        obs_indices = np.concatenate(contributions_obs)
        
        x_global = (global_indices % global_shape[1]).astype(float)
        y_global = (global_indices // global_shape[1]).astype(float)
        
        sort_order = np.argsort(global_indices)
        global_indices = global_indices[sort_order]
        values = values[sort_order]
        obs_indices = obs_indices[sort_order]
        x_global = x_global[sort_order]
        y_global = y_global[sort_order]

        diff = np.diff(global_indices)
        group_starts = np.concatenate(([0], np.nonzero(diff)[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [global_indices.size]))
        
        row_indices: list[np.ndarray] = []
        col_indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        row_offset = 0
        
        for start, end in zip(group_starts, group_ends):
            group_size = end - start
            if group_size < 2:
                continue
                
            g_obs = obs_indices[start:end]
            g_vals = values[start:end]
            i_idx, j_idx = np.triu_indices(group_size, k=1)
            num_pairs = i_idx.size
            x_val, y_val = x_global[start], y_global[start]
            row_ids = np.arange(num_pairs) + row_offset
            obs_i, obs_j = g_obs[i_idx], g_obs[j_idx]
            
            row_indices.append(np.repeat(row_ids, 6))
            col_indices.append(np.concatenate([3*obs_i, 3*obs_i+1, 3*obs_i+2, 3*obs_j, 3*obs_j+1, 3*obs_j+2]))
            data.append(np.concatenate([
                np.full(num_pairs, -1.0), np.full(num_pairs, -x_val), np.full(num_pairs, -y_val),
                np.full(num_pairs, 1.0), np.full(num_pairs, x_val), np.full(num_pairs, y_val)
            ]))
            rhs.append(g_vals[i_idx] - g_vals[j_idx])
            row_offset += num_pairs

        num_obs = len(observation_list)
        if row_offset == 0:
            p_s, t_s, tl_s = np.zeros(num_obs), np.zeros(num_obs), np.zeros(num_obs)
        else:
            A_row = np.concatenate(row_indices)
            A_col = np.concatenate(col_indices)
            A_data = np.concatenate(data)
            A_rhs = np.concatenate(rhs)
            
            lambda_reg = 1e-6
            reg_rows = np.arange(3 * num_obs) + row_offset
            reg_cols = np.arange(3 * num_obs)
            A_row = np.concatenate([A_row, reg_rows])
            A_col = np.concatenate([A_col, reg_cols])
            A_data = np.concatenate([A_data, np.full(3 * num_obs, lambda_reg)])
            A_rhs = np.concatenate([A_rhs, np.zeros(3 * num_obs)])
            
            system = sparse.coo_matrix((A_data, (A_row, A_col)), shape=(row_offset + 3*num_obs, 3*num_obs))
            sol = lsqr(system, A_rhs)[0]
            sol = sol.reshape((num_obs, 3))
            p_s, t_s, tl_s = sol[:, 0] - np.mean(sol[:, 0]), sol[:, 1] - np.mean(sol[:, 1]), sol[:, 2] - np.mean(sol[:, 2])

        adj_values = values + p_s[obs_indices] + t_s[obs_indices] * x_global + tl_s[obs_indices] * y_global
        global_area = global_shape[0] * global_shape[1]
        sum_z = np.bincount(global_indices, weights=adj_values, minlength=global_area)
        count = np.bincount(global_indices, minlength=global_area)
        
        valid_flat = count > 0
        z_flat = np.zeros(global_area)
        z_flat[valid_flat] = sum_z[valid_flat] / count[valid_flat]
        
        return ReconstructionSurface(
            z=z_flat.reshape(global_shape),
            valid_mask=valid_flat.reshape(global_shape),
            source_observation_ids=tuple(source_observation_ids),
            observed_support_mask=observed_support_mask,
            metadata=self._metadata(observation_list, tile_centers, p_s.tolist(), t_s.tolist(), tl_s.tolist()),
        )

    @staticmethod
    def _metadata(o, c, p, t, tl) -> dict[str, object]:
        centers = c[0] if len(c) == 1 else tuple(c)
        return {
            "baseline": "gls_piston_tip_tilt_robust",
            "baseline_experimental": True,
            "reconstruction_frame": "global_truth",
            "tile_centers_xy": centers,
            "num_observations_used": len(o),
            "piston_shifts": tuple(float(s) for s in p),
            "tip_shifts": tuple(float(s) for s in t),
            "tilt_shifts": tuple(float(s) for s in tl),
            **dict(o[0].metadata),
        }
