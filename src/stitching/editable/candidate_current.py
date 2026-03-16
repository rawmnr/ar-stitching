"""GLS piston+tip/tilt correction candidate for optical stitching."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices

# Hypothesis: Including tip/tilt estimation with GLS will reduce RMS.
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
            
            gy, gx, ly, lx = placement_slices(global_shape, observation.tile_shape, observation.center_xy)
            
            local_mask = np.asarray(observation.valid_mask, dtype=bool)[ly, lx]
            if not local_mask.any():
                continue
                
            valid_rows, valid_cols = np.nonzero(local_mask)
            global_rows = gy.start + valid_rows
            global_cols = gx.start + valid_cols
            
            linear_indices = np.ravel_multi_index((global_rows, global_cols), global_shape)
            local_values = np.asarray(observation.z, dtype=float)[ly, lx][valid_rows, valid_cols]
            
            contributions_idx.append(linear_indices)
            contributions_val.append(local_values)
            contributions_obs.append(np.full(linear_indices.shape, obs_idx, dtype=int))

        if not contributions_idx:
            zero_z = np.zeros(global_shape, dtype=float)
            zero_mask = np.zeros(global_shape, dtype=bool)
            return ReconstructionSurface(
                z=zero_z,
                valid_mask=zero_mask,
                source_observation_ids=tuple(source_observation_ids),
                observed_support_mask=zero_mask,
                metadata=self._metadata(observation_list, tile_centers, (), (), ()),
            )

        global_indices = np.concatenate(contributions_idx)
        values = np.concatenate(contributions_val)
        obs_indices = np.concatenate(contributions_obs)
        
        sort_order = np.argsort(global_indices)
        global_indices = global_indices[sort_order]
        values = values[sort_order]
        obs_indices = obs_indices[sort_order]

        # Build GLS system for piston+tip/tilt
        x_global = (global_indices % global_shape[1]).astype(float)
        y_global = (global_indices // global_shape[1]).astype(float)
        
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
                
            group_obs = obs_indices[start:end]
            group_vals = values[start:end]
            
            # Pairwise differences between all observations covering this pixel
            i_idx, j_idx = np.triu_indices(group_size, k=1)
            obs_i = group_obs[i_idx]
            obs_j = group_obs[j_idx]
            
            x_val = x_global[start] # All pixels in this group have same x,y
            y_val = y_global[start]
            
            num_pairs = i_idx.size
            row_ids = np.arange(num_pairs) + row_offset
            
            # For each pair (i, j): (P_j + Tx_j*x + Ty_j*y) - (P_i + Tx_i*x + Ty_i*y) = W_i - W_j
            # Matrix A: rows are pairs, cols are 3*N (piston, tip, tilt per obs)
            
            # Column indices for obs i and j
            ci_p, ci_tx, ci_ty = 3*obs_i, 3*obs_i+1, 3*obs_i+2
            cj_p, cj_tx, cj_ty = 3*obs_j, 3*obs_j+1, 3*obs_j+2
            
            # Row indices (each pair gets 6 entries in A)
            row_indices.append(np.repeat(row_ids, 6))
            
            # Col indices
            col_indices.append(np.concatenate([ci_p, ci_tx, ci_ty, cj_p, cj_tx, cj_ty]))
            
            # Data: -1, -x, -y, 1, x, y
            data.append(np.concatenate([
                np.full(num_pairs, -1.0), np.full(num_pairs, -x_val), np.full(num_pairs, -y_val),
                np.full(num_pairs, 1.0), np.full(num_pairs, x_val), np.full(num_pairs, y_val)
            ]))
            
            rhs.append(group_vals[i_idx] - group_vals[j_idx])
            row_offset += num_pairs

        num_obs = len(observation_list)
        if row_offset == 0:
            piston_shifts = np.zeros(num_obs)
            tip_shifts = np.zeros(num_obs)
            tilt_shifts = np.zeros(num_obs)
        else:
            A_row = np.concatenate(row_indices)
            A_col = np.concatenate(col_indices)
            A_data = np.concatenate(data)
            A_rhs = np.concatenate(rhs)
            
            # Regularization to handle rank-deficiency (global piston/tip/tilt)
            lambda_reg = 1e-6
            reg_rows = np.arange(3 * num_obs) + row_offset
            reg_cols = np.arange(3 * num_obs)
            reg_data = np.full(3 * num_obs, lambda_reg)
            
            A_row = np.concatenate([A_row, reg_rows])
            A_col = np.concatenate([A_col, reg_cols])
            A_data = np.concatenate([A_data, reg_data])
            A_rhs = np.concatenate([A_rhs, np.zeros(3 * num_obs)])
            
            system = sparse.coo_matrix((A_data, (A_row, A_col)), shape=(row_offset + 3*num_obs, 3*num_obs))
            sol = lsqr(system, A_rhs)[0]
            sol = sol.reshape((num_obs, 3))
            
            piston_shifts = sol[:, 0]
            tip_shifts = sol[:, 1]
            tilt_shifts = sol[:, 2]
            
            # Zero-mean normalization
            piston_shifts -= np.mean(piston_shifts)
            tip_shifts -= np.mean(tip_shifts)
            tilt_shifts -= np.mean(tilt_shifts)

        # Apply corrections
        adjusted_values = values + piston_shifts[obs_indices] + tip_shifts[obs_indices] * x_global + tilt_shifts[obs_indices] * y_global
        
        global_area = global_shape[0] * global_shape[1]
        sum_z = np.bincount(global_indices, weights=adjusted_values, minlength=global_area)
        count = np.bincount(global_indices, minlength=global_area)
        
        valid_flat = count > 0
        z_flat = np.zeros(global_area)
        z_flat[valid_flat] = sum_z[valid_flat] / count[valid_flat]
        
        z = z_flat.reshape(global_shape)
        valid_mask = valid_flat.reshape(global_shape)
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(source_observation_ids),
            observed_support_mask=valid_mask.copy(),
            metadata=self._metadata(
                observation_list, tile_centers,
                piston_shifts.tolist(), tip_shifts.tolist(), tilt_shifts.tolist(),
            ),
        )

    @staticmethod
    def _metadata(
        observations: Sequence[SubApertureObservation],
        tile_centers: list[tuple[float, float]],
        piston_shifts: Sequence[float],
        tip_shifts: Sequence[float],
        tilt_shifts: Sequence[float],
    ) -> dict[str, object]:
        first = observations[0]
        centers = tile_centers[0] if len(tile_centers) == 1 else tuple(tile_centers)
        metadata: dict[str, object] = {
            "baseline": "gls_piston_tip_tilt",
            "baseline_experimental": True,
            "reconstruction_frame": "global_truth",
            "tile_centers_xy": centers,
            "num_observations_used": len(observations),
            "piston_shifts": tuple(float(s) for s in piston_shifts),
            "tip_shifts": tuple(float(s) for s in tip_shifts),
            "tilt_shifts": tuple(float(s) for s in tilt_shifts),
            **dict(first.metadata),
        }
        return metadata
