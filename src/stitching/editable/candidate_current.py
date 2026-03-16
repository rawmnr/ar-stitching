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
        for observation in observation_list:
            if observation.global_shape != global_shape:
                raise ValueError("All observations must share the same global_shape.")
        contributions_idx: list[np.ndarray] = []
        contributions_val: list[np.ndarray] = []
        contributions_obs: list[np.ndarray] = []
        source_observation_ids: list[str] = []
        tile_centers: list[tuple[float, float]] = []
        for obs_idx, observation in enumerate(observation_list):
            source_observation_ids.append(observation.observation_id)
            tile_centers.append(observation.center_xy)
            center_x, center_y = float(observation.center_xy[0]), float(observation.center_xy[1])
            tile_rows, tile_cols = observation.tile_shape
            top = int(round(center_y - (tile_rows - 1) / 2.0))
            left = int(round(center_x - (tile_cols - 1) / 2.0))
            bottom = top + tile_rows
            right = left + tile_cols
            gy_start, gy_end = max(0, top), min(global_shape[0], bottom)
            gx_start, gx_end = max(0, left), min(global_shape[1], right)
            ly_start, lx_start = max(0, -top), max(0, -left)
            ly_end = ly_start + (gy_end - gy_start)
            lx_end = lx_start + (gx_end - gx_start)
            if gy_end <= gy_start or gx_end <= gx_start:
                continue
            local_mask = np.asarray(observation.valid_mask, dtype=bool)[ly_start:ly_end, lx_start:lx_end]
            if not local_mask.any():
                continue
            valid_rows, valid_cols = np.nonzero(local_mask)
            global_rows = gy_start + valid_rows
            global_cols = gx_start + valid_cols
            linear_indices = np.ravel_multi_index((global_rows, global_cols), global_shape)
            local_values = np.asarray(observation.z, dtype=float)[ly_start:ly_end, lx_start:lx_end][valid_rows, valid_cols]
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
        if global_indices.size == 0:
            zero_z = np.zeros(global_shape, dtype=float)
            zero_mask = np.zeros(global_shape, dtype=bool)
            return ReconstructionSurface(
                z=zero_z,
                valid_mask=zero_mask,
                source_observation_ids=tuple(source_observation_ids),
                observed_support_mask=zero_mask,
                metadata=self._metadata(observation_list, tile_centers, (), (), ()),
            )
        # Build GLS system for piston+tip/tilt with Tikhonov regularisation
        x_global = global_indices % global_shape[1]
        y_global = global_indices // global_shape[1]
        diff = np.diff(global_indices)
        group_starts = np.concatenate(([0], np.nonzero(diff)[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [global_indices.size]))
        row_offset = 0
        row_indices: list[np.ndarray] = []
        col_indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        for start, end in zip(group_starts, group_ends):
            group_size = end - start
            if group_size < 2:
                continue
            group_obs = obs_indices[start:end]
            group_vals = values[start:end]
            i_idx, j_idx = np.triu_indices(group_size, k=1)
            if not i_idx.size:
                continue
            obs_i = group_obs[i_idx]
            obs_j = group_obs[j_idx]
            x_i = x_global[start + i_idx]
            x_j = x_global[start + j_idx]
            y_i = y_global[start + i_idx]
            y_j = y_global[start + j_idx]
            row_ids = np.arange(i_idx.size) + row_offset
            col_i_piston = obs_i * 3
            col_i_tip = col_i_piston + 1
            col_i_tilt = col_i_piston + 2
            col_j_piston = obs_j * 3
            col_j_tip = col_j_piston + 1
            col_j_tilt = col_j_piston + 2
            col_idx = np.concatenate([
                np.full(i_idx.size, col_i_piston, dtype=int),
                np.full(i_idx.size, col_i_tip, dtype=int),
                np.full(i_idx.size, col_i_tilt, dtype=int),
                np.full(i_idx.size, col_j_piston, dtype=int),
                np.full(i_idx.size, col_j_tip, dtype=int),
                np.full(i_idx.size, col_j_tilt, dtype=int),
            ])
            row_indices.append(np.tile(row_ids, 6))
            col_indices.append(col_idx)
            data_arr = np.concatenate([
                np.full(i_idx.size, 1.0, dtype=float),
                x_i,
                y_i,
                -np.full(i_idx.size, 1.0, dtype=float),
                -x_j,
                -y_j,
            ])
            data.append(data_arr)
            rhs.append(group_vals[j_idx] - group_vals[i_idx])
            row_offset += i_idx.size
        if row_offset == 0:
            piston_shifts = np.zeros(len(observation_list), dtype=float)
            tip_shifts = np.zeros(len(observation_list), dtype=float)
            tilt_shifts = np.zeros(len(observation_list), dtype=float)
        else:
            row_idx = np.concatenate(row_indices)
            col_idx = np.concatenate(col_indices)
            data_arr = np.concatenate(data)
            rhs_arr = np.concatenate(rhs)
            # Add Tikhonov regularisation to stabilize the system
# lambda weight
lambda_ = 1e-6
reg_rows = np.arange(row_offset, row_offset + 3 * len(observation_list))
reg_cols = np.arange(3 * len(observation_list))
reg_data = np.full(3 * len(observation_list), lambda_, dtype=float)
# Combine with original system
row_idx = np.concatenate([row_idx, reg_rows])
col_idx = np.concatenate([col_idx, reg_cols])
data_arr = np.concatenate([data_arr, reg_data])
# Construct the augmented sparse matrix
system = sparse.coo_matrix((data_arr, (row_idx, col_idx)), shape=(row_offset + 3 * len(observation_list), len(observation_list) * 3))
sol = lsqr(system, rhs_arr)[0]
            sol = sol.reshape((len(observation_list), 3))
            piston_shifts = sol[:, 0]
            tip_shifts = sol[:, 1]
            tilt_shifts = sol[:, 2]
            piston_shifts = piston_shifts - float(np.mean(piston_shifts))
            tip_shifts = tip_shifts - float(np.mean(tip_shifts))
            tilt_shifts = tilt_shifts - float(np.mean(tilt_shifts))
        # Adjust values
        adjusted_values = values + piston_shifts[obs_indices] + tip_shifts[obs_indices] * x_global + tilt_shifts[obs_indices] * y_global
        global_area = global_shape[0] * global_shape[1]
        sum_z = np.bincount(global_indices, weights=adjusted_values, minlength=global_area).astype(float)
        count = np.bincount(global_indices, minlength=global_area).astype(int)
        valid_flat = count > 0
        averaged = np.zeros_like(sum_z)
        averaged[valid_flat] = sum_z[valid_flat] / count[valid_flat]
        z = averaged.reshape(global_shape)
        valid_mask = valid_flat.reshape(global_shape)
        observed_support_mask = valid_mask.copy()
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(source_observation_ids),
            observed_support_mask=observed_support_mask,
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
