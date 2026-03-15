"""GLS piston correction candidate for optical stitching."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.trusted.scan.transforms import placement_slices


# Hypothesis: Estimating per-observation piston offsets via GLS on overlap residuals reduces the aggregate RMS.
class CandidateStitcher:
    """Candidate that aligns overlapping tiles using a sparse GLS piston solve."""

    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        """Aggregate observations after global piston correction."""

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

            global_y, global_x, local_y, local_x = placement_slices(
                observation.global_shape,
                observation.tile_shape,
                observation.center_xy,
            )

            local_mask = np.asarray(observation.valid_mask, dtype=bool)[local_y, local_x]
            if not local_mask.any():
                continue

            valid_rows, valid_cols = np.nonzero(local_mask)
            global_rows = global_y.start + valid_rows
            global_cols = global_x.start + valid_cols
            linear_indices = np.ravel_multi_index((global_rows, global_cols), global_shape)
            local_values = np.asarray(observation.z, dtype=float)[local_y, local_x][valid_rows, valid_cols]

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
                metadata=self._metadata(observation_list, tile_centers, (), ()),
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
                z=zero_z, valid_mask=zero_mask,
                source_observation_ids=tuple(source_observation_ids),
                observed_support_mask=zero_mask,
                metadata=self._metadata(observation_list, tile_centers, (), ()),
            )

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

            row_ids = np.repeat(np.arange(i_idx.size, dtype=int) + row_offset, 2)
            cols = np.empty(i_idx.size * 2, dtype=int)
            cols[0::2] = group_obs[i_idx]
            cols[1::2] = group_obs[j_idx]

            coeffs = np.empty(i_idx.size * 2, dtype=float)
            coeffs[0::2] = 1.0
            coeffs[1::2] = -1.0

            row_indices.append(row_ids)
            col_indices.append(cols)
            data.append(coeffs)
            rhs.append(group_vals[j_idx] - group_vals[i_idx])

            row_offset += i_idx.size

        if row_offset == 0:
            piston_shifts = np.zeros(len(observation_list), dtype=float)
        else:
            row_idx = np.concatenate(row_indices)
            col_idx = np.concatenate(col_indices)
            data_arr = np.concatenate(data)
            rhs_arr = np.concatenate(rhs)

            system = sparse.coo_matrix(
                (data_arr, (row_idx, col_idx)),
                shape=(row_offset, len(observation_list)),
            )
            piston_shifts = lsqr(system, rhs_arr)[0]
            piston_shifts = piston_shifts - float(np.mean(piston_shifts))

        adjusted_values = values + piston_shifts[obs_indices]
        global_area = global_shape[0] * global_shape[1]
        sum_z = np.zeros(global_area, dtype=float)
        count = np.zeros(global_area, dtype=int)
        np.add.at(sum_z, global_indices, adjusted_values)
        np.add.at(count, global_indices, 1)

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
                piston_shifts.tolist(), (),
            ),
        )

    @staticmethod
    def _metadata(
        observations: Sequence[SubApertureObservation],
        tile_centers: list[tuple[float, float]],
        piston_shifts: Sequence[float],
        extras: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        first = observations[0]
        centers = tile_centers[0] if len(tile_centers) == 1 else tuple(tile_centers)
        metadata: dict[str, object] = {
            "baseline": "gls_piston",
            "baseline_experimental": True,
            "reconstruction_frame": "global_truth",
            "tile_centers_xy": centers,
            "num_observations_used": len(observations),
            "piston_shifts": tuple(float(s) for s in piston_shifts),
            **dict(first.metadata),
        }
        for key, value in extras:
            metadata[key] = value
        return metadata
