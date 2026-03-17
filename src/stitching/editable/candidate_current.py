"""Least-Squares Stitcher for optical sub-aperture alignment."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

class CandidateStitcher:
    """Least-Squares stitcher estimating alignment nuisances from overlaps."""

    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        # 1. Estimate alignment parameters [Piston, Tip, Tilt, Focus] via Global Least Squares
        nuisances = self._solve_global_alignment(observations)
        
        global_shape = observations[0].global_shape
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=int)
        support = np.zeros(global_shape, dtype=bool)

        # 2. Correct and Assemble
        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            
            # Reconstruct the nuisance model for this observation
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
            
            p, tip, tilt, focus = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm + focus * (x_norm**2 + y_norm**2)
            
            z_corr = obs.z - model
            
            cx, cy = obs.center_xy
            top = int(round(cy - (rows - 1) / 2.0))
            left = int(round(cx - (cols - 1) / 2.0))
            
            gy_s, gy_e = max(0, top), min(global_shape[0], top + rows)
            gx_s, gx_e = max(0, left), min(global_shape[1], left + cols)
            
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            
            if gy_e > gy_s and gx_e > gx_s:
                local_z = z_corr[ly_s:ly_e, lx_s:lx_e]
                local_mask = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
                
                sum_z[gy_s:gy_e, gx_s:gx_e][local_mask] += local_z[local_mask]
                count[gy_s:gy_e, gx_s:gx_e][local_mask] += 1
                support[gy_s:gy_e, gx_s:gx_e][local_mask] = True

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={"method": "global_least_squares_full_alignment"},
        )

    def _solve_global_alignment(self, observations: tuple[SubApertureObservation, ...]) -> np.ndarray:
        """Solve for per-subaperture alignment nuisances [Piston, Tip, Tilt, Focus]."""
        n_obs = len(observations)
        n_params = 4 # [Piston, Tip, Tilt, Focus]
        if n_obs <= 1:
            return np.zeros((n_obs, n_params))

        global_shape = observations[0].global_shape
        
        # Vectorized collection of all valid pixels across all observations
        all_obs_indices = []
        all_flat_indices = []
        all_z = []
        all_xn = []
        all_yn = []
        all_fn = []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))
            
            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])
            
            # Sub-selection
            yy, xx = yy[valid_global], xx[valid_global]
            gy, gx = gy[valid_global], gx[valid_global]
            
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
            
            all_obs_indices.append(np.full(len(yy), i, dtype=int))
            all_flat_indices.append(gy * global_shape[1] + gx)
            all_z.append(obs.z[yy, xx])
            all_xn.append(x_norm)
            all_yn.append(y_norm)
            all_fn.append(x_norm**2 + y_norm**2)

        obs_idx = np.concatenate(all_obs_indices)
        flat_idx = np.concatenate(all_flat_indices)
        z_vals = np.concatenate(all_z)
        xn_vals = np.concatenate(all_xn)
        yn_vals = np.concatenate(all_yn)
        fn_vals = np.concatenate(all_fn)

        # Sort by flat_idx to group observations by pixel
        sort_order = np.argsort(flat_idx)
        obs_idx = obs_idx[sort_order]
        flat_idx = flat_idx[sort_order]
        z_vals = z_vals[sort_order]
        xn_vals = xn_vals[sort_order]
        yn_vals = yn_vals[sort_order]
        fn_vals = fn_vals[sort_order]

        # Find boundaries of pixel groups
        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))

        # Build Sparse Matrix A and vector b using groups
        rows_a, cols_a, data_a = [], [], []
        b = []
        row_count = 0
        
        # Only iterate over groups with size > 1 (overlaps)
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            if e - s < 2:
                continue
            
            # Use first obs in group as reference
            ref_o = obs_idx[s]
            ref_z = z_vals[s]
            ref_xn = xn_vals[s]
            ref_yn = yn_vals[s]
            ref_fn = fn_vals[s]
            
            for j in range(s + 1, e):
                oth_o = obs_idx[j]
                oth_z = z_vals[j]
                oth_xn = xn_vals[j]
                oth_yn = yn_vals[j]
                oth_fn = fn_vals[j]
                
                # Equation: Model_ref - Model_oth = S_ref - S_oth
                # Ref
                rows_a.extend([row_count] * 4)
                cols_a.extend([ref_o * n_params + k for k in range(4)])
                data_a.extend([1.0, ref_yn, ref_xn, ref_fn])
                # Oth
                rows_a.extend([row_count] * 4)
                cols_a.extend([oth_o * n_params + k for k in range(4)])
                data_a.extend([-1.0, -oth_yn, -oth_xn, -oth_fn])
                
                b.append(ref_z - oth_z)
                row_count += 1

        if not b:
            return np.zeros((n_obs, n_params))

        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params))
        
        # Constraint: Sum of parameters = 0
        C_data, C_rows, C_cols = [], [], []
        for k in range(n_params):
            for i in range(n_obs):
                C_data.append(1.0)
                C_rows.append(k)
                C_cols.append(i * n_params + k)
        
        Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(n_params, n_obs * n_params))
        
        A_final = sp.vstack([A, Constraint])
        b_final = np.concatenate([b, np.zeros(n_params)])

        # Solve LS with damping
        x, *_ = spla.lsqr(A_final, b_final, damp=1e-8, atol=1e-10, btol=1e-10)
        
        return x.reshape((n_obs, n_params))
