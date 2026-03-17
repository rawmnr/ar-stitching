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
        # 1. Estimate alignment parameters (Piston) via Global Least Squares
        nuisances = self._solve_global_alignment(observations)
        
        global_shape = observations[0].global_shape
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=int)
        support = np.zeros(global_shape, dtype=bool)

        # 2. Correct and Assemble
        for i, obs in enumerate(observations):
            # Apply estimated piston correction
            piston = nuisances[i, 0]
            z_corr = obs.z - piston
            
            cx, cy = obs.center_xy
            rows, cols = obs.tile_shape
            
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
            metadata={"method": "global_least_squares_piston"},
        )

    def _solve_global_alignment(self, observations: tuple[SubApertureObservation, ...]) -> np.ndarray:
        """Solve for per-subaperture alignment nuisances (currently Piston only)."""
        n_obs = len(observations)
        if n_obs <= 1:
            return np.zeros((n_obs, 4)) # [Piston, Tip, Tilt, Focus]

        # Map global pixels to contributions for overlap detection
        global_shape = observations[0].global_shape
        pixel_to_obs = [[] for _ in range(global_shape[0] * global_shape[1])]
        
        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))
            
            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])
            
            flat_idx = gy[valid_global] * global_shape[1] + gx[valid_global]
            for idx, val in zip(flat_idx, obs.z[yy[valid_global], xx[valid_global]]):
                pixel_to_obs[idx].append((i, val))

        # Build Sparse Matrix A and vector b
        # Equation: (P_i - P_j) = S_i - S_j
        rows_a, cols_a, data_a = [], [], []
        b = []
        
        row_idx = 0
        for contributions in pixel_to_obs:
            if len(contributions) < 2:
                continue
            
            # Use the first observation in the pixel as a reference for others
            ref_idx, ref_val = contributions[0]
            for i in range(1, len(contributions)):
                other_idx, other_val = contributions[i]
                
                # Row: 1*P_ref - 1*P_other = S_ref - S_other
                rows_a.extend([row_idx, row_idx])
                cols_a.extend([ref_idx, other_idx])
                data_a.extend([1.0, -1.0])
                b.append(ref_val - other_val)
                row_idx += 1

        if not b:
            return np.zeros((n_obs, 4))

        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_idx, n_obs))
        
        # Regularize to fix global piston (set P0 = 0)
        # Adding a small identity term or a hard constraint
        A_reg = sp.vstack([A, sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n_obs))])
        b_reg = np.concatenate([b, [0.0]])

        # Solve LS with damping (Tikhonov regularization) to prevent over-fitting
        # especially for higher order modes like focus if they are added later.
        x, *_ = spla.lsqr(A_reg, b_reg, damp=1e-4, atol=1e-8, btol=1e-8)
        
        nuisances = np.zeros((n_obs, 4))
        nuisances[:, 0] = x
        return nuisances
