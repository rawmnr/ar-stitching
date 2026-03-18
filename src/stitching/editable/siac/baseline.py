"""Stitching Interferometry using Alternating Calibration (SIAC) Baseline."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        # SIAC separates solving for Piston and Tip/Tilt to prevent crosstalk
        nuisances = self._solve_alternating(observations)
        
        global_shape = observations[0].global_shape
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
            
            p, tip, tilt, focus = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm
            z_corr = obs.z - model
            
            weights = np.ones_like(z_corr)
            
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
                local_weights = weights[ly_s:ly_e, lx_s:lx_e]
                
                sum_z[gy_s:gy_e, gx_s:gx_e][local_mask] += local_z[local_mask] * local_weights[local_mask]
                count[gy_s:gy_e, gx_s:gx_e][local_mask] += local_weights[local_mask]
                support[gy_s:gy_e, gx_s:gx_e][local_mask] = True

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={"method": "siac_alternating"},
        )

    def _solve_alternating(self, observations: tuple[SubApertureObservation, ...]) -> np.ndarray:
        n_obs = len(observations)
        if n_obs <= 1:
            return np.zeros((n_obs, 4))
            
        global_shape = observations[0].global_shape
        all_obs_indices, all_flat_indices, all_z, all_xn, all_yn = [], [], [], [], []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))
            
            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])
            
            yy, xx = yy[valid_global], xx[valid_global]
            gy, gx = gy[valid_global], gx[valid_global]
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
            
            all_obs_indices.append(np.full(len(yy), i, dtype=int))
            all_flat_indices.append(gy * global_shape[1] + gx)
            all_z.append(obs.z[yy, xx])
            all_xn.append(x_norm)
            all_yn.append(y_norm)

        obs_idx = np.concatenate(all_obs_indices)
        flat_idx = np.concatenate(all_flat_indices)
        z_vals = np.concatenate(all_z)
        xn_vals = np.concatenate(all_xn)
        yn_vals = np.concatenate(all_yn)

        sort_order = np.argsort(flat_idx)
        obs_idx = obs_idx[sort_order]
        flat_idx = flat_idx[sort_order]
        z_vals = z_vals[sort_order]
        xn_vals = xn_vals[sort_order]
        yn_vals = yn_vals[sort_order]

        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))

        overlap_pairs = []
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            if e - s < 2:
                continue
            ref_o = obs_idx[s]
            for j in range(s + 1, e):
                overlap_pairs.append((ref_o, obs_idx[j], z_vals[s], z_vals[j], xn_vals[s], yn_vals[s], xn_vals[j], yn_vals[j]))

        nuisances = np.zeros((n_obs, 4)) # p, tip, tilt, focus
        max_alternating_iter = 3
        
        for _ in range(max_alternating_iter):
            # Step 1: Solve for Tip/Tilt assuming Piston is fixed
            rows_a, cols_a, data_a, b = [], [], [], []
            row_count = 0
            for (r_o, o_o, r_z, o_z, r_xn, r_yn, o_xn, o_yn) in overlap_pairs:
                r_p = nuisances[r_o, 0]
                o_p = nuisances[o_o, 0]
                
                rows_a.extend([row_count] * 2)
                cols_a.extend([r_o * 2 + 0, r_o * 2 + 1])
                data_a.extend([r_yn, r_xn])
                
                rows_a.extend([row_count] * 2)
                cols_a.extend([o_o * 2 + 0, o_o * 2 + 1])
                data_a.extend([-o_yn, -o_xn])
                
                b.append((r_z - r_p) - (o_z - o_p))
                row_count += 1
                
            if b:
                A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * 2))
                b_np = np.array(b)
                x, *_ = spla.lsqr(A, b_np, damp=1e-8, atol=1e-10, btol=1e-10)
                nuisances[:, 1:3] = x.reshape((n_obs, 2))
                
            # Step 2: Solve for Piston assuming Tip/Tilt is fixed
            rows_a, cols_a, data_a, b = [], [], [], []
            row_count = 0
            for (r_o, o_o, r_z, o_z, r_xn, r_yn, o_xn, o_yn) in overlap_pairs:
                r_tip, r_tilt = nuisances[r_o, 1], nuisances[r_o, 2]
                o_tip, o_tilt = nuisances[o_o, 1], nuisances[o_o, 2]
                
                r_t_model = r_tip * r_yn + r_tilt * r_xn
                o_t_model = o_tip * o_yn + o_tilt * o_xn
                
                rows_a.extend([row_count, row_count])
                cols_a.extend([r_o, o_o])
                data_a.extend([1.0, -1.0])
                
                b.append((r_z - r_t_model) - (o_z - o_t_model))
                row_count += 1
                
            if b:
                A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs))
                b_np = np.array(b)
                C_data, C_rows, C_cols = [], [], []
                for i in range(n_obs):
                    C_data.append(1.0)
                    C_rows.append(0)
                    C_cols.append(i)
                Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(1, n_obs))
                A_aug = sp.vstack([A, Constraint, 1e-4 * sp.eye(n_obs)])
                b_aug = np.concatenate([b_np, [0.0], np.zeros(n_obs)])
                
                x, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-10, btol=1e-10)
                nuisances[:, 0] = x[:n_obs]
                
        return nuisances
