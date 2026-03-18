"""Stochastic Optimization Baseline (Simplified PSO/Random Search)."""
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
        # Initialize with standard GLS
        nuisances = self._solve_global_alignment(observations)
        
        # Stochastic Optimization (Random Search refining the GLS solution)
        # In a real PSO, we would have a swarm of particles. 
        # Here we do a simplified stochastic hill climbing to avoid local minima.
        nuisances = self._stochastic_refinement(observations, nuisances)
        
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
            metadata={"method": "stochastic_pso_baseline"},
        )

    def _stochastic_refinement(self, observations: tuple[SubApertureObservation, ...], nuisances: np.ndarray) -> np.ndarray:
        n_obs = len(observations)
        if n_obs <= 1:
            return nuisances
            
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

        def cost_function(n_arr):
            cost = 0.0
            for (r_o, o_o, r_z, o_z, r_xn, r_yn, o_xn, o_yn) in overlap_pairs:
                r_p, r_tip, r_tilt = n_arr[r_o, 0], n_arr[r_o, 1], n_arr[r_o, 2]
                o_p, o_tip, o_tilt = n_arr[o_o, 0], n_arr[o_o, 1], n_arr[o_o, 2]
                r_model = r_p + r_tip * r_yn + r_tilt * r_xn
                o_model = o_p + o_tip * o_yn + o_tilt * o_xn
                cost += abs((r_z - r_model) - (o_z - o_model))
            return cost

        current_nuisances = nuisances.copy()
        current_cost = cost_function(current_nuisances)
        
        rng = np.random.default_rng(42)
        best_nuisances = current_nuisances.copy()
        best_cost = current_cost
        
        # Simple random search (simulating stochastic exploration)
        for _ in range(50):
            # Perturb randomly
            noise = rng.normal(0, 1e-4, size=(n_obs, 4))
            noise[:, 3] = 0 # No focus
            test_nuisances = current_nuisances + noise
            
            # Constrain Piston mean to 0
            test_nuisances[:, 0] -= np.mean(test_nuisances[:, 0])
            
            test_cost = cost_function(test_nuisances)
            
            if test_cost < best_cost:
                best_cost = test_cost
                best_nuisances = test_nuisances.copy()
                current_nuisances = test_nuisances.copy()

        return best_nuisances

    def _solve_global_alignment(self, observations: tuple[SubApertureObservation, ...]) -> np.ndarray:
        n_obs = len(observations)
        n_params = 3
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

        rows_a, cols_a, data_a, b = [], [], [], []
        row_count = 0
        
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            if e - s < 2:
                continue
            
            ref_o = obs_idx[s]
            ref_z = z_vals[s]
            ref_xn = xn_vals[s]
            ref_yn = yn_vals[s]
            
            for j in range(s + 1, e):
                oth_o = obs_idx[j]
                oth_z = z_vals[j]
                oth_xn = xn_vals[j]
                oth_yn = yn_vals[j]
                
                rows_a.extend([row_count] * 3)
                cols_a.extend([ref_o * n_params + k for k in range(3)])
                data_a.extend([1.0, ref_yn, ref_xn])
                
                rows_a.extend([row_count] * 3)
                cols_a.extend([oth_o * n_params + k for k in range(3)])
                data_a.extend([-1.0, -oth_yn, -oth_xn])
                
                b.append(ref_z - oth_z)
                row_count += 1

        if not b:
            return np.zeros((n_obs, 4))
        
        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params))
        b_np = np.array(b)
        
        C_data, C_rows, C_cols = [], [], []
        for k in range(n_params):
            for i in range(n_obs):
                C_data.append(1.0)
                C_rows.append(k)
                C_cols.append(i * n_params + k)
        Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(n_params, n_obs * n_params))
        
        lambda_reg = 1e-4
        A_aug = sp.vstack([A, Constraint, lambda_reg * sp.eye(n_obs * n_params)])
        b_aug = np.concatenate([b_np, np.zeros(n_params), np.zeros(n_obs * n_params)])
        
        x, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-10, btol=1e-10)
        
        result = np.zeros((n_obs, 4), dtype=float)
        result[:, :3] = x.reshape((n_obs, n_params))
        return result
