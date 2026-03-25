"""Simultaneous Calibration and Stitching (SCS) - Simplified SCS with SCS gauge + IRLS."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

EDGE_EROSION_PX = 2
FEATHER_WIDTH = 0.20
sigma_filter = 1.2

class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        if not observations:
            return ReconstructionSurface(
                z=np.array([]),
                valid_mask=np.array([], dtype=bool),
                source_observation_ids=(),
                observed_support_mask=np.array([], dtype=bool)
            )
            
        n_obs = len(observations)
        n_params = 3
        tile_shape = observations[0].tile_shape
        global_shape = observations[0].global_shape

        master_mask = np.zeros(tile_shape, dtype=bool)
        for obs in observations:
            master_mask |= obs.valid_mask
            
        n_R_pixels = int(np.sum(master_mask))
        R_idx_map = np.full(tile_shape, -1, dtype=int)
        R_idx_map[master_mask] = np.arange(n_R_pixels)

        all_obs_indices, all_flat_indices, all_z = [], [], []
        all_xn, all_yn, all_r_idx = [], [], []

        for i, obs in enumerate(observations):
            top = int(round(obs.center_xy[1] - (tile_shape[0] - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (tile_shape[1] - 1) / 2.0))
            
            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])
            
            yy, xx = yy[valid_global], xx[valid_global]
            gy, gx = gy[valid_global], gx[valid_global]
            
            y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0
            
            all_obs_indices.append(np.full(len(yy), i, dtype=int))
            all_flat_indices.append(gy * global_shape[1] + gx)
            all_z.append(obs.z[yy, xx])
            all_xn.append(x_norm)
            all_yn.append(y_norm)
            all_r_idx.append(R_idx_map[yy, xx])

        obs_idx = np.concatenate(all_obs_indices)
        flat_idx = np.concatenate(all_flat_indices)
        z_vals = np.concatenate(all_z)
        xn_vals = np.concatenate(all_xn)
        yn_vals = np.concatenate(all_yn)
        r_idx_vals = np.concatenate(all_r_idx)

        sort_order = np.argsort(flat_idx)
        obs_idx = obs_idx[sort_order]
        flat_idx = flat_idx[sort_order]
        z_vals = z_vals[sort_order]
        xn_vals = xn_vals[sort_order]
        yn_vals = yn_vals[sort_order]
        r_idx_vals = r_idx_vals[sort_order]

        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))

        rows_a, cols_a, data_a, b = [], [], [], []
        row_count = 0
        
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            if e - s < 2:
                continue
            
            for j in range(s, e - 1):
                ref_o = obs_idx[j]
                ref_z = z_vals[j]
                ref_xn = xn_vals[j]
                ref_yn = yn_vals[j]
                ref_r = r_idx_vals[j]
                
                oth_o = obs_idx[j+1]
                oth_z = z_vals[j+1]
                oth_xn = xn_vals[j+1]
                oth_yn = yn_vals[j+1]
                oth_r = r_idx_vals[j+1]
                
                rows_a.extend([row_count] * 3)
                cols_a.extend([ref_o * n_params + k for k in range(3)])
                data_a.extend([1.0, ref_yn, ref_xn])
                
                rows_a.append(row_count)
                cols_a.append(n_obs * n_params + ref_r)
                data_a.append(1.0)
                
                rows_a.extend([row_count] * 3)
                cols_a.extend([oth_o * n_params + k for k in range(3)])
                data_a.extend([-1.0, -oth_yn, -oth_xn])
                
                rows_a.append(row_count)
                cols_a.append(n_obs * n_params + oth_r)
                data_a.append(-1.0)
                
                b.append(ref_z - oth_z)
                row_count += 1

        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params + n_R_pixels))
        b_np = np.array(b)
        
        C_data, C_rows, C_cols = [], [], []
        c_idx = 0
        
        for i in range(n_obs):
            C_data.append(1.0)
            C_rows.append(c_idx)
            C_cols.append(i * n_params)
        c_idx += 1
        
        for i in range(n_obs):
            C_data.append(1.0)
            C_rows.append(c_idx)
            C_cols.append(i * n_params + 1)
        c_idx += 1
        
        for i in range(n_obs):
            C_data.append(1.0)
            C_rows.append(c_idx)
            C_cols.append(i * n_params + 2)
        c_idx += 1
        
        r_yy, r_xx = np.where(master_mask)
        r_y_norm = 2.0 * r_yy / max(tile_shape[0] - 1, 1) - 1.0
        r_x_norm = 2.0 * r_xx / max(tile_shape[1] - 1, 1) - 1.0
        r_rad2 = r_x_norm**2 + r_y_norm**2
        
        for idx in range(n_R_pixels):
            col = n_obs * n_params + idx
            C_data.append(1.0); C_rows.append(c_idx); C_cols.append(col)
            C_data.append(r_y_norm[idx]); C_rows.append(c_idx + 1); C_cols.append(col)
            C_data.append(r_x_norm[idx]); C_rows.append(c_idx + 2); C_cols.append(col)
            C_data.append(r_rad2[idx]); C_rows.append(c_idx + 3); C_cols.append(col)
            C_data.append(r_x_norm[idx]**2 - r_y_norm[idx]**2); C_rows.append(c_idx + 4); C_cols.append(col)
            C_data.append(2.0 * r_x_norm[idx] * r_y_norm[idx]); C_rows.append(c_idx + 5); C_cols.append(col)
            
        c_idx += 6
            
        Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(c_idx, n_obs * n_params + n_R_pixels))
        
        lambda_reg = 1e-6
        
        robust_weights = np.ones(row_count, dtype=float)
        x = np.zeros(n_obs * n_params + n_R_pixels, dtype=float)
        
        for _ in range(5):
            w_sqrt = np.sqrt(robust_weights)
            W = sp.diags(w_sqrt)
            A_w = W @ A
            b_w = w_sqrt * b_np
            
            A_aug = sp.vstack([A_w, Constraint, lambda_reg * sp.eye(n_obs * n_params + n_R_pixels)])
            b_aug = np.concatenate([b_w, np.zeros(c_idx), np.zeros(n_obs * n_params + n_R_pixels)])
            
            x_new, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-8, btol=1e-8)
            
            residuals = A.dot(x_new) - b_np
            
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(residuals)), 1e-6)
            c = max(1.345 * sigma, 1e-6)
            
            abs_r = np.abs(residuals)
            robust_weights = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-12))
            
            if np.max(np.abs(x_new - x)) < 1e-6:
                x = x_new
                break
            x = x_new
        
        nuisances = x[:n_obs * n_params].reshape((n_obs, n_params))
        R_vals = x[n_obs * n_params:]
        
        R_map = np.zeros(tile_shape, dtype=float)
        R_map[master_mask] = R_vals
        
        ref_filled = np.where(master_mask, R_map, 0.0)
        ref_smoothed = ndimage.gaussian_filter(ref_filled, sigma=sigma_filter)
        R_map[master_mask] = ref_smoothed[master_mask]
        
        R_map = self._project_degenerate_modes(R_map, master_mask)

        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)

        for i, obs in enumerate(observations):
            yy, xx = np.indices(tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0
            
            p, tip, tilt = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm
            
            z_corr = obs.z - model - R_map
            
            working_mask = self._get_eroded_mask(obs.valid_mask)
            feather_weights = self._smooth_feather_weights(working_mask)
            
            cx, cy = obs.center_xy
            top = int(round(cy - (tile_shape[0] - 1) / 2.0))
            left = int(round(cx - (tile_shape[1] - 1) / 2.0))
            
            gy_s, gy_e = max(0, top), min(global_shape[0], top + tile_shape[0])
            gx_s, gx_e = max(0, left), min(global_shape[1], left + tile_shape[1])
            
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            
            if gy_e > gy_s and gx_e > gx_s:
                local_z = z_corr[ly_s:ly_e, lx_s:lx_e]
                local_mask_orig = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
                local_weights = feather_weights[ly_s:ly_e, lx_s:lx_e]
                
                sum_z[gy_s:gy_e, gx_s:gx_e][local_mask_orig] += local_z[local_mask_orig] * local_weights[local_mask_orig]
                count[gy_s:gy_e, gx_s:gx_e][local_mask_orig] += local_weights[local_mask_orig]
                support[gy_s:gy_e, gx_s:gx_e][local_mask_orig] = True

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        
        R_map_final = np.full(tile_shape, np.nan, dtype=float)
        R_map_final[master_mask] = R_map[master_mask]
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={"method": "scs_simultaneous", "instrument_calibration": R_map_final},
        )

    def _get_eroded_mask(self, valid_mask: np.ndarray) -> np.ndarray:
        if EDGE_EROSION_PX <= 0:
            return valid_mask.copy()
        structure = np.ones((3, 3), dtype=bool)
        eroded = ndimage.binary_erosion(valid_mask, structure=structure, iterations=EDGE_EROSION_PX)
        return eroded

    def _project_degenerate_modes(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(data, dtype=float)
        if not np.any(mask):
            return result
        yy, xx = np.indices(data.shape, dtype=float)
        y_norm = 2.0 * yy[mask] / max(data.shape[0] - 1, 1) - 1.0
        x_norm = 2.0 * xx[mask] / max(data.shape[1] - 1, 1) - 1.0
        A = np.column_stack([
            np.ones(mask.sum(), dtype=float), y_norm, x_norm,
            x_norm**2 + y_norm**2, x_norm**2 - y_norm**2, 2.0 * x_norm * y_norm
        ])
        coeff, *_ = np.linalg.lstsq(A, data[mask], rcond=None)
        result[mask] = data[mask] - (A @ coeff)
        return result

    def _smooth_feather_weights(self, valid_mask: np.ndarray) -> np.ndarray:
        weights = np.zeros(valid_mask.shape, dtype=float)
        if not np.any(valid_mask):
            return weights
        
        dist = ndimage.distance_transform_edt(valid_mask)
        max_dist = np.max(dist)
        if max_dist <= 0:
            weights[valid_mask] = 1.0
            return weights
        
        feather_dist = FEATHER_WIDTH * max_dist
        
        in_feather = valid_mask & (dist <= feather_dist)
        if np.any(in_feather):
            d_norm = dist[in_feather] / max(feather_dist, 1.0)
            weights[in_feather] = 0.5 * (1.0 - np.cos(np.pi * d_norm))
        
        plateau = valid_mask & (dist > feather_dist)
        weights[plateau] = 1.0
        
        return weights
