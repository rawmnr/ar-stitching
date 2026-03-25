"""Simultaneous Calibration and Stitching (SCS) with SIAC-style Alternating Refinement."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

EDGE_EROSION_PX = 1
FEATHER_WIDTH = 0.04
SOLVE_FEATHER_WIDTH = 0.51
sigma_filter = 1.55
CALIBRATION_BLOCK = 1
CALIBRATION_SMOOTH_SIGMA = 0.75
CALIBRATION_MF_ALPHA = 0.38
CALIBRATION_MF_MIN_OBS = 8
CALIBRATION_MF_GATE_SIGMA = 0.5
CALIBRATION_BP_SIGMA = 0.5
CALIBRATION_MF_LO_SIGMA = 1.55
CALIBRATION_MF_COUNT_RAMP = 1.0
CALIBRATION_MF_CONSENSUS_GAIN = 3.5
CALIBRATION_MF_USE_CONSENSUS = False
CALIBRATION_MF_SUPPORT_AWARE_BP = False
CALIBRATION_MF_DETREND_LOWORDER = False
CALIBRATION_LF_UPDATE_BLEND = 0.50
CALIBRATION_MF_UPDATE_BLEND = 0.10
n_irls = 1
n_siac = 56
POSE_SHIFT_STEPS = (-0.5, 0.0, 0.5)
HF_SPLIT_SIGMA = 0
NUISANCE_DIM = 6
NUISANCE_REG_LAMBDA = 14.0
NUISANCE_QUADRATIC_DAMPING = 0.125

class CandidateStitcher:
    def _basis_family(self, config: ScenarioConfig) -> str:
        truth_basis = str(config.metadata.get("truth_basis", "")).lower()
        truth_pupil = str(config.metadata.get("truth_pupil", "")).lower()
        detector_pupil = str(config.metadata.get("detector_pupil", "")).lower()
        if truth_basis == "legendre" or truth_pupil == "square" or detector_pupil == "square":
            return "legendre"
        return "radial"

    def _low_order_terms(
        self,
        y_norm: np.ndarray,
        x_norm: np.ndarray,
        basis_family: str,
    ) -> np.ndarray:
        if basis_family == "legendre":
            return np.stack([
                np.ones_like(y_norm),
                y_norm,
                x_norm,
                0.5 * (3.0 * y_norm**2 - 1.0),
                0.5 * (3.0 * x_norm**2 - 1.0),
                y_norm * x_norm,
            ], axis=-1)
        return np.stack([
            np.ones_like(y_norm),
            y_norm,
            x_norm,
            x_norm**2 + y_norm**2,
            x_norm**2 - y_norm**2,
            2.0 * x_norm * y_norm,
        ], axis=-1)

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
        basis_family = self._basis_family(config)
        overlap_fraction = float(config.metadata.get("overlap_fraction", 0.25))
        self._calibration_mf_detrend_loworder = max(global_shape) <= 128

        master_mask = np.zeros(tile_shape, dtype=bool)
        for obs in observations:
            master_mask |= obs.valid_mask
            
        n_R_pixels = int(np.sum(master_mask))
        R_idx_map = np.full(tile_shape, -1, dtype=int)
        R_idx_map[master_mask] = np.arange(n_R_pixels)

        all_obs_indices, all_flat_indices, all_z = [], [], []
        all_xn, all_yn, all_r_idx = [], [], []
        all_solve_w = []

        for i, obs in enumerate(observations):
            top = int(round(obs.center_xy[1] - (tile_shape[0] - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (tile_shape[1] - 1) / 2.0))
            solve_weights = self._smooth_feather_weights(obs.valid_mask, feather_width=SOLVE_FEATHER_WIDTH)
            
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
            all_solve_w.append(solve_weights[yy, xx])

        obs_idx = np.concatenate(all_obs_indices)
        flat_idx = np.concatenate(all_flat_indices)
        z_vals = np.concatenate(all_z)
        xn_vals = np.concatenate(all_xn)
        yn_vals = np.concatenate(all_yn)
        r_idx_vals = np.concatenate(all_r_idx)
        solve_w_vals = np.concatenate(all_solve_w)

        sort_order = np.argsort(flat_idx)
        obs_idx = obs_idx[sort_order]
        flat_idx = flat_idx[sort_order]
        z_vals = z_vals[sort_order]
        xn_vals = xn_vals[sort_order]
        yn_vals = yn_vals[sort_order]
        r_idx_vals = r_idx_vals[sort_order]
        solve_w_vals = solve_w_vals[sort_order]

        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))
        
        # Compute overlap counts per pixel for inverse weighting
        overlap_counts = np.array([e - s for s, e in zip(boundaries[:-1], boundaries[1:])])

        rows_a, cols_a, data_a, b = [], [], [], []
        row_weights = []
        row_count = 0
        
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            n_overlap = e - s
            if n_overlap < 2:
                continue
            
            slice_indices = list(range(s, e))
            slice_indices.sort(key=lambda j: solve_w_vals[j], reverse=True)
            
            inv_overlap_w = 1.0 / max(n_overlap - 1, 1)
            
            for k in range(len(slice_indices) - 1):
                j = slice_indices[k]
                ref_o = obs_idx[j]
                ref_z = z_vals[j]
                ref_xn = xn_vals[j]
                ref_yn = yn_vals[j]
                ref_r = r_idx_vals[j]
                
                l = slice_indices[k + 1]
                oth_o = obs_idx[l]
                oth_z = z_vals[l]
                oth_xn = xn_vals[l]
                oth_yn = yn_vals[l]
                oth_r = r_idx_vals[l]
                
                row_weights.append(np.sqrt(solve_w_vals[j] * solve_w_vals[l]) * inv_overlap_w)
                
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
        row_weights_np = np.asarray(row_weights, dtype=float) if row_weights else np.ones(row_count, dtype=float)
        row_weights_np = np.clip(row_weights_np, 1e-6, None)
        
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
        r_low_order = self._low_order_terms(r_y_norm, r_x_norm, basis_family)
        
        for idx in range(n_R_pixels):
            col = n_obs * n_params + idx
            for term_idx in range(r_low_order.shape[1]):
                C_data.append(float(r_low_order[idx, term_idx]))
                C_rows.append(c_idx + term_idx)
                C_cols.append(col)
            
        c_idx += 6
            
        Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(c_idx, n_obs * n_params + n_R_pixels))
        
        lambda_reg = 1e-6
        
        robust_weights = np.ones(row_count, dtype=float)
        x = np.zeros(n_obs * n_params + n_R_pixels, dtype=float)
        
        for _ in range(n_irls):
            w_sqrt = np.sqrt(robust_weights * row_weights_np)
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
        
        nuisances = np.zeros((n_obs, NUISANCE_DIM), dtype=float)
        nuisances[:, :n_params] = x[:n_obs * n_params].reshape((n_obs, n_params))
        R_vals = x[n_obs * n_params:]
        
        R_map = np.zeros(tile_shape, dtype=float)
        R_map[master_mask] = R_vals
        
        R_lf = self._low_frequency_calibration_map(R_map, master_mask)
        R_mf = np.zeros_like(R_lf)
        R_map = R_lf + CALIBRATION_MF_ALPHA * R_mf
        pose_shifts = None
        
        for siac_iter in range(n_siac):
            fused_z, fused_mask = self._fuse_for_calibration(
                observations,
                nuisances,
                R_map,
                tile_shape,
                global_shape,
                basis_family=basis_family,
            )
            if not np.any(fused_mask):
                break
            
            R_lf_new, R_mf_new, mf_gate = self._estimate_reference_components(
                observations,
                fused_z,
                fused_mask,
                nuisances,
                tile_shape,
                master_mask,
            )
            R_lf = (1.0 - CALIBRATION_LF_UPDATE_BLEND) * R_lf + CALIBRATION_LF_UPDATE_BLEND * R_lf_new
            gated_mf_new = mf_gate * R_mf_new
            R_mf = (1.0 - CALIBRATION_MF_UPDATE_BLEND) * R_mf + CALIBRATION_MF_UPDATE_BLEND * gated_mf_new
            R_map_new = R_lf + CALIBRATION_MF_ALPHA * R_mf
            ref_delta = float(np.max(np.abs(R_map_new - R_map)))
            R_map = R_map_new
            
            # Refine nuisances every iteration using current calibration
            nuisances = self._refine_nuisances(
                observations,
                fused_z,
                fused_mask,
                R_map,
                tile_shape,
                nuisances,
                basis_family=basis_family,
            )
            
            if ref_delta < 1e-5:
                break

        pose_shifts = self._estimate_pose_shifts(
            observations,
            nuisances,
            R_map,
            fused_z,
            fused_mask,
            tile_shape,
            global_shape,
            basis_family=basis_family,
        )
        if np.any(pose_shifts):
            fused_z, fused_mask = self._fuse_for_calibration(
                observations,
                nuisances,
                R_map,
                tile_shape,
                global_shape,
                basis_family=basis_family,
                pose_shifts=pose_shifts,
            )
            nuisances = self._refine_nuisances(
                observations,
                fused_z,
                fused_mask,
                R_map,
                tile_shape,
                nuisances,
                basis_family=basis_family,
            )

        # R_map = self._project_degenerate_modes(R_map, master_mask)

        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)
        best_z = np.full(global_shape, np.nan, dtype=float)
        best_w = np.zeros(global_shape, dtype=float)

        for i, obs in enumerate(observations):
            yy, xx = np.indices(tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0
            
            model = self._nuisance_model(nuisances[i], y_norm, x_norm, basis_family=basis_family)
            
            z_corr = obs.z - model - R_map
            if pose_shifts is not None:
                z_corr = self._apply_pose_shift(z_corr, pose_shifts[i])
            
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
                
                region_mask = local_mask_orig & (local_weights > best_w[gy_s:gy_e, gx_s:gx_e])
                best_z[gy_s:gy_e, gx_s:gx_e][region_mask] = local_z[region_mask]
                best_w[gy_s:gy_e, gx_s:gx_e][region_mask] = local_weights[region_mask]

        valid_mask = count > 0
        z_mean = np.full(global_shape, np.nan, dtype=float)
        z_mean[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        
        z_final = np.full(global_shape, np.nan, dtype=float)
        if HF_SPLIT_SIGMA > 0:
            z_mean_filled = np.where(valid_mask, z_mean, 0.0)
            z_best_filled = np.where(valid_mask, best_z, 0.0)
            z_mean_lf = ndimage.gaussian_filter(z_mean_filled, sigma=HF_SPLIT_SIGMA)
            z_best_lf = ndimage.gaussian_filter(z_best_filled, sigma=HF_SPLIT_SIGMA)
            z_hf_from_best = best_z - z_best_lf
            z_final[valid_mask] = z_mean_lf[valid_mask] + z_hf_from_best[valid_mask]
        else:
            z_final = z_mean
        
        R_map_final = np.full(tile_shape, np.nan, dtype=float)
        R_map_final[master_mask] = R_map[master_mask]
        
        return ReconstructionSurface(
            z=z_final,
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

    def _nuisance_model(
        self,
        coeffs: np.ndarray,
        y_norm: np.ndarray,
        x_norm: np.ndarray,
        basis_family: str = "radial",
    ) -> np.ndarray:
        padded = np.zeros(NUISANCE_DIM, dtype=float)
        padded[: min(len(coeffs), NUISANCE_DIM)] = coeffs[:NUISANCE_DIM]
        if basis_family == "legendre":
            return (
                padded[0]
                + padded[1] * y_norm
                + padded[2] * x_norm
                + padded[3] * (0.5 * (3.0 * y_norm**2 - 1.0))
                + padded[4] * (0.5 * (3.0 * x_norm**2 - 1.0))
                + padded[5] * (y_norm * x_norm)
            )
        return (
            padded[0]
            + padded[1] * y_norm
            + padded[2] * x_norm
            + padded[3] * (x_norm**2 + y_norm**2)
            + padded[4] * (x_norm**2 - y_norm**2)
            + padded[5] * (2.0 * x_norm * y_norm)
        )

    def _fuse_for_calibration(self, observations, nuisances, R_map, tile_shape, global_shape, basis_family="radial", pose_shifts=None):
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        for i, obs in enumerate(observations):
            yy, xx = np.indices(tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0
            model = self._nuisance_model(nuisances[i], y_norm, x_norm, basis_family=basis_family)
            z_corr = obs.z - model - R_map
            if pose_shifts is not None:
                z_corr = self._apply_pose_shift(z_corr, pose_shifts[i])
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
        valid_mask = count > 0
        fused_z = np.full(global_shape, np.nan, dtype=float)
        fused_z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        return fused_z, valid_mask

    def _refine_nuisances(self, observations, fused_z, fused_mask, R_map, tile_shape, nuisances, basis_family="radial"):
        """Refine nuisance parameters using current fused surface and calibration map."""
        n_obs = len(observations)
        new_nuisances = nuisances.copy()
        
        for i, obs in enumerate(observations):
            rows, cols = tile_shape
            yy_full, xx_full = np.indices(tile_shape, dtype=float)
            y_norm_full = 2.0 * yy_full / max(rows - 1, 1) - 1.0
            x_norm_full = 2.0 * xx_full / max(cols - 1, 1) - 1.0
            
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))
            yy, xx = np.where(obs.valid_mask)
            if yy.size == 0:
                continue
            gy, gx = yy + top, xx + left
            valid_global = (
                (gy >= 0) & (gy < fused_z.shape[0]) &
                (gx >= 0) & (gx < fused_z.shape[1]) &
                fused_mask[gy, gx]
            )
            if not np.any(valid_global):
                continue
            
            yy, xx, gy, gx = yy[valid_global], xx[valid_global], gy[valid_global], gx[valid_global]
            
            # Target: obs.z - R_map - fused_z should equal nuisance model
            target = obs.z[yy, xx] - R_map[yy, xx] - fused_z[gy, gx]
            
            # Build design matrix for nuisance fit
            A_nuis = self._low_order_terms(
                y_norm_full[yy, xx],
                x_norm_full[yy, xx],
                basis_family,
            )

            A_reg = np.zeros((3, NUISANCE_DIM), dtype=float)
            A_reg[0, 3] = NUISANCE_REG_LAMBDA
            A_reg[1, 4] = NUISANCE_REG_LAMBDA
            A_reg[2, 5] = NUISANCE_REG_LAMBDA
            b_reg = np.zeros(3, dtype=float)

            coeff, *_ = np.linalg.lstsq(
                np.vstack([A_nuis, A_reg]),
                np.concatenate([target, b_reg]),
                rcond=None,
            )

            new_nuisances[i, :3] = coeff[:3]
            new_nuisances[i, 3:] = (
                (1.0 - NUISANCE_QUADRATIC_DAMPING) * nuisances[i, 3:]
                + NUISANCE_QUADRATIC_DAMPING * coeff[3:]
            )
        
        return new_nuisances

    def _estimate_reference_components(
        self,
        observations,
        fused_z,
        fused_mask,
        nuisances,
        tile_shape,
        master_mask,
        basis_family="radial",
    ):
        sum_r = np.zeros(tile_shape, dtype=float)
        sum_w = np.zeros(tile_shape, dtype=float)
        sum_mf = np.zeros(tile_shape, dtype=float)
        sum_mf_w = np.zeros(tile_shape, dtype=float)
        sum_mf_sq = np.zeros(tile_shape, dtype=float)
        n_contributing = np.zeros(tile_shape, dtype=float)
        for i, obs in enumerate(observations):
            rows, cols = tile_shape
            yy_full, xx_full = np.indices(tile_shape, dtype=float)
            y_norm_full = 2.0 * yy_full / max(rows - 1, 1) - 1.0
            x_norm_full = 2.0 * xx_full / max(cols - 1, 1) - 1.0
            model = self._nuisance_model(nuisances[i], y_norm_full, x_norm_full, basis_family=basis_family)
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))
            yy, xx = np.where(obs.valid_mask)
            if yy.size == 0:
                continue
            gy, gx = yy + top, xx + left
            valid_global = (
                (gy >= 0) & (gy < fused_z.shape[0]) &
                (gx >= 0) & (gx < fused_z.shape[1]) &
                fused_mask[gy, gx]
            )
            if not np.any(valid_global):
                continue
            yy, xx, gy, gx = yy[valid_global], xx[valid_global], gy[valid_global], gx[valid_global]
            residual = obs.z[yy, xx] - model[yy, xx] - fused_z[gy, gx]
            all_residuals = residual
            median = float(np.median(all_residuals))
            mad = float(np.median(np.abs(all_residuals - median)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(all_residuals)), 1e-6)
            c = max(6.5 * sigma, 1e-6)
            u = (residual - median) / c
            weights = np.zeros_like(u, dtype=float)
            inside = np.abs(u) < 1.0
            weights[inside] = (1.0 - u[inside] ** 2) ** 2
            sum_r[yy, xx] += weights * residual
            sum_w[yy, xx] += weights
            n_contributing[yy, xx] += (weights > 0.0).astype(float)

            mf_residual = residual
            if CALIBRATION_MF_DETREND_LOWORDER or getattr(self, "_calibration_mf_detrend_loworder", False):
                mf_residual = self._remove_detector_low_order(
                    mf_residual,
                    yy,
                    xx,
                    tile_shape,
                    basis_family=basis_family,
                )
            residual_image = np.zeros(tile_shape, dtype=float)
            residual_image[yy, xx] = mf_residual
            local_mask = np.zeros(tile_shape, dtype=bool)
            local_mask[yy, xx] = True
            if CALIBRATION_MF_SUPPORT_AWARE_BP:
                mf_hi = self._masked_gaussian_filter(
                    residual_image,
                    local_mask,
                    sigma=CALIBRATION_BP_SIGMA,
                )
                mf_lo = self._masked_gaussian_filter(
                    residual_image,
                    local_mask,
                    sigma=CALIBRATION_MF_LO_SIGMA,
                )
            else:
                mf_hi = ndimage.gaussian_filter(residual_image, sigma=CALIBRATION_BP_SIGMA)
                mf_lo = ndimage.gaussian_filter(residual_image, sigma=CALIBRATION_MF_LO_SIGMA)
            mf_image = mf_hi - mf_lo
            sum_mf[yy, xx] += weights * mf_image[yy, xx]
            sum_mf_w[yy, xx] += weights
            sum_mf_sq[yy, xx] += weights * mf_image[yy, xx] ** 2

        reference_raw = np.zeros(tile_shape, dtype=float)
        valid = sum_w > 0
        reference_raw[valid] = sum_r[valid] / sum_w[valid]

        reference_lf = self._low_frequency_calibration_map(reference_raw, valid)
        reference_mf = np.zeros_like(reference_raw)
        mf_valid = sum_mf_w > 0
        reference_mf[mf_valid] = sum_mf[mf_valid] / sum_mf_w[mf_valid]
        reference_mf = self._project_degenerate_modes(reference_mf, valid)

        count_gate = np.clip(
            (n_contributing - (CALIBRATION_MF_MIN_OBS - 1.0)) / max(CALIBRATION_MF_COUNT_RAMP, 1e-6),
            0.0,
            1.0,
        )
        gate = count_gate
        if CALIBRATION_MF_USE_CONSENSUS and np.any(mf_valid):
            mf_var = np.zeros_like(reference_raw)
            mf_var[mf_valid] = np.maximum(
                sum_mf_sq[mf_valid] / np.maximum(sum_mf_w[mf_valid], 1e-12)
                - reference_mf[mf_valid] ** 2,
                0.0,
            )
            mf_spread = np.sqrt(mf_var)
            scale_mask = mf_valid & (n_contributing >= CALIBRATION_MF_MIN_OBS)
            if np.any(scale_mask):
                spread_scale = float(np.median(mf_spread[scale_mask]))
            else:
                spread_scale = float(np.median(mf_spread[mf_valid]))
            spread_scale = max(spread_scale * CALIBRATION_MF_CONSENSUS_GAIN, 1e-6)
            consensus_gate = np.zeros_like(reference_raw)
            consensus_gate[mf_valid] = 1.0 / (1.0 + (mf_spread[mf_valid] / spread_scale) ** 2)
            gate *= consensus_gate

        gate = ndimage.gaussian_filter(gate, sigma=CALIBRATION_MF_GATE_SIGMA)
        gate = np.clip(gate, 0.0, 1.0)

        gated_reference_mf = np.zeros(tile_shape, dtype=float)
        gated_reference_mf[valid] = reference_mf[valid]
        gate[~valid] = 0.0
        return reference_lf, gated_reference_mf, gate

    def _smooth_feather_weights(self, valid_mask: np.ndarray, feather_width: float = FEATHER_WIDTH) -> np.ndarray:
        weights = np.zeros(valid_mask.shape, dtype=float)
        if not np.any(valid_mask):
            return weights
        
        dist = ndimage.distance_transform_edt(valid_mask)
        max_dist = np.max(dist)
        if max_dist <= 0:
            weights[valid_mask] = 1.0
            return weights
        
        feather_dist = feather_width * max_dist
        
        in_feather = valid_mask & (dist <= feather_dist)
        if np.any(in_feather):
            d_norm = dist[in_feather] / max(feather_dist, 1.0)
            weights[in_feather] = 0.5 * (1.0 - np.cos(np.pi * d_norm))
        
        plateau = valid_mask & (dist > feather_dist)
        weights[plateau] = 1.0
        
        return weights

    def _remove_detector_low_order(
        self,
        values: np.ndarray,
        yy: np.ndarray,
        xx: np.ndarray,
        tile_shape: tuple[int, int],
        basis_family: str = "radial",
    ) -> np.ndarray:
        if values.size == 0:
            return values
        rows, cols = tile_shape
        y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
        x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
        design = self._low_order_terms(y_norm, x_norm, basis_family)
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coeff

    def _masked_gaussian_filter(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        if sigma <= 0 or not np.any(mask):
            return np.where(mask, data, 0.0)
        weighted = ndimage.gaussian_filter(np.where(mask, data, 0.0), sigma=sigma)
        norm = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
        result = np.zeros_like(data, dtype=float)
        valid = norm > 1e-6
        result[valid] = weighted[valid] / norm[valid]
        return result

    def _low_frequency_calibration_map(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(data, dtype=float)

        block = int(CALIBRATION_BLOCK)
        if block <= 1:
            result = np.zeros_like(data, dtype=float)
            ref_filled = np.where(mask, data, 0.0)
            ref_smoothed = ndimage.gaussian_filter(ref_filled, sigma=sigma_filter)
            result[mask] = ref_smoothed[mask]
            return result

        h, w = data.shape
        pad_h = (-h) % block
        pad_w = (-w) % block
        if pad_h or pad_w:
            padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode="constant")
            padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        else:
            padded_data = data
            padded_mask = mask

        ph, pw = padded_data.shape
        coarse_h = ph // block
        coarse_w = pw // block
        reshaped_data = padded_data.reshape(coarse_h, block, coarse_w, block)
        reshaped_mask = padded_mask.reshape(coarse_h, block, coarse_w, block)
        sum_blocks = np.sum(reshaped_data * reshaped_mask, axis=(1, 3))
        count_blocks = np.sum(reshaped_mask, axis=(1, 3))

        coarse = np.zeros((coarse_h, coarse_w), dtype=float)
        valid = count_blocks > 0
        coarse[valid] = sum_blocks[valid] / count_blocks[valid]
        if np.any(valid):
            coarse = ndimage.gaussian_filter(coarse, sigma=CALIBRATION_SMOOTH_SIGMA)

        upsampled = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
        result = np.zeros_like(padded_data, dtype=float)
        result[:, :] = upsampled
        return result[:h, :w]

    def _estimate_pose_shifts(self, observations, nuisances, R_map, fused_z, fused_mask, tile_shape, global_shape, basis_family="radial"):
        shifts = np.zeros((len(observations), 2), dtype=float)
        if not np.any(fused_mask):
            return shifts

        yy, xx = np.indices(tile_shape, dtype=float)
        y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
        x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0

        for i, obs in enumerate(observations):
            model = self._nuisance_model(nuisances[i], y_norm, x_norm, basis_family=basis_family)
            z_corr = obs.z - model - R_map
            working_mask = self._get_eroded_mask(obs.valid_mask)

            cx, cy = obs.center_xy
            top = int(round(cy - (tile_shape[0] - 1) / 2.0))
            left = int(round(cx - (tile_shape[1] - 1) / 2.0))
            gy_s, gy_e = max(0, top), min(global_shape[0], top + tile_shape[0])
            gx_s, gx_e = max(0, left), min(global_shape[1], left + tile_shape[1])
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            if gy_e <= gy_s or gx_e <= gx_s:
                continue

            fused_view = fused_z[gy_s:gy_e, gx_s:gx_e]
            fused_mask_view = fused_mask[gy_s:gy_e, gx_s:gx_e]
            local_mask = working_mask[ly_s:ly_e, lx_s:lx_e] & fused_mask_view
            if not np.any(local_mask):
                continue

            best_shift = np.zeros(2, dtype=float)
            best_score = np.inf

            pose_steps = getattr(self, "_pose_shift_steps", POSE_SHIFT_STEPS)
            for dy in pose_steps:
                for dx in pose_steps:
                    shifted = self._apply_pose_shift(z_corr, (dy, dx))
                    local_z = shifted[ly_s:ly_e, lx_s:lx_e]
                    residual = local_z[local_mask] - fused_view[local_mask]
                    score = float(np.median(np.abs(residual)))
                    if score < best_score:
                        best_score = score
                        best_shift[:] = (dx, dy)

            shifts[i] = best_shift

        return shifts

    def _apply_pose_shift(self, tile: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
        dx, dy = shift_xy
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return tile
        return ndimage.shift(tile, shift=(dy, dx), order=3, mode="nearest", prefilter=True)
