"""Simultaneous Calibration and Stitching (SCS) with SIAC-style Alternating Refinement."""
from __future__ import annotations

import os
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

class StitchingConfig:
    """Configuration for the stitching algorithm, derived from geometry or overridden via environment."""
    def __init__(self, **kwargs):
        # === HYPERPARAMETERS (Tunable) ===
        self.nuisance_reg_lambda = kwargs.get("nuisance_reg_lambda", 15.0)
        self.feather_width = kwargs.get("feather_width", 0.04)
        self.solve_feather_width = kwargs.get("solve_feather_width", 0.51)
        self.siac_convergence_tol = kwargs.get("siac_convergence_tol", 1e-5)
        self.max_siac_iter = kwargs.get("max_siac_iter", 200)
        self.c_sigma = kwargs.get("c_sigma", 110.0)  # Historical basin; still sweepable via STITCH_CONFIG
        
        # Blending and stabilization
        self.calibration_lf_update_blend = kwargs.get("calibration_lf_update_blend", 0.50)
        self.calibration_mf_update_blend = kwargs.get("calibration_mf_update_blend", 0.10)
        self.calibration_mf_alpha = kwargs.get("calibration_mf_alpha", 0.38)
        self.nuisance_quadratic_damping = kwargs.get("nuisance_quadratic_damping", 0.125)
        
        # Fixed architecture choices
        self.n_irls = kwargs.get("n_irls", 1)
        self.nuisance_dim = kwargs.get("nuisance_dim", 6)
        self.pose_shift_steps = kwargs.get("pose_shift_steps", (-0.5, 0.0, 0.5))
        
        # === DERIVED PARAMETERS (Calculated from geometry) ===
        self.sigma_filter = 1.55
        self.edge_erosion_px = 1
        self.calibration_bp_sigma = 0.5
        self.calibration_mf_lo_sigma = 1.55
        self.calibration_mf_min_obs = 8
        self.basis_family = "radial"
    
    def derive_from_geometry(
        self,
        tile_shape: tuple[int, int],
        n_obs: int,
        redundancy: float = 3.0,
        detector_pupil: str = "circular",
        truth_pupil: str = "circular",
    ):
        """Compute all resolution-dependent parameters."""
        tile_min = min(tile_shape)
        
        # Sigma filter: scales with tile size AND redundancy
        # More redundancy (overlap) -> smaller sigma
        # Reference redundancy is ~3.0 for standard scans
        raw_sigma = (tile_min / self.c_sigma) / np.sqrt(max(redundancy, 1.0) / 3.0)
        self.sigma_filter = float(np.clip(raw_sigma, 0.8, 3.0))
        
        # Erosion: at least 1px, approx 1% of tile size
        self.edge_erosion_px = max(1, tile_min // 100)
        
        # Bandwidth for MF calibration
        self.calibration_bp_sigma = max(0.3, tile_min / 256.0)
        self.calibration_mf_lo_sigma = self.sigma_filter
        self.calibration_mf_min_obs = max(3, int(n_obs * 0.35))
        
        if detector_pupil == "square" or truth_pupil == "square":
            self.basis_family = "legendre"
        else:
            self.basis_family = "radial"

def _get_config_overrides() -> dict:
    """Reads configuration overrides from STITCH_CONFIG environment variable."""
    raw = os.environ.get("STITCH_CONFIG", "{}")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

class CandidateStitcher:
    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        if not observations:
            return ReconstructionSurface(
                z=np.array([]), valid_mask=np.array([], dtype=bool),
                source_observation_ids=(), observed_support_mask=np.array([], dtype=bool)
            )
            
        n_obs = len(observations)
        tile_shape = observations[0].tile_shape
        global_shape = observations[0].global_shape
        
        # Initial master mask for redundancy calculation
        master_mask = np.zeros(tile_shape, dtype=bool)
        total_valid_pix = 0
        for obs in observations:
            master_mask |= obs.valid_mask
            total_valid_pix += np.sum(obs.valid_mask)
        
        master_valid_pix = np.sum(master_mask)
        redundancy = total_valid_pix / master_valid_pix if master_valid_pix > 0 else 1.0
        
        # Config with overrides
        overrides = _get_config_overrides()
        self.cfg = StitchingConfig(**overrides)
        self.cfg.derive_from_geometry(
            tile_shape=tile_shape,
            n_obs=n_obs,
            redundancy=redundancy,
            detector_pupil=str(config.metadata.get("detector_pupil", "circular")),
            truth_pupil=str(config.metadata.get("truth_pupil", "circular")),
        )
        
        basis_family = self.cfg.basis_family
        n_params = 3 # Nuisance params (piston, tip, tilt)
        self._calibration_mf_detrend_loworder = max(global_shape) <= 128

        n_R_pixels = int(master_valid_pix)
        R_idx_map = np.full(tile_shape, -1, dtype=int)
        R_idx_map[master_mask] = np.arange(n_R_pixels)

        all_obs_indices, all_flat_indices, all_z = [], [], []
        all_xn, all_yn, all_r_idx, all_solve_w = [], [], [], []

        for i, obs in enumerate(observations):
            top = int(round(obs.center_xy[1] - (tile_shape[0] - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (tile_shape[1] - 1) / 2.0))
            solve_weights = self._smooth_feather_weights(obs.valid_mask, feather_width=self.cfg.solve_feather_width)
            
            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])
            
            yy, xx, gy, gx = yy[valid_global], xx[valid_global], gy[valid_global], gx[valid_global]
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
        obs_idx, flat_idx, z_vals, xn_vals, yn_vals, r_idx_vals, solve_w_vals = (
            v[sort_order] for v in (obs_idx, flat_idx, z_vals, xn_vals, yn_vals, r_idx_vals, solve_w_vals)
        )

        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))
        
        rows_a, cols_a, data_a, b, row_weights = [], [], [], [], []
        row_count = 0
        
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            n_overlap = e - s
            if n_overlap < 2: continue
            slice_indices = list(range(s, e))
            slice_indices.sort(key=lambda j: solve_w_vals[j], reverse=True)
            inv_overlap_w = 1.0 / max(n_overlap - 1, 1)
            
            for k in range(len(slice_indices) - 1):
                j, l = slice_indices[k], slice_indices[k + 1]
                row_weights.append(np.sqrt(solve_w_vals[j] * solve_w_vals[l]) * inv_overlap_w)
                rows_a.extend([row_count] * 3); cols_a.extend([obs_idx[j] * n_params + m for m in range(3)]); data_a.extend([1.0, yn_vals[j], xn_vals[j]])
                rows_a.append(row_count); cols_a.append(n_obs * n_params + r_idx_vals[j]); data_a.append(1.0)
                rows_a.extend([row_count] * 3); cols_a.extend([obs_idx[l] * n_params + m for m in range(3)]); data_a.extend([-1.0, -yn_vals[l], -xn_vals[l]])
                rows_a.append(row_count); cols_a.append(n_obs * n_params + r_idx_vals[l]); data_a.append(-1.0)
                b.append(z_vals[j] - z_vals[l])
                row_count += 1

        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params + n_R_pixels))
        b_np, row_weights_np = np.array(b), np.clip(np.asarray(row_weights, dtype=float), 1e-6, None)
        
        C_data, C_rows, C_cols, c_idx = [], [], [], 0
        for m in range(3):
            for i in range(n_obs):
                C_data.append(1.0); C_rows.append(c_idx); C_cols.append(i * n_params + m)
            c_idx += 1
        
        r_yy, r_xx = np.where(master_mask)
        r_y_norm, r_x_norm = 2.0 * r_yy / max(tile_shape[0]-1, 1) - 1.0, 2.0 * r_xx / max(tile_shape[1]-1, 1) - 1.0
        r_low_order = self._low_order_terms(r_y_norm, r_x_norm, basis_family)
        for idx in range(n_R_pixels):
            col = n_obs * n_params + idx
            for term_idx in range(r_low_order.shape[1]):
                C_data.append(float(r_low_order[idx, term_idx])); C_rows.append(c_idx + term_idx); C_cols.append(col)
        c_idx += 6
        Constraint = sp.csr_matrix((C_data, (C_rows, C_cols)), shape=(c_idx, n_obs * n_params + n_R_pixels))
        
        lambda_reg, robust_weights, x = 1e-6, np.ones(row_count, dtype=float), np.zeros(n_obs * n_params + n_R_pixels, dtype=float)
        for _ in range(self.cfg.n_irls):
            w_sqrt = np.sqrt(robust_weights * row_weights_np)
            A_w, b_w = sp.diags(w_sqrt) @ A, w_sqrt * b_np
            A_aug = sp.vstack([A_w, Constraint, lambda_reg * sp.eye(n_obs * n_params + n_R_pixels)])
            b_aug = np.concatenate([b_w, np.zeros(c_idx), np.zeros(n_obs * n_params + n_R_pixels)])
            x_new, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-8, btol=1e-8)
            residuals = A.dot(x_new) - b_np
            sigma = np.median(np.abs(residuals - np.median(residuals))) / 0.6745 if np.any(residuals) else 1e-6
            c = max(1.345 * sigma, 1e-6)
            abs_r = np.abs(residuals)
            robust_weights = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-12))
            if np.max(np.abs(x_new - x)) < 1e-6: x = x_new; break
            x = x_new
        
        nuisances = np.zeros((n_obs, self.cfg.nuisance_dim), dtype=float)
        nuisances[:, :n_params] = x[:n_obs * n_params].reshape((n_obs, n_params))
        R_map = np.zeros(tile_shape, dtype=float); R_map[master_mask] = x[n_obs * n_params:]
        R_lf = self._low_frequency_calibration_map(R_map, master_mask)
        R_mf = np.zeros_like(R_lf); R_map = R_lf + self.cfg.calibration_mf_alpha * R_mf
        
        # SIAC Loop
        prev_rms, stagnation_count = float("inf"), 0
        final_iter = 0
        for siac_iter in range(self.cfg.max_siac_iter):
            final_iter = siac_iter
            fused_z, fused_mask = self._fuse_for_calibration(observations, nuisances, R_map, tile_shape, global_shape, basis_family=basis_family)
            if not np.any(fused_mask): break
            R_lf_new, R_mf_new, mf_gate = self._estimate_reference_components(observations, fused_z, fused_mask, nuisances, tile_shape, master_mask, basis_family=basis_family)
            R_lf = (1.0 - self.cfg.calibration_lf_update_blend) * R_lf + self.cfg.calibration_lf_update_blend * R_lf_new
            R_mf = (1.0 - self.cfg.calibration_mf_update_blend) * R_mf + self.cfg.calibration_mf_update_blend * (mf_gate * R_mf_new)
            R_map_new = R_lf + self.cfg.calibration_mf_alpha * R_mf
            ref_delta = float(np.max(np.abs(R_map_new[master_mask] - R_map[master_mask])))
            R_map = R_map_new
            nuisances = self._refine_nuisances(observations, fused_z, fused_mask, R_map, tile_shape, nuisances, basis_family=basis_family)
            if ref_delta < self.cfg.siac_convergence_tol: break
            relative_change = abs(ref_delta - prev_rms) / max(prev_rms, 1e-12)
            if relative_change < 0.01:
                stagnation_count += 1
                if stagnation_count >= 5: break
            else: stagnation_count = 0
            prev_rms = ref_delta

        # Optional instrumentation for runtime diagnostics.
        if final_iter > 0 and os.environ.get("AR_STITCH_LOG_SIAC", "0") == "1":
            import sys
            print(
                f"SIAC converged at iter {final_iter+1}/{self.cfg.max_siac_iter}, "
                f"last delta={ref_delta:.2e}, redundancy={redundancy:.2f}, "
                f"sigma={self.cfg.sigma_filter:.3f}",
                file=sys.stderr,
            )

        pose_shifts = self._estimate_pose_shifts(observations, nuisances, R_map, fused_z, fused_mask, tile_shape, global_shape, basis_family=basis_family)
        if np.any(pose_shifts):
            fused_z, fused_mask = self._fuse_for_calibration(observations, nuisances, R_map, tile_shape, global_shape, basis_family=basis_family, pose_shifts=pose_shifts)
            nuisances = self._refine_nuisances(observations, fused_z, fused_mask, R_map, tile_shape, nuisances, basis_family=basis_family)

        sum_z, count, support = np.zeros(global_shape), np.zeros(global_shape), np.zeros(global_shape, dtype=bool)
        for i, obs in enumerate(observations):
            yy, xx = np.indices(tile_shape, dtype=float)
            y_norm, x_norm = 2.0 * yy / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx / max(tile_shape[1]-1, 1)-1.0
            model = self._nuisance_model(nuisances[i], y_norm, x_norm, basis_family=basis_family)
            z_corr = obs.z - model - R_map
            if pose_shifts is not None: z_corr = self._apply_pose_shift(z_corr, pose_shifts[i])
            w = self._smooth_feather_weights(self._get_eroded_mask(obs.valid_mask), feather_width=self.cfg.feather_width)
            top, left = int(round(obs.center_xy[1] - (tile_shape[0]-1)/2.0)), int(round(obs.center_xy[0] - (tile_shape[1]-1)/2.0))
            gy_s, gy_e = max(0, top), min(global_shape[0], top + tile_shape[0])
            gx_s, gx_e = max(0, left), min(global_shape[1], left + tile_shape[1])
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            if gy_e > gy_s and gx_e > gx_s:
                local_mask = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
                sum_z[gy_s:gy_e, gx_s:gx_e][local_mask] += z_corr[ly_s:ly_e, lx_s:lx_e][local_mask] * w[ly_s:ly_e, lx_s:lx_e][local_mask]
                count[gy_s:gy_e, gx_s:gx_e][local_mask] += w[ly_s:ly_e, lx_s:lx_e][local_mask]
                support[gy_s:gy_e, gx_s:gx_e][local_mask] = True

        valid_mask = count > 0
        z_final = np.full(global_shape, np.nan); z_final[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        R_map_final = np.full(tile_shape, np.nan); R_map_final[master_mask] = R_map[master_mask]
        return ReconstructionSurface(
            z=z_final, valid_mask=valid_mask, source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support, metadata={"method": "scs_adaptive", "instrument_calibration": R_map_final},
        )

    def _low_order_terms(self, y_norm, x_norm, basis_family) -> np.ndarray:
        if basis_family == "legendre":
            return np.stack([np.ones_like(y_norm), y_norm, x_norm, 0.5*(3.0*y_norm**2-1.0), 0.5*(3.0*x_norm**2-1.0), y_norm*x_norm], axis=-1)
        return np.stack([np.ones_like(y_norm), y_norm, x_norm, x_norm**2+y_norm**2, x_norm**2-y_norm**2, 2.0*x_norm*y_norm], axis=-1)

    def _get_eroded_mask(self, valid_mask: np.ndarray) -> np.ndarray:
        if self.cfg.edge_erosion_px <= 0: return valid_mask.copy()
        return ndimage.binary_erosion(valid_mask, structure=np.ones((3, 3), dtype=bool), iterations=self.cfg.edge_erosion_px)

    def _nuisance_model(self, coeffs, y_norm, x_norm, basis_family="radial") -> np.ndarray:
        p = np.zeros(self.cfg.nuisance_dim); p[:min(len(coeffs), self.cfg.nuisance_dim)] = coeffs[:self.cfg.nuisance_dim]
        if basis_family == "legendre":
            return p[0] + p[1]*y_norm + p[2]*x_norm + p[3]*(0.5*(3.0*y_norm**2-1.0)) + p[4]*(0.5*(3.0*x_norm**2-1.0)) + p[5]*(y_norm*x_norm)
        return p[0] + p[1]*y_norm + p[2]*x_norm + p[3]*(x_norm**2+y_norm**2) + p[4]*(x_norm**2-y_norm**2) + p[5]*(2.0*x_norm*y_norm)

    def _fuse_for_calibration(self, observations, nuisances, R_map, tile_shape, global_shape, basis_family="radial", pose_shifts=None):
        sum_z, count = np.zeros(global_shape), np.zeros(global_shape)
        for i, obs in enumerate(observations):
            yy, xx = np.indices(tile_shape, dtype=float)
            y_norm, x_norm = 2.0 * yy / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx / max(tile_shape[1]-1, 1)-1.0
            z_corr = obs.z - self._nuisance_model(nuisances[i], y_norm, x_norm, basis_family=basis_family) - R_map
            if pose_shifts is not None: z_corr = self._apply_pose_shift(z_corr, pose_shifts[i])
            w = self._smooth_feather_weights(self._get_eroded_mask(obs.valid_mask), feather_width=self.cfg.solve_feather_width)
            top, left = int(round(obs.center_xy[1] - (tile_shape[0]-1)/2.0)), int(round(obs.center_xy[0] - (tile_shape[1]-1)/2.0))
            gy_s, gy_e = max(0, top), min(global_shape[0], top + tile_shape[0])
            gx_s, gx_e = max(0, left), min(global_shape[1], left + tile_shape[1])
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            if gy_e > gy_s and gx_e > gx_s:
                m = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
                sum_z[gy_s:gy_e, gx_s:gx_e][m] += z_corr[ly_s:ly_e, lx_s:lx_e][m] * w[ly_s:ly_e, lx_s:lx_e][m]
                count[gy_s:gy_e, gx_s:gx_e][m] += w[ly_s:ly_e, lx_s:lx_e][m]
        valid_mask = count > 0; fused_z = np.full(global_shape, np.nan); fused_z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        return fused_z, valid_mask

    def _refine_nuisances(self, observations, fused_z, fused_mask, R_map, tile_shape, nuisances, basis_family="radial"):
        new_nuisances = nuisances.copy()
        for i, obs in enumerate(observations):
            yy_full, xx_full = np.indices(tile_shape, dtype=float)
            y_norm_f, x_norm_f = 2.0 * yy_full / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx_full / max(tile_shape[1]-1, 1)-1.0
            top, left = int(round(obs.center_xy[1] - (tile_shape[0]-1)/2.0)), int(round(obs.center_xy[0] - (tile_shape[1]-1)/2.0))
            yy, xx = np.where(obs.valid_mask); gy, gx = yy + top, xx + left
            valid_g = (gy >= 0) & (gy < fused_z.shape[0]) & (gx >= 0) & (gx < fused_z.shape[1]) & fused_mask[gy, gx]
            if not np.any(valid_g): continue
            yy, xx, gy, gx = yy[valid_g], xx[valid_g], gy[valid_g], gx[valid_g]
            target = obs.z[yy, xx] - R_map[yy, xx] - fused_z[gy, gx]
            A_nuis = self._low_order_terms(y_norm_f[yy, xx], x_norm_f[yy, xx], basis_family)
            A_reg = np.zeros((3, self.cfg.nuisance_dim)); A_reg[0,3] = A_reg[1,4] = A_reg[2,5] = self.cfg.nuisance_reg_lambda
            coeff, *_ = np.linalg.lstsq(np.vstack([A_nuis, A_reg]), np.concatenate([target, np.zeros(3)]), rcond=None)
            new_nuisances[i,:3] = coeff[:3]; new_nuisances[i,3:] = (1.0 - self.cfg.nuisance_quadratic_damping)*nuisances[i,3:] + self.cfg.nuisance_quadratic_damping*coeff[3:]
        return new_nuisances

    def _estimate_reference_components(self, observations, fused_z, fused_mask, nuisances, tile_shape, master_mask, basis_family="radial"):
        sum_r, sum_w, sum_mf, sum_mf_w, n_contributing = np.zeros(tile_shape), np.zeros(tile_shape), np.zeros(tile_shape), np.zeros(tile_shape), np.zeros(tile_shape)
        for i, obs in enumerate(observations):
            yy_f, xx_f = np.indices(tile_shape, dtype=float)
            y_norm_f, x_norm_f = 2.0 * yy_f / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx_f / max(tile_shape[1]-1, 1)-1.0
            model = self._nuisance_model(nuisances[i], y_norm_f, x_norm_f, basis_family=basis_family)
            top, left = int(round(obs.center_xy[1] - (tile_shape[0]-1)/2.0)), int(round(obs.center_xy[0] - (tile_shape[1]-1)/2.0))
            yy, xx = np.where(obs.valid_mask); gy, gx = yy + top, xx + left
            valid_g = (gy >= 0) & (gy < fused_z.shape[0]) & (gx >= 0) & (gx < fused_z.shape[1]) & fused_mask[gy, gx]
            if not np.any(valid_g): continue
            yy, xx, gy, gx = yy[valid_g], xx[valid_g], gy[valid_g], gx[valid_g]
            residual = obs.z[yy, xx] - model[yy, xx] - fused_z[gy, gx]
            med = float(np.median(residual)); mad = float(np.median(np.abs(residual - med)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(residual)), 1e-6)
            c = max(6.5 * sigma, 1e-6); u = (residual - med) / c
            w = np.zeros_like(u); inside = np.abs(u) < 1.0; w[inside] = (1.0 - u[inside]**2)**2
            sum_r[yy, xx] += w * residual; sum_w[yy, xx] += w; n_contributing[yy, xx] += (w > 0.0).astype(float)
            mf_res = residual
            if getattr(self, "_calibration_mf_detrend_loworder", False): mf_res = self._remove_detector_low_order(mf_res, yy, xx, tile_shape, basis_family=basis_family)
            residual_im = np.zeros(tile_shape); residual_im[yy, xx] = mf_res
            mf_hi = ndimage.gaussian_filter(residual_im, sigma=self.cfg.calibration_bp_sigma)
            mf_lo = ndimage.gaussian_filter(residual_im, sigma=self.cfg.calibration_mf_lo_sigma)
            sum_mf[yy, xx] += w * (mf_hi[yy, xx] - mf_lo[yy, xx]); sum_mf_w[yy, xx] += w
        ref_raw = np.zeros(tile_shape); valid = sum_w > 0; ref_raw[valid] = sum_r[valid] / sum_w[valid]
        ref_lf = self._low_frequency_calibration_map(ref_raw, valid)
        ref_mf = np.zeros(tile_shape); mf_v = sum_mf_w > 0; ref_mf[mf_v] = sum_mf[mf_v] / sum_mf_w[mf_v]
        ref_mf = self._project_degenerate_modes(ref_mf, mf_v, basis_family=basis_family)
        gate = np.clip((n_contributing - (self.cfg.calibration_mf_min_obs - 1.0)), 0.0, 1.0)
        gate = ndimage.gaussian_filter(gate, sigma=0.5); gate = np.clip(gate, 0.0, 1.0); gate[~valid] = 0.0
        return ref_lf, ref_mf, gate

    def _project_degenerate_modes(self, data: np.ndarray, mask: np.ndarray, basis_family: str = "radial") -> np.ndarray:
        """Remove low-order modes from calibration map to prevent crosstalk."""
        result = np.zeros_like(data, dtype=float)
        if not np.any(mask): return result
        yy, xx = np.indices(data.shape, dtype=float)
        y_n = 2.0 * yy[mask] / max(data.shape[0]-1, 1)-1.0
        x_n = 2.0 * xx[mask] / max(data.shape[1]-1, 1)-1.0
        A = self._low_order_terms(y_n, x_n, basis_family)
        coeff, *_ = np.linalg.lstsq(A, data[mask], rcond=None)
        result[mask] = data[mask] - (A @ coeff); return result

    def _smooth_feather_weights(self, valid_mask: np.ndarray, feather_width: float = 0.04) -> np.ndarray:
        weights = np.zeros(valid_mask.shape, dtype=float)
        if not np.any(valid_mask): return weights
        dist = ndimage.distance_transform_edt(valid_mask); max_d = np.max(dist)
        if max_d <= 0: weights[valid_mask] = 1.0; return weights
        f_dist = feather_width * max_d; in_f = valid_mask & (dist <= f_dist)
        if np.any(in_f): weights[in_f] = 0.5 * (1.0 - np.cos(np.pi * dist[in_f] / max(f_dist, 1.0)))
        weights[valid_mask & (dist > f_dist)] = 1.0; return weights

    def _remove_detector_low_order(self, values, yy, xx, tile_shape, basis_family="radial"):
        if values.size == 0: return values
        y_n, x_n = 2.0 * yy / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx / max(tile_shape[1]-1, 1)-1.0
        design = self._low_order_terms(y_n, x_n, basis_family)
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coeff

    def _low_frequency_calibration_map(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Low-frequency calibration map - NON-NORMALIZED smoothing (restored)."""
        if not np.any(mask): return np.zeros_like(data)
        result = np.zeros_like(data); ref_f = np.where(mask, data, 0.0)
        ref_s = ndimage.gaussian_filter(ref_f, sigma=self.cfg.sigma_filter)
        result[mask] = ref_s[mask]; return result

    def _estimate_pose_shifts(self, observations, nuisances, R_map, fused_z, fused_mask, tile_shape, global_shape, basis_family="radial"):
        shifts = np.zeros((len(observations), 2), dtype=float)
        if not np.any(fused_mask): return shifts
        yy, xx = np.indices(tile_shape, dtype=float)
        y_n, x_n = 2.0 * yy / max(tile_shape[0]-1, 1)-1.0, 2.0 * xx / max(tile_shape[1]-1, 1)-1.0
        for i, obs in enumerate(observations):
            z_corr = obs.z - self._nuisance_model(nuisances[i], y_n, x_n, basis_family=basis_family) - R_map
            top, left = int(round(obs.center_xy[1] - (tile_shape[0]-1)/2.0)), int(round(obs.center_xy[0] - (tile_shape[1]-1)/2.0))
            gy_s, gy_e = max(0, top), min(global_shape[0], top + tile_shape[0])
            gx_s, gx_e = max(0, left), min(global_shape[1], left + tile_shape[1])
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)
            if gy_e <= gy_s or gx_e <= gx_s: continue
            f_v, f_m = fused_z[gy_s:gy_e, gx_s:gx_e], fused_mask[gy_s:gy_e, gx_s:gx_e]
            loc_m = self._get_eroded_mask(obs.valid_mask)[ly_s:ly_e, lx_s:lx_e] & f_m
            if not np.any(loc_m): continue
            best_shift, best_score = np.zeros(2), np.inf
            for dy in self.cfg.pose_shift_steps:
                for dx in self.cfg.pose_shift_steps:
                    s_z = self._apply_pose_shift(z_corr, (dy, dx))[ly_s:ly_e, lx_s:lx_e]
                    score = float(np.median(np.abs(s_z[loc_m] - f_v[loc_m])))
                    if score < best_score: best_score, best_shift[:] = score, (dx, dy)
            shifts[i] = best_shift
        return shifts

    def _apply_pose_shift(self, tile: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
        dx, dy = shift_xy
        if abs(dx) < 1e-12 and abs(dy) < 1e-12: return tile
        return ndimage.shift(tile, shift=(dy, dx), order=3, mode="nearest", prefilter=True)
