"""SIAC with Joint Pose Optimization - Simplified version."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.fft import fft2, ifft2, fftshift
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

EDGE_EROSION_PX = 2
FEATHER_WIDTH = 0.20
SIGMA_FILTER = 0.7


def phase_correlation_subpixel(ref, mov, mask_ref=None, mask_mov=None):
    """Simple phase correlation."""
    ref_work = np.nan_to_num(ref, nan=0.0)
    mov_work = np.nan_to_num(mov, nan=0.0)
    
    if mask_ref is not None:
        ref_work = ref_work * mask_ref
    if mask_mov is not None:
        mov_work = mov_work * mask_mov
    
    ref_work = (ref_work - np.mean(ref_work)) / (np.std(ref_work) + 1e-10)
    mov_work = (mov_work - np.mean(mov_work)) / (np.std(mov_work) + 1e-10)
    
    F_ref = fft2(ref_work)
    F_mov = fft2(mov_work)
    
    R = F_ref * np.conj(F_mov)
    R /= np.abs(R) + 1e-10
    
    correlation = np.real(ifft2(R))
    correlation = fftshift(correlation)
    
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_value = correlation[peak_idx]
    
    center = np.array(correlation.shape) // 2
    shift = np.array(peak_idx) - center
    
    return float(shift[0]), float(shift[1]), float(peak_value)


class JointPoseOptimizerSimple:
    """Simplified joint pose optimization."""
    
    def __init__(self, observations, config):
        self.observations = observations
        self.config = config
        self.n_obs = len(observations)
        self.global_shape = observations[0].global_shape
        self.tile_shape = observations[0].tile_shape
        
    def optimize_poses(self, max_iter=3):
        """Returns pose corrections [n_obs x 2] (dy, dx)."""
        print(f"  Optimizing {self.n_obs} poses...", flush=True)
        
        corrections = np.zeros((self.n_obs, 2), dtype=float)
        
        # Simple pairwise registration in a chain
        for i in range(self.n_obs - 1):
            obs_i = self.observations[i]
            obs_j = self.observations[i + 1]
            
            overlap = self._extract_overlap(obs_i, obs_j)
            if overlap is not None:
                ref, mov, mask = overlap
                dy, dx, corr = phase_correlation_subpixel(ref, mov, mask, mask)
                corrections[i + 1] = corrections[i] + np.array([dy, dx])
                print(f"    Pair {i}-{i+1}: dy={dy:.3f}, dx={dx:.3f}, corr={corr:.3f}", flush=True)
        
        corrections -= corrections.mean(axis=0)
        print(f"  Pose corrections: max={np.max(np.abs(corrections)):.4f}", flush=True)
        
        return corrections
    
    def _extract_overlap(self, obs_i, obs_j):
        """Extract overlapping region."""
        rows, cols = self.tile_shape
        
        cy_i, cx_i = obs_i.center_xy[1], obs_i.center_xy[0]
        cy_j, cx_j = obs_j.center_xy[1], obs_j.center_xy[0]
        
        top_i = int(round(cy_i - (rows - 1) / 2.0))
        left_i = int(round(cx_i - (cols - 1) / 2.0))
        top_j = int(round(cy_j - (rows - 1) / 2.0))
        left_j = int(round(cx_j - (cols - 1) / 2.0))
        
        g_top = max(top_i, top_j)
        g_left = max(left_i, left_j)
        g_bottom = min(top_i + rows, top_j + rows)
        g_right = min(left_i + cols, left_j + cols)
        
        if g_bottom <= g_top or g_right <= g_left:
            return None
        
        li_top, li_left = g_top - top_i, g_left - left_i
        li_bottom, li_right = g_bottom - top_i, g_right - left_i
        
        lj_top, lj_left = g_top - top_j, g_left - left_j
        lj_bottom, lj_right = g_bottom - top_j, g_right - left_j
        
        ref_patch = obs_i.z[li_top:li_bottom, li_left:li_right]
        mov_patch = obs_j.z[lj_top:lj_bottom, lj_left:lj_right]
        mask_patch = obs_i.valid_mask[li_top:li_bottom, li_left:li_right] & \
                     obs_j.valid_mask[lj_top:lj_bottom, lj_left:lj_right]
        
        if np.sum(mask_patch) < 100:
            return None
        
        return ref_patch, mov_patch, mask_patch


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

        # Skip pose optimization for now - use identity corrections
        pose_corrections = np.zeros((len(observations), 2), dtype=float)
        registered_observations = observations
        
        tile_shape = observations[0].tile_shape
        global_shape = observations[0].global_shape
        reference_map = np.zeros(tile_shape, dtype=float)
        nuisances = self._solve_global_alignment(registered_observations, reference_map)

        max_outer_iter = 6
        for iteration in range(max_outer_iter):
            fused_z, fused_mask, _ = self._fuse_observations(
                observations=registered_observations,
                nuisances=nuisances,
                reference_map=reference_map,
            )
            if not np.any(fused_mask):
                break

            estimated_reference = self._estimate_reference_map(
                observations=registered_observations,
                fused_z=fused_z,
                fused_mask=fused_mask,
                nuisances=nuisances,
            )
            estimated_nuisances = self._solve_global_alignment(
                observations=registered_observations,
                reference_map=estimated_reference,
            )

            reference_map = 0.5 * reference_map + 0.5 * estimated_reference
            nuisances = 0.5 * nuisances + 0.5 * estimated_nuisances

        z, valid_mask, support = self._fuse_observations(
            observations=registered_observations,
            nuisances=nuisances,
            reference_map=reference_map,
        )

        ref_rms = float(np.sqrt(np.mean(reference_map[observations[0].valid_mask] ** 2))) if np.any(observations[0].valid_mask) else 0.0
        
        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={
                "method": "siac_with_registration",
                "reference_map_rms": ref_rms,
                "pose_corrections": pose_corrections,
                "pose_correction_rms": float(np.sqrt(np.mean(pose_corrections**2))),
                "instrument_calibration": reference_map,
            },
        )
    
    def _apply_pose_corrections(self, observations, corrections):
        """Apply pose corrections by interpolation."""
        registered = []
        
        for i, obs in enumerate(observations):
            dy, dx = corrections[i]
            
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            coords = np.array([yy - dy, xx - dx])
            
            z_corrected = map_coordinates(
                obs.z, coords, order=3, mode='constant', cval=np.nan
            )
            mask_corrected = map_coordinates(
                obs.valid_mask.astype(float), coords, order=0, mode='constant', cval=0
            ) > 0.5
            
            registered.append(SubApertureObservation(
                observation_id=obs.observation_id,
                z=z_corrected,
                valid_mask=mask_corrected,
                center_xy=obs.center_xy,
                tile_shape=obs.tile_shape,
                global_shape=obs.global_shape,
            ))
        
        return tuple(registered)

    def _fuse_observations(
        self,
        observations: tuple[SubApertureObservation, ...],
        nuisances: np.ndarray,
        reference_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        global_shape = observations[0].global_shape
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0
            
            working_mask = self._get_eroded_mask(obs.valid_mask)
            blend_weights = self._smooth_feather_weights(working_mask)

            p, tip, tilt, _focus = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm
            z_corr = obs.z - model - reference_map

            cx, cy = obs.center_xy
            top = int(round(cy - (rows - 1) / 2.0))
            left = int(round(cx - (cols - 1) / 2.0))

            gy_s, gy_e = max(0, top), min(global_shape[0], top + rows)
            gx_s, gx_e = max(0, left), min(global_shape[1], left + cols)

            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)

            if gy_e <= gy_s or gx_e <= gx_s:
                continue

            local_z = z_corr[ly_s:ly_e, lx_s:lx_e]
            local_mask_orig = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
            local_weights = blend_weights[ly_s:ly_e, lx_s:lx_e]

            sum_view = sum_z[gy_s:gy_e, gx_s:gx_e]
            count_view = count[gy_s:gy_e, gx_s:gx_e]
            support_view = support[gy_s:gy_e, gx_s:gx_e]

            weighted_local = local_z * local_weights
            sum_view[local_mask_orig] += weighted_local[local_mask_orig]
            count_view[local_mask_orig] += local_weights[local_mask_orig]
            support_view[local_mask_orig] = True

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        return z, valid_mask, support

    def _estimate_reference_map(
        self,
        observations: tuple[SubApertureObservation, ...],
        fused_z: np.ndarray,
        fused_mask: np.ndarray,
        nuisances: np.ndarray,
    ) -> np.ndarray:
        tile_shape = observations[0].tile_shape
        sum_r = np.zeros(tile_shape, dtype=float)
        sum_w = np.zeros(tile_shape, dtype=float)

        samples = []
        residual_bank = []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            yy_full, xx_full = np.indices(obs.tile_shape, dtype=float)
            y_norm_full = 2.0 * yy_full / max(rows - 1, 1) - 1.0
            x_norm_full = 2.0 * xx_full / max(cols - 1, 1) - 1.0

            p, tip, tilt, _focus = nuisances[i]
            model = p + tip * y_norm_full + tilt * x_norm_full

            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))

            yy, xx = np.where(obs.valid_mask)
            if yy.size == 0:
                continue

            gy = yy + top
            gx = xx + left
            valid_global = (
                (gy >= 0)
                & (gy < fused_z.shape[0])
                & (gx >= 0)
                & (gx < fused_z.shape[1])
                & fused_mask[gy, gx]
            )
            if not np.any(valid_global):
                continue

            yy = yy[valid_global]
            xx = xx[valid_global]
            gy = gy[valid_global]
            gx = gx[valid_global]

            residual = obs.z[yy, xx] - model[yy, xx] - fused_z[gy, gx]
            residual_bank.append(residual)
            samples.append((yy, xx, residual))

        if not residual_bank:
            return np.zeros(tile_shape, dtype=float)

        all_residuals = np.concatenate(residual_bank)
        median = float(np.median(all_residuals))
        mad = float(np.median(np.abs(all_residuals - median)))
        sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(all_residuals)), 1e-6)
        c = max(4.685 * sigma, 1e-6)

        for yy, xx, residual in samples:
            u = (residual - median) / c
            weights = np.zeros_like(u, dtype=float)
            inside = np.abs(u) < 1.0
            weights[inside] = (1.0 - u[inside] ** 2) ** 2

            sum_r[yy, xx] += weights * residual
            sum_w[yy, xx] += weights

        reference_map = np.zeros(tile_shape, dtype=float)
        valid = sum_w > 0
        reference_map[valid] = sum_r[valid] / sum_w[valid]
        
        sigma_filter = SIGMA_FILTER
        ref_filled = np.where(valid, reference_map, 0.0)
        ref_smoothed = ndimage.gaussian_filter(ref_filled, sigma=sigma_filter)
        
        reference_map[valid] = ref_smoothed[valid]

        return self._remove_degenerate_modes(reference_map, valid)

    def _solve_global_alignment(
        self,
        observations: tuple[SubApertureObservation, ...],
        reference_map: np.ndarray,
    ) -> np.ndarray:
        n_obs = len(observations)
        n_params = 3
        if n_obs <= 1:
            return np.zeros((n_obs, 4))

        global_shape = observations[0].global_shape
        all_obs_indices = []
        all_flat_indices = []
        all_z = []
        all_xn = []
        all_yn = []
        all_w = []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))

            yy, xx = np.where(obs.valid_mask)
            if yy.size == 0:
                continue
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])

            yy, xx = yy[valid_global], xx[valid_global]
            gy, gx = gy[valid_global], gx[valid_global]
            if yy.size == 0:
                continue
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0

            working_mask = self._get_eroded_mask(obs.valid_mask)
            weight_map = self._smooth_feather_weights(working_mask)
            all_obs_indices.append(np.full(len(yy), i, dtype=int))
            all_flat_indices.append(gy * global_shape[1] + gx)
            all_z.append(obs.z[yy, xx] - reference_map[yy, xx])
            all_xn.append(x_norm)
            all_yn.append(y_norm)
            all_w.append(weight_map[yy, xx])

        if not all_obs_indices:
            return np.zeros((n_obs, 4))

        obs_idx = np.concatenate(all_obs_indices)
        flat_idx = np.concatenate(all_flat_indices)
        z_vals = np.concatenate(all_z)
        xn_vals = np.concatenate(all_xn)
        yn_vals = np.concatenate(all_yn)
        w_vals = np.concatenate(all_w)

        sort_order = np.argsort(flat_idx)
        obs_idx = obs_idx[sort_order]
        flat_idx = flat_idx[sort_order]
        z_vals = z_vals[sort_order]
        xn_vals = xn_vals[sort_order]
        yn_vals = yn_vals[sort_order]
        w_vals = w_vals[sort_order]

        diff = np.diff(flat_idx)
        boundaries = np.where(diff > 0)[0] + 1
        boundaries = np.concatenate(([0], boundaries, [len(flat_idx)]))

        rows_a = []
        cols_a = []
        data_a = []
        b = []
        base_weights = []
        row_count = 0
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            if e - s < 2:
                continue

            for a in range(s, e - 1):
                ref_o = obs_idx[a]
                ref_z = z_vals[a]
                ref_xn = xn_vals[a]
                ref_yn = yn_vals[a]
                ref_w = w_vals[a]

                for j in range(a + 1, e):
                    oth_o = obs_idx[j]
                    oth_z = z_vals[j]
                    oth_xn = xn_vals[j]
                    oth_yn = yn_vals[j]
                    oth_w = w_vals[j]

                    rows_a.extend([row_count] * 3)
                    cols_a.extend([ref_o * n_params + k for k in range(3)])
                    data_a.extend([1.0, ref_yn, ref_xn])

                    rows_a.extend([row_count] * 3)
                    cols_a.extend([oth_o * n_params + k for k in range(3)])
                    data_a.extend([-1.0, -oth_yn, -oth_xn])

                    b.append(ref_z - oth_z)
                    base_weights.append(float(np.sqrt(max(ref_w * oth_w, 1e-8))))
                    row_count += 1

        if not b:
            return np.zeros((n_obs, 4))

        A_raw = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params))
        b_raw = np.asarray(b, dtype=float)
        base_weights_np = np.asarray(base_weights, dtype=float)

        c_data = []
        c_rows = []
        c_cols = []
        for k in range(n_params):
            for i in range(n_obs):
                c_data.append(1.0)
                c_rows.append(k)
                c_cols.append(i * n_params + k)
        constraint = sp.csr_matrix((c_data, (c_rows, c_cols)), shape=(n_params, n_obs * n_params))

        lambda_reg = 1e-4
        robust_weights = np.ones_like(b_raw)
        x = np.zeros(n_obs * n_params, dtype=float)

        for _ in range(8):
            row_weights = np.sqrt(np.clip(base_weights_np * robust_weights, 1e-8, None))
            W = sp.diags(row_weights)
            A_w = W @ A_raw
            b_w = row_weights * b_raw

            A_aug = sp.vstack([A_w, constraint, lambda_reg * sp.eye(n_obs * n_params)])
            b_aug = np.concatenate([b_w, np.zeros(n_params), np.zeros(n_obs * n_params)])

            x_new, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-10, btol=1e-10)

            residuals = A_raw.dot(x_new[: n_obs * n_params]) - b_raw
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(residuals)), 1e-6)
            c = max(1.345 * sigma, 1e-6)

            abs_r = np.abs(residuals)
            robust_weights = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-12))

            if np.max(np.abs(x_new - x)) < 1e-8:
                x = x_new
                break
            x = x_new

        result = np.zeros((n_obs, 4), dtype=float)
        result[:, :3] = x.reshape((n_obs, n_params))
        return result

    def _get_eroded_mask(self, valid_mask: np.ndarray) -> np.ndarray:
        if EDGE_EROSION_PX <= 0:
            return valid_mask.copy()
        structure = np.ones((3, 3), dtype=bool)
        eroded = ndimage.binary_erosion(valid_mask, structure=structure, iterations=EDGE_EROSION_PX)
        return eroded

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

    def _remove_degenerate_modes(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(data, dtype=float)
        if not np.any(mask):
            return result

        yy, xx = np.indices(data.shape, dtype=float)
        y_norm = 2.0 * yy[mask] / max(data.shape[0] - 1, 1) - 1.0
        x_norm = 2.0 * xx[mask] / max(data.shape[1] - 1, 1) - 1.0
        
        A = np.column_stack([
            np.ones(mask.sum(), dtype=float), 
            y_norm, 
            x_norm,
            x_norm**2 + y_norm**2,
            x_norm**2 - y_norm**2,
            2.0 * x_norm * y_norm
        ])
        coeff, *_ = np.linalg.lstsq(A, data[mask], rcond=None)
        result[mask] = data[mask] - (A @ coeff)
        return result
