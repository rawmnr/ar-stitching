"""Simultaneous calibration and stitching with one bounded pose refinement."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.ndimage import map_coordinates
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation

EDGE_EROSION_PX = 2
FEATHER_WIDTH = 0.20
SIGMA_FILTER = 0.7
MAX_POSE_SHIFT = 0.5
POSE_ACCEPT_THRESHOLD = 0.05
POSE_IRLS_ITERS = 5


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
                observed_support_mask=np.array([], dtype=bool),
            )

        original_support = self._physical_support_mask(observations)
        nuisances, ref_map, fused_z, valid_mask, _ = self._solve_once(observations)

        pose_corrections = self._estimate_pose_corrections(
            observations=observations,
            fused_z=fused_z,
            nuisances=nuisances,
            reference_map=ref_map,
        )

        if np.max(np.abs(pose_corrections)) > POSE_ACCEPT_THRESHOLD:
            registered = self._apply_pose_corrections(observations, pose_corrections)
            nuisances, ref_map, fused_z, valid_mask, _ = self._solve_once(registered)

        ref_mask = observations[0].valid_mask
        ref_rms = float(np.sqrt(np.mean(ref_map[ref_mask] ** 2))) if np.any(ref_mask) else 0.0
        ref_final = np.full(observations[0].tile_shape, np.nan, dtype=float)
        ref_final[ref_mask] = ref_map[ref_mask]

        return ReconstructionSurface(
            z=fused_z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=original_support,
            metadata={
                "method": "optimized_scs_pose_refine",
                "reference_map_rms": ref_rms,
                "pose_corrections": pose_corrections,
                "instrument_calibration": ref_final,
            },
        )

    def _solve_once(
        self,
        observations: tuple[SubApertureObservation, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_obs = len(observations)
        n_params = 3
        tile_shape = observations[0].tile_shape
        global_shape = observations[0].global_shape

        master_mask = np.zeros(tile_shape, dtype=bool)
        for obs in observations:
            master_mask |= obs.valid_mask

        n_r_pixels = int(np.sum(master_mask))
        r_idx_map = np.full(tile_shape, -1, dtype=int)
        r_idx_map[master_mask] = np.arange(n_r_pixels)

        obs_idx_list, flat_idx_list, z_list = [], [], []
        xn_list, yn_list, r_idx_list = [], [], []

        for i, obs in enumerate(observations):
            top = int(round(obs.center_xy[1] - (tile_shape[0] - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (tile_shape[1] - 1) / 2.0))

            yy, xx = np.where(obs.valid_mask)
            gy, gx = yy + top, xx + left
            valid_global = (gy >= 0) & (gy < global_shape[0]) & (gx >= 0) & (gx < global_shape[1])

            yy, xx = yy[valid_global], xx[valid_global]
            gy, gx = gy[valid_global], gx[valid_global]
            if yy.size == 0:
                continue

            y_norm = 2.0 * yy / max(tile_shape[0] - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(tile_shape[1] - 1, 1) - 1.0

            obs_idx_list.append(np.full(len(yy), i, dtype=int))
            flat_idx_list.append(gy * global_shape[1] + gx)
            z_list.append(obs.z[yy, xx])
            xn_list.append(x_norm)
            yn_list.append(y_norm)
            r_idx_list.append(r_idx_map[yy, xx])

        if not obs_idx_list:
            z = np.full(global_shape, np.nan, dtype=float)
            valid_mask = np.zeros(global_shape, dtype=bool)
            support = np.zeros(global_shape, dtype=bool)
            return np.zeros((n_obs, 4), dtype=float), np.zeros(tile_shape, dtype=float), z, valid_mask, support

        obs_idx = np.concatenate(obs_idx_list)
        flat_idx = np.concatenate(flat_idx_list)
        z_vals = np.concatenate(z_list)
        xn_vals = np.concatenate(xn_list)
        yn_vals = np.concatenate(yn_list)
        r_idx_vals = np.concatenate(r_idx_list)

        order = np.argsort(flat_idx)
        obs_idx = obs_idx[order]
        flat_idx = flat_idx[order]
        z_vals = z_vals[order]
        xn_vals = xn_vals[order]
        yn_vals = yn_vals[order]
        r_idx_vals = r_idx_vals[order]

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

                oth_o = obs_idx[j + 1]
                oth_z = z_vals[j + 1]
                oth_xn = xn_vals[j + 1]
                oth_yn = yn_vals[j + 1]
                oth_r = r_idx_vals[j + 1]

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

        A = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(row_count, n_obs * n_params + n_r_pixels))
        b_np = np.asarray(b, dtype=float)

        c_data, c_rows, c_cols = [], [], []
        c_idx = 0
        for i in range(n_obs):
            c_data.append(1.0)
            c_rows.append(c_idx)
            c_cols.append(i * n_params)
        c_idx += 1
        for i in range(n_obs):
            c_data.append(1.0)
            c_rows.append(c_idx)
            c_cols.append(i * n_params + 1)
        c_idx += 1
        for i in range(n_obs):
            c_data.append(1.0)
            c_rows.append(c_idx)
            c_cols.append(i * n_params + 2)
        c_idx += 1

        r_yy, r_xx = np.where(master_mask)
        r_y_norm = 2.0 * r_yy / max(tile_shape[0] - 1, 1) - 1.0
        r_x_norm = 2.0 * r_xx / max(tile_shape[1] - 1, 1) - 1.0
        r_rad2 = r_x_norm**2 + r_y_norm**2

        for idx in range(n_r_pixels):
            col = n_obs * n_params + idx
            c_data.append(1.0)
            c_rows.append(c_idx)
            c_cols.append(col)
            c_data.append(r_y_norm[idx])
            c_rows.append(c_idx + 1)
            c_cols.append(col)
            c_data.append(r_x_norm[idx])
            c_rows.append(c_idx + 2)
            c_cols.append(col)
            c_data.append(r_rad2[idx])
            c_rows.append(c_idx + 3)
            c_cols.append(col)
            c_data.append(r_x_norm[idx]**2 - r_y_norm[idx]**2)
            c_rows.append(c_idx + 4)
            c_cols.append(col)
            c_data.append(2.0 * r_x_norm[idx] * r_y_norm[idx])
            c_rows.append(c_idx + 5)
            c_cols.append(col)
        c_idx += 6

        constraint = sp.csr_matrix((c_data, (c_rows, c_cols)), shape=(c_idx, n_obs * n_params + n_r_pixels))

        lambda_reg = 1e-6
        robust_weights = np.ones(row_count, dtype=float)
        x = np.zeros(n_obs * n_params + n_r_pixels, dtype=float)

        for _ in range(5):
            w_sqrt = np.sqrt(robust_weights)
            A_w = sp.diags(w_sqrt) @ A
            b_w = w_sqrt * b_np

            A_aug = sp.vstack([A_w, constraint, lambda_reg * sp.eye(n_obs * n_params + n_r_pixels)])
            b_aug = np.concatenate([b_w, np.zeros(c_idx), np.zeros(n_obs * n_params + n_r_pixels)])

            x_new, *_ = spla.lsqr(A_aug, b_aug, damp=1e-8, atol=1e-8, btol=1e-8)
            residuals = A.dot(x_new) - b_np
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(residuals)), 1e-6)
            c = max(1.345 * sigma, 1e-6)
            robust_weights = np.where(np.abs(residuals) <= c, 1.0, c / np.maximum(np.abs(residuals), 1e-12))

            if np.max(np.abs(x_new - x)) < 1e-6:
                x = x_new
                break
            x = x_new

        nuisances = x[: n_obs * n_params].reshape((n_obs, n_params))
        r_vals = x[n_obs * n_params :]
        ref_map = np.zeros(tile_shape, dtype=float)
        ref_map[master_mask] = r_vals

        ref_filled = np.where(master_mask, ref_map, 0.0)
        ref_smoothed = ndimage.gaussian_filter(ref_filled, sigma=SIGMA_FILTER)
        ref_map[master_mask] = ref_smoothed[master_mask]

        fused_z, valid_mask, support = self._fuse_observations(observations, nuisances, ref_map)
        return nuisances, ref_map, fused_z, valid_mask, support

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

            p, tip, tilt = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm
            z_corr = obs.z - model - reference_map

            working_mask = self._get_eroded_mask(obs.valid_mask)
            feather_weights = self._smooth_feather_weights(working_mask)

            top = int(round(obs.center_xy[1] - (rows - 1) / 2.0))
            left = int(round(obs.center_xy[0] - (cols - 1) / 2.0))

            gy_s, gy_e = max(0, top), min(global_shape[0], top + rows)
            gx_s, gx_e = max(0, left), min(global_shape[1], left + cols)
            ly_s, lx_s = max(0, -top), max(0, -left)
            ly_e, lx_e = ly_s + (gy_e - gy_s), lx_s + (gx_e - gx_s)

            if gy_e <= gy_s or gx_e <= gx_s:
                continue

            local_z = z_corr[ly_s:ly_e, lx_s:lx_e]
            local_mask_orig = obs.valid_mask[ly_s:ly_e, lx_s:lx_e]
            local_weights = feather_weights[ly_s:ly_e, lx_s:lx_e]

            sum_z[gy_s:gy_e, gx_s:gx_e][local_mask_orig] += local_z[local_mask_orig] * local_weights[local_mask_orig]
            count[gy_s:gy_e, gx_s:gx_e][local_mask_orig] += local_weights[local_mask_orig]
            support[gy_s:gy_e, gx_s:gx_e][local_mask_orig] = True

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        return z, valid_mask, support

    def _estimate_pose_corrections(
        self,
        observations: tuple[SubApertureObservation, ...],
        fused_z: np.ndarray,
        nuisances: np.ndarray,
        reference_map: np.ndarray,
    ) -> np.ndarray:
        n_obs = len(observations)
        corrections = np.zeros((n_obs, 2), dtype=float)
        fused_filled = np.nan_to_num(fused_z, nan=0.0)
        fused_mask = np.isfinite(fused_z).astype(float)

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            working_mask = self._get_eroded_mask(obs.valid_mask)
            yy, xx = np.where(working_mask)
            if yy.size < 20:
                continue

            yy_g, xx_g = self._local_to_global_coords(obs, yy.astype(float), xx.astype(float))
            valid_global = (
                (yy_g >= 2.0)
                & (yy_g < fused_z.shape[0] - 2.0)
                & (xx_g >= 2.0)
                & (xx_g < fused_z.shape[1] - 2.0)
            )
            if not np.any(valid_global):
                continue

            yy = yy[valid_global]
            xx = xx[valid_global]
            yy_g = yy_g[valid_global]
            xx_g = xx_g[valid_global]

            center_valid = self._sample_global_array(fused_mask, yy_g, xx_g, order=1)
            up_valid = self._sample_global_array(fused_mask, yy_g - 1.0, xx_g, order=1)
            down_valid = self._sample_global_array(fused_mask, yy_g + 1.0, xx_g, order=1)
            left_valid = self._sample_global_array(fused_mask, yy_g, xx_g - 1.0, order=1)
            right_valid = self._sample_global_array(fused_mask, yy_g, xx_g + 1.0, order=1)

            valid = (center_valid > 0.75) & (up_valid > 0.75) & (down_valid > 0.75) & (left_valid > 0.75) & (right_valid > 0.75)
            if not np.any(valid):
                continue

            yy = yy[valid]
            xx = xx[valid]
            yy_g = yy_g[valid]
            xx_g = xx_g[valid]

            rows_f, cols_f = obs.tile_shape
            y_norm = 2.0 * yy / max(rows_f - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols_f - 1, 1) - 1.0
            p, tip, tilt = nuisances[i]
            model = p + tip * y_norm + tilt * x_norm

            fused_sample = self._sample_global_array(fused_filled, yy_g, xx_g, order=1)
            residual = obs.z[yy, xx] - model - reference_map[yy, xx] - fused_sample

            gz_y = 0.5 * (
                self._sample_global_array(fused_filled, yy_g + 1.0, xx_g, order=1)
                - self._sample_global_array(fused_filled, yy_g - 1.0, xx_g, order=1)
            )
            gz_x = 0.5 * (
                self._sample_global_array(fused_filled, yy_g, xx_g + 1.0, order=1)
                - self._sample_global_array(fused_filled, yy_g, xx_g - 1.0, order=1)
            )
            grad_mag = np.sqrt(gz_y**2 + gz_x**2)
            valid_grad = np.isfinite(residual) & np.isfinite(gz_y) & np.isfinite(gz_x) & (grad_mag > 0.01)
            if np.count_nonzero(valid_grad) < 10:
                continue

            residual = residual[valid_grad]
            gz_y = gz_y[valid_grad]
            gz_x = gz_x[valid_grad]
            grad_mag = grad_mag[valid_grad]

            delta = np.zeros(2, dtype=float)
            weights = np.maximum(grad_mag, 0.01)
            weights = weights / max(float(np.max(weights)), 1e-12)

            for _ in range(POSE_IRLS_ITERS):
                g = np.column_stack([-gz_y, -gz_x])
                r = residual - g @ delta
                mad = np.median(np.abs(r - np.median(r)))
                sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(r)), 1e-6)
                c = max(1.345 * sigma, 1e-6)
                irls = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
                w = np.sqrt(weights * irls)
                wg = w[:, None] * g
                wr = w * r
                H = wg.T @ wg + 1e-6 * np.eye(2)
                b = wg.T @ wr
                try:
                    if np.linalg.cond(H) > 1e10:
                        break
                    delta_new = np.linalg.solve(H, b)
                except np.linalg.LinAlgError:
                    break
                if np.any(~np.isfinite(delta_new)):
                    break
                if np.max(np.abs(delta_new - delta)) < 1e-4:
                    delta = delta_new
                    break
                delta = delta_new

            corrections[i] = delta

        corrections -= corrections.mean(axis=0)
        max_shift = float(np.max(np.abs(corrections)))
        if max_shift > MAX_POSE_SHIFT:
            corrections *= MAX_POSE_SHIFT / max(max_shift, 1e-12)
        return corrections

    def _apply_pose_corrections(
        self,
        observations: tuple[SubApertureObservation, ...],
        corrections: np.ndarray,
    ) -> tuple[SubApertureObservation, ...]:
        registered = []
        for i, obs in enumerate(observations):
            dy, dx = corrections[i]
            if abs(dy) < 1e-8 and abs(dx) < 1e-8:
                registered.append(obs)
                continue
            registered.append(
                SubApertureObservation(
                    observation_id=obs.observation_id,
                    z=obs.z,
                    valid_mask=obs.valid_mask.copy(),
                    center_xy=(float(obs.center_xy[0] + dx), float(obs.center_xy[1] + dy)),
                    tile_shape=obs.tile_shape,
                    global_shape=obs.global_shape,
                    rotation_deg=obs.rotation_deg,
                    reference_bias=obs.reference_bias,
                    nuisance_terms=dict(obs.nuisance_terms),
                    metadata=dict(obs.metadata),
                )
            )
        return tuple(registered)

    def _physical_support_mask(
        self,
        observations: tuple[SubApertureObservation, ...],
    ) -> np.ndarray:
        support = np.zeros(observations[0].global_shape, dtype=bool)
        for obs in observations:
            rows, cols = obs.tile_shape
            top = int(round(float(obs.center_xy[1]) - (rows - 1) / 2.0))
            left = int(round(float(obs.center_xy[0]) - (cols - 1) / 2.0))
            gy_s = max(0, top)
            gy_e = min(obs.global_shape[0], top + rows)
            gx_s = max(0, left)
            gx_e = min(obs.global_shape[1], left + cols)
            ly_s = max(0, -top)
            lx_s = max(0, -left)
            ly_e = ly_s + (gy_e - gy_s)
            lx_e = lx_s + (gx_e - gx_s)
            if gy_e > gy_s and gx_e > gx_s:
                support[gy_s:gy_e, gx_s:gx_e][obs.valid_mask[ly_s:ly_e, lx_s:lx_e]] = True
        return support

    def _sample_global_array(
        self,
        values: np.ndarray,
        yy_global: np.ndarray,
        xx_global: np.ndarray,
        *,
        order: int = 1,
        cval: float = 0.0,
    ) -> np.ndarray:
        coords = np.array([yy_global.ravel(), xx_global.ravel()])
        sampled = map_coordinates(values, coords, order=order, mode="constant", cval=cval)
        return sampled.reshape(yy_global.shape)

    def _local_to_global_coords(
        self,
        obs: SubApertureObservation,
        yy_local: np.ndarray,
        xx_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = obs.tile_shape
        angle_rad = np.deg2rad(obs.rotation_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        yy_rel = yy_local - (rows - 1) / 2.0
        xx_rel = xx_local - (cols - 1) / 2.0
        xx_global = obs.center_xy[0] + xx_rel * cos_a - yy_rel * sin_a
        yy_global = obs.center_xy[1] + xx_rel * sin_a + yy_rel * cos_a
        return yy_global, xx_global

    def _get_eroded_mask(self, valid_mask: np.ndarray) -> np.ndarray:
        if EDGE_EROSION_PX <= 0:
            return valid_mask.copy()
        structure = np.ones((3, 3), dtype=bool)
        return ndimage.binary_erosion(valid_mask, structure=structure, iterations=EDGE_EROSION_PX)

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
