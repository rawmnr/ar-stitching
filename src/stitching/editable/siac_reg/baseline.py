"""SIAC with Leave-One-Out Pose Estimation.

Uses leave-one-out fused predictions to estimate pose errors without
attenuation bias. Per-observation robust gradient solve with IRLS.
"""
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
GRADIENT_FLOOR = 0.01
POSE_DAMPING_LADDER = [1.0, 0.5, 0.25, 0.1]
MAX_POSE_ITER = 5
MAX_TOTAL_POSE_SHIFT = 0.75
POSE_IMPROVEMENT_RATIO = 0.995


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
        tile_shape = observations[0].tile_shape
        global_shape = observations[0].global_shape
        zero_pose = np.zeros((n_obs, 2), dtype=float)

        base_reference_map, base_nuisances, base_fused_result = self._run_siac_outer_loop(
            observations=observations,
            tile_shape=tile_shape,
            n_outer_iter=6,
        )
        base_objective = self._compute_loo_objective(
            observations,
            base_fused_result,
            base_nuisances,
            base_reference_map,
            global_shape,
            zero_pose,
        )

        pose_corrections = np.zeros((n_obs, 2), dtype=float)

        for reg_iter in range(MAX_POSE_ITER):
            registered_observations = self._apply_pose_corrections(observations, pose_corrections)
            reference_map, nuisances, fused_result = self._run_siac_outer_loop(
                observations=registered_observations,
                tile_shape=tile_shape,
                n_outer_iter=3,
            )
            current_objective = self._compute_loo_objective(
                registered_observations,
                fused_result,
                nuisances,
                reference_map,
                global_shape,
                zero_pose,
            )

            delta_poses, accepted = self._estimate_poses_loo(
                registered_observations,
                fused_result,
                nuisances,
                reference_map,
                global_shape,
            )

            if delta_poses is not None and accepted:
                delta_poses -= delta_poses.mean(axis=0)
                trial_pose_corrections = pose_corrections + delta_poses

                max_total = float(np.max(np.abs(trial_pose_corrections)))
                if max_total > MAX_TOTAL_POSE_SHIFT:
                    scale = MAX_TOTAL_POSE_SHIFT / max(max_total, 1e-12)
                    delta_poses *= scale
                    trial_pose_corrections = pose_corrections + delta_poses

                trial_registered = self._apply_pose_corrections(observations, trial_pose_corrections)
                trial_reference_map, trial_nuisances, trial_fused_result = self._run_siac_outer_loop(
                    observations=trial_registered,
                    tile_shape=tile_shape,
                    n_outer_iter=3,
                )
                trial_objective = self._compute_loo_objective(
                    trial_registered,
                    trial_fused_result,
                    trial_nuisances,
                    trial_reference_map,
                    global_shape,
                    zero_pose,
                )

                if (
                    current_objective is not None
                    and trial_objective is not None
                    and np.isfinite(current_objective)
                    and np.isfinite(trial_objective)
                    and trial_objective < current_objective * POSE_IMPROVEMENT_RATIO
                ):
                    pose_corrections = trial_pose_corrections
                    max_delta = np.max(np.abs(delta_poses))
                    print(
                        f"  Reg iter {reg_iter}: delta = {max_delta:.4f} px "
                        f"(accepted, obj {current_objective:.5f} -> {trial_objective:.5f})",
                        flush=True,
                    )

                    if max_delta < 0.02:
                        break
                else:
                    current_text = 'nan' if current_objective is None else f'{current_objective:.5f}'
                    trial_text = 'nan' if trial_objective is None else f'{trial_objective:.5f}'
                    print(
                        f"  Reg iter {reg_iter}: rejected candidate "
                        f"(obj {current_text} -> {trial_text})",
                        flush=True,
                    )
                    break
            elif delta_poses is not None:
                print(f"  Reg iter {reg_iter}: no improvement, stopping", flush=True)
                break
            else:
                print(f"  Reg iter {reg_iter}: insufficient data", flush=True)
                break

        max_corr = np.max(np.abs(pose_corrections))
        rms_corr = np.sqrt(np.mean(pose_corrections**2))
        print(f"  Final pose corrections: max={max_corr:.3f}, rms={rms_corr:.3f}", flush=True)

        registered_observations = self._apply_pose_corrections(observations, pose_corrections)
        reference_map, nuisances, fused_result = self._run_siac_outer_loop(
            observations=registered_observations,
            tile_shape=tile_shape,
            n_outer_iter=6,
        )
        final_objective = self._compute_loo_objective(
            registered_observations,
            fused_result,
            nuisances,
            reference_map,
            global_shape,
            zero_pose,
        )

        use_registered_solution = (
            final_objective is not None
            and base_objective is not None
            and np.isfinite(final_objective)
            and np.isfinite(base_objective)
            and final_objective < base_objective * POSE_IMPROVEMENT_RATIO
        )

        if not use_registered_solution:
            pose_corrections = zero_pose
            reference_map = base_reference_map
            nuisances = base_nuisances
            fused_result = base_fused_result
            print("  Final selection: fallback to SIAC baseline state", flush=True)
        else:
            print(
                f"  Final selection: kept registered solution "
                f"(obj {base_objective:.5f} -> {final_objective:.5f})",
                flush=True,
            )

        z = fused_result['fused_z']
        valid_mask = fused_result['valid_mask']
        support = self._physical_support_mask(observations)

        ref_rms = float(np.sqrt(np.mean(reference_map[observations[0].valid_mask] ** 2))) if np.any(observations[0].valid_mask) else 0.0

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={
                "method": "siac_iterative_pose",
                "reference_map_rms": ref_rms,
                "pose_corrections": pose_corrections,
                "pose_correction_rms": float(np.sqrt(np.mean(pose_corrections**2))),
                "instrument_calibration": reference_map,
            },
        )

    def _run_siac_outer_loop(
        self,
        observations: tuple[SubApertureObservation, ...],
        tile_shape: tuple[int, int],
        n_outer_iter: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        reference_map = np.zeros(tile_shape, dtype=float)
        nuisances = self._solve_global_alignment(observations, reference_map)

        for _ in range(n_outer_iter):
            fused_z, fused_mask, _ = self._fuse_observations(
                observations=observations,
                nuisances=nuisances,
                reference_map=reference_map,
            )
            if not np.any(fused_mask):
                break

            estimated_reference = self._estimate_reference_map(
                observations=observations,
                fused_z=fused_z,
                fused_mask=fused_mask,
                nuisances=nuisances,
            )
            estimated_nuisances = self._solve_global_alignment(
                observations=observations,
                reference_map=estimated_reference,
            )

            reference_map = 0.5 * reference_map + 0.5 * estimated_reference
            nuisances = 0.5 * nuisances + 0.5 * estimated_nuisances

        fused_result = self._fuse_observations_with_contrib(
            observations,
            nuisances,
            reference_map,
        )
        return reference_map, nuisances, fused_result

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

    def _local_to_global_coords(
        self,
        obs: SubApertureObservation,
        yy_local: np.ndarray,
        xx_local: np.ndarray,
        center_xy: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = obs.tile_shape
        center_x, center_y = obs.center_xy if center_xy is None else center_xy
        angle_rad = np.deg2rad(obs.rotation_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        yy_rel = yy_local - (rows - 1) / 2.0
        xx_rel = xx_local - (cols - 1) / 2.0
        xx_global = center_x + xx_rel * cos_a - yy_rel * sin_a
        yy_global = center_y + xx_rel * sin_a + yy_rel * cos_a
        return yy_global, xx_global

    def _global_to_local_coords(
        self,
        obs: SubApertureObservation,
        yy_global: np.ndarray,
        xx_global: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = obs.tile_shape
        angle_rad = np.deg2rad(obs.rotation_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        dx = xx_global - obs.center_xy[0]
        dy = yy_global - obs.center_xy[1]
        xx_rel = dx * cos_a + dy * sin_a
        yy_rel = -dx * sin_a + dy * cos_a
        xx_local = xx_rel + (cols - 1) / 2.0
        yy_local = yy_rel + (rows - 1) / 2.0
        return yy_local, xx_local

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

    def _project_local_field_to_global(
        self,
        obs: SubApertureObservation,
        local_values: np.ndarray,
        local_weights: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        global_shape = obs.global_shape
        rows, cols = obs.tile_shape
        angle_rad = np.deg2rad(obs.rotation_deg)
        cos_a = abs(float(np.cos(angle_rad)))
        sin_a = abs(float(np.sin(angle_rad)))

        half_h = 0.5 * ((rows - 1) * cos_a + (cols - 1) * sin_a)
        half_w = 0.5 * ((cols - 1) * cos_a + (rows - 1) * sin_a)
        cy = float(obs.center_xy[1])
        cx = float(obs.center_xy[0])

        gy_s = max(0, int(np.floor(cy - half_h - 1.0)))
        gy_e = min(global_shape[0], int(np.ceil(cy + half_h + 1.0)) + 1)
        gx_s = max(0, int(np.floor(cx - half_w - 1.0)))
        gx_e = min(global_shape[1], int(np.ceil(cx + half_w + 1.0)) + 1)

        if gy_e <= gy_s or gx_e <= gx_s:
            empty_shape = (0, 0)
            return np.array([], dtype=int), np.array([], dtype=int), np.zeros(empty_shape), np.zeros(empty_shape), np.zeros(empty_shape, dtype=bool)

        gy, gx = np.indices((gy_e - gy_s, gx_e - gx_s), dtype=float)
        gy += gy_s
        gx += gx_s
        yy_local, xx_local = self._global_to_local_coords(obs, gy, gx)
        coords = np.array([yy_local.ravel(), xx_local.ravel()])

        filled_values = np.where(valid_mask, local_values, 0.0)
        sampled_values = map_coordinates(filled_values, coords, order=1, mode="constant", cval=0.0).reshape(gy.shape)
        sampled_weights = map_coordinates(local_weights, coords, order=1, mode="constant", cval=0.0).reshape(gy.shape)
        sampled_mask = map_coordinates(valid_mask.astype(float), coords, order=1, mode="constant", cval=0.0).reshape(gy.shape) >= 0.5
        sampled_mask &= sampled_weights > 1e-6
        return gy.astype(int), gx.astype(int), sampled_values, sampled_weights, sampled_mask

    def _fuse_observations_with_contrib(
        self,
        observations: tuple[SubApertureObservation, ...],
        nuisances: np.ndarray,
        reference_map: np.ndarray,
    ) -> dict:
        global_shape = observations[0].global_shape

        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        overlap_count = np.zeros(global_shape, dtype=int)
        support = np.zeros(global_shape, dtype=bool)

        contrib_sums = []
        contrib_counts = []

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

            gy, gx, sampled_values, sampled_weights, sampled_mask = self._project_local_field_to_global(
                obs,
                z_corr,
                blend_weights,
                obs.valid_mask,
            )

            contrib_sum = np.zeros(global_shape, dtype=float)
            contrib_count = np.zeros(global_shape, dtype=float)
            if gy.size == 0:
                contrib_sums.append(contrib_sum)
                contrib_counts.append(contrib_count)
                continue

            weighted_values = sampled_values * sampled_weights
            contrib_sum[gy, gx] = np.where(sampled_mask, weighted_values, 0.0)
            contrib_count[gy, gx] = np.where(sampled_mask, sampled_weights, 0.0)

            valid_idx = sampled_mask
            sum_z[gy[valid_idx], gx[valid_idx]] += weighted_values[valid_idx]
            count[gy[valid_idx], gx[valid_idx]] += sampled_weights[valid_idx]
            overlap_count[gy[valid_idx], gx[valid_idx]] += 1
            support[gy[valid_idx], gx[valid_idx]] = True

            contrib_sums.append(contrib_sum)
            contrib_counts.append(contrib_count)

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]

        return {
            'fused_z': z,
            'valid_mask': valid_mask,
            'support': support,
            'sum_z': sum_z,
            'count': count,
            'overlap_count': overlap_count,
            'contrib_sums': contrib_sums,
            'contrib_counts': contrib_counts,
        }

    def _estimate_poses_loo(
        self,
        observations: tuple,
        fused_result: dict,
        nuisances: np.ndarray,
        reference_map: np.ndarray,
        global_shape: tuple,
    ) -> tuple[np.ndarray | None, bool]:
        n_obs = len(observations)

        sum_z = fused_result['sum_z']
        count = fused_result['count']
        overlap_count = fused_result['overlap_count']
        contrib_sums = fused_result['contrib_sums']
        contrib_counts = fused_result['contrib_counts']
        fused_z = np.nan_to_num(fused_result['fused_z'], nan=0.0)
        fused_valid = fused_result['valid_mask'].astype(float)

        eps = 1e-12
        loo_threshold = 0.1

        delta_poses = np.zeros((n_obs, 2), dtype=float)
        debug_info = []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            working_mask = self._get_eroded_mask(obs.valid_mask)
            yy, xx = np.where(working_mask)
            if yy.size == 0:
                continue

            yy_g, xx_g = self._local_to_global_coords(obs, yy.astype(float), xx.astype(float))
            valid_global = (
                (yy_g >= 2.0) & (yy_g < global_shape[0] - 2.0) &
                (xx_g >= 2.0) & (xx_g < global_shape[1] - 2.0)
            )
            if not np.any(valid_global):
                continue

            yy_v = yy[valid_global]
            xx_v = xx[valid_global]
            yy_g = yy_g[valid_global]
            xx_g = xx_g[valid_global]

            loo_sum = self._sample_global_array(sum_z - contrib_sums[i], yy_g, xx_g, order=1)
            loo_w = self._sample_global_array(count - contrib_counts[i], yy_g, xx_g, order=1)
            overlap_local = self._sample_global_array(overlap_count.astype(float), yy_g, xx_g, order=1)

            valid_loo = (loo_w > loo_threshold) & (overlap_local >= 1.5)
            if not np.any(valid_loo):
                continue

            yy_v = yy_v[valid_loo]
            xx_v = xx_v[valid_loo]
            yy_g = yy_g[valid_loo]
            xx_g = xx_g[valid_loo]
            loo_sum = loo_sum[valid_loo]
            loo_w = loo_w[valid_loo]
            pred_loo = loo_sum / np.maximum(loo_w, eps)

            p, tip, tilt, _ = nuisances[i]
            y_norm = 2.0 * yy_v / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx_v / max(cols - 1, 1) - 1.0
            model = p + tip * y_norm + tilt * x_norm
            z_local = obs.z[yy_v, xx_v]
            ref_local = reference_map[yy_v, xx_v]
            residual = z_local - model - ref_local - pred_loo

            center_valid = self._sample_global_array(fused_valid, yy_g, xx_g, order=1)
            up_valid = self._sample_global_array(fused_valid, yy_g - 1.0, xx_g, order=1)
            down_valid = self._sample_global_array(fused_valid, yy_g + 1.0, xx_g, order=1)
            left_valid = self._sample_global_array(fused_valid, yy_g, xx_g - 1.0, order=1)
            right_valid = self._sample_global_array(fused_valid, yy_g, xx_g + 1.0, order=1)

            gz_y = 0.5 * (
                self._sample_global_array(fused_z, yy_g + 1.0, xx_g, order=1)
                - self._sample_global_array(fused_z, yy_g - 1.0, xx_g, order=1)
            )
            gz_x = 0.5 * (
                self._sample_global_array(fused_z, yy_g, xx_g + 1.0, order=1)
                - self._sample_global_array(fused_z, yy_g, xx_g - 1.0, order=1)
            )
            grad_mag = np.sqrt(gz_y ** 2 + gz_x ** 2)

            valid_grad = (
                np.isfinite(residual)
                & np.isfinite(gz_y)
                & np.isfinite(gz_x)
                & (grad_mag > GRADIENT_FLOOR)
                & (center_valid > 0.75)
                & (up_valid > 0.75)
                & (down_valid > 0.75)
                & (left_valid > 0.75)
                & (right_valid > 0.75)
            )
            if not np.any(valid_grad):
                continue

            delta_i = self._solve_per_obs_pose_irls(
                residual[valid_grad],
                gz_y[valid_grad],
                gz_x[valid_grad],
                grad_mag[valid_grad],
            )
            if delta_i is not None:
                delta_poses[i] = delta_i
                debug_info.append((i, np.linalg.norm(delta_i), int(np.sum(valid_grad)), float(np.mean(grad_mag[valid_grad]))))

        if np.all(np.abs(delta_poses) < 1e-6):
            return None, False

        if debug_info:
            mean_delta = np.mean([d[1] for d in debug_info])
            print(f"    LOO debug: obs_contribs={len(debug_info)}, mean_delta={mean_delta:.4f}, mean_grad={np.mean([d[3] for d in debug_info]):.6f}", flush=True)

        delta_poses -= delta_poses.mean(axis=0)
        max_abs = np.max(np.abs(delta_poses))
        if max_abs < 0.01:
            return delta_poses, False

        for damping in POSE_DAMPING_LADDER:
            candidate = delta_poses * damping
            if np.max(np.abs(candidate)) <= 0.5:
                return candidate, True

        return delta_poses * 0.25, True

    def _solve_per_obs_pose_irls(
        self,
        residual: np.ndarray,
        gz_y: np.ndarray,
        gz_x: np.ndarray,
        grad_mag: np.ndarray,
        n_irls_iter: int = 5,
    ) -> np.ndarray | None:
        if len(residual) < 10:
            return None
        
        weights = np.maximum(grad_mag, GRADIENT_FLOOR)
        weights = weights / np.max(weights)
        
        delta = np.zeros(2, dtype=float)
        
        for _ in range(n_irls_iter):
            g = np.column_stack([-gz_y, -gz_x])
            r = residual - g @ delta
            
            mad = np.median(np.abs(r - np.median(r)))
            sigma = mad / 0.6745 if mad > 1e-12 else max(float(np.std(r)), 1e-6)
            c = max(1.345 * sigma, 1e-6)
            
            abs_r = np.abs(r)
            irls_w = np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-12))
            
            combined_w = weights * irls_w
            combined_w = np.sqrt(combined_w)
            
            Wg = combined_w[:, np.newaxis] * g
            Wr = combined_w * r
            
            H = Wg.T @ Wg
            b = Wg.T @ Wr
            
            H += 1e-6 * np.eye(2)
            
            try:
                if np.linalg.cond(H) > 1e10:
                    break
                delta_new = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break
            
            if np.any(~np.isfinite(delta_new)):
                break
            
            delta = delta_new
        
        return delta if np.any(np.abs(delta) > 1e-6) else np.zeros(2)

    def _compute_loo_objective(
        self,
        observations: tuple,
        fused_result: dict,
        nuisances: np.ndarray,
        reference_map: np.ndarray,
        global_shape: tuple,
        pose_delta: np.ndarray,
    ) -> float | None:
        sum_z = fused_result['sum_z']
        count = fused_result['count']
        overlap_count = fused_result['overlap_count']
        contrib_sums = fused_result['contrib_sums']
        contrib_counts = fused_result['contrib_counts']

        eps = 1e-12
        loo_threshold = 0.1
        total_residual_sq = 0.0
        total_count = 0

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            yy, xx = np.where(self._get_eroded_mask(obs.valid_mask))
            if yy.size == 0:
                continue

            center_xy = (obs.center_xy[0] + pose_delta[i, 1], obs.center_xy[1] + pose_delta[i, 0])
            yy_g, xx_g = self._local_to_global_coords(obs, yy.astype(float), xx.astype(float), center_xy=center_xy)
            valid_global = (
                (yy_g >= 0.0) & (yy_g < global_shape[0]) &
                (xx_g >= 0.0) & (xx_g < global_shape[1])
            )
            if not np.any(valid_global):
                continue

            yy_v = yy[valid_global]
            xx_v = xx[valid_global]
            yy_g = yy_g[valid_global]
            xx_g = xx_g[valid_global]

            loo_sum = self._sample_global_array(sum_z - contrib_sums[i], yy_g, xx_g, order=1)
            loo_w = self._sample_global_array(count - contrib_counts[i], yy_g, xx_g, order=1)
            overlap_local = self._sample_global_array(overlap_count.astype(float), yy_g, xx_g, order=1)
            valid_loo = (loo_w > loo_threshold) & (overlap_local >= 1.5)
            if not np.any(valid_loo):
                continue

            pred_loo = loo_sum[valid_loo] / np.maximum(loo_w[valid_loo], eps)
            p, tip, tilt, _ = nuisances[i]
            y_norm = 2.0 * yy_v[valid_loo] / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx_v[valid_loo] / max(cols - 1, 1) - 1.0
            model = p + tip * y_norm + tilt * x_norm
            z_local = obs.z[yy_v[valid_loo], xx_v[valid_loo]]
            ref_local = reference_map[yy_v[valid_loo], xx_v[valid_loo]]
            residual = z_local - model - ref_local - pred_loo

            valid = np.isfinite(residual)
            if np.any(valid):
                total_residual_sq += np.sum(residual[valid] ** 2)
                total_count += int(np.sum(valid))

        if total_count == 0:
            return None
        return total_residual_sq / total_count

    def _apply_pose_corrections(self, observations: tuple, corrections: np.ndarray) -> tuple:
        registered = []
        for i, obs in enumerate(observations):
            dy, dx = corrections[i]
            if abs(dy) < 1e-6 and abs(dx) < 1e-6:
                registered.append(obs)
                continue

            registered.append(SubApertureObservation(
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
            ))

        return tuple(registered)

    def _fuse_observations(
        self,
        observations: tuple[SubApertureObservation, ...],
        nuisances: np.ndarray,
        reference_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fused_result = self._fuse_observations_with_contrib(
            observations,
            nuisances,
            reference_map,
        )
        return fused_result['fused_z'], fused_result['valid_mask'], fused_result['support']

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

        fused_filled = np.nan_to_num(fused_z, nan=0.0)
        fused_mask_f = fused_mask.astype(float)
        samples = []
        residual_bank = []

        for i, obs in enumerate(observations):
            rows, cols = obs.tile_shape
            yy_full, xx_full = np.indices(obs.tile_shape, dtype=float)
            y_norm_full = 2.0 * yy_full / max(rows - 1, 1) - 1.0
            x_norm_full = 2.0 * xx_full / max(cols - 1, 1) - 1.0

            p, tip, tilt, _focus = nuisances[i]
            model = p + tip * y_norm_full + tilt * x_norm_full

            yy, xx = np.where(obs.valid_mask)
            if yy.size == 0:
                continue

            yy_g, xx_g = self._local_to_global_coords(obs, yy.astype(float), xx.astype(float))
            valid_global = (
                (yy_g >= 0.0)
                & (yy_g < fused_z.shape[0])
                & (xx_g >= 0.0)
                & (xx_g < fused_z.shape[1])
            )
            if not np.any(valid_global):
                continue

            yy = yy[valid_global]
            xx = xx[valid_global]
            yy_g = yy_g[valid_global]
            xx_g = xx_g[valid_global]
            mask_sample = self._sample_global_array(fused_mask_f, yy_g, xx_g, order=1)
            valid_sample = mask_sample > 0.75
            if not np.any(valid_sample):
                continue

            yy = yy[valid_sample]
            xx = xx[valid_sample]
            yy_g = yy_g[valid_sample]
            xx_g = xx_g[valid_sample]
            fused_sample = self._sample_global_array(fused_filled, yy_g, xx_g, order=1)

            residual = obs.z[yy, xx] - model[yy, xx] - fused_sample
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

        ref_filled = np.where(valid, reference_map, 0.0)
        ref_smoothed = ndimage.gaussian_filter(ref_filled, sigma=SIGMA_FILTER)
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
            yy, xx = np.indices(obs.tile_shape, dtype=float)
            y_norm = 2.0 * yy / max(rows - 1, 1) - 1.0
            x_norm = 2.0 * xx / max(cols - 1, 1) - 1.0

            working_mask = self._get_eroded_mask(obs.valid_mask)
            weight_map = self._smooth_feather_weights(working_mask)
            local_values = obs.z - reference_map

            gy, gx, sampled_z, sampled_w, sampled_mask = self._project_local_field_to_global(
                obs,
                local_values,
                weight_map,
                obs.valid_mask,
            )
            if gy.size == 0:
                continue

            _, _, sampled_xn, _, _ = self._project_local_field_to_global(
                obs,
                x_norm,
                np.ones_like(weight_map, dtype=float),
                obs.valid_mask,
            )
            _, _, sampled_yn, _, _ = self._project_local_field_to_global(
                obs,
                y_norm,
                np.ones_like(weight_map, dtype=float),
                obs.valid_mask,
            )

            valid = sampled_mask & np.isfinite(sampled_z) & (sampled_w > 1e-6)
            if not np.any(valid):
                continue

            gy_v = gy[valid]
            gx_v = gx[valid]
            all_obs_indices.append(np.full(int(np.sum(valid)), i, dtype=int))
            all_flat_indices.append(gy_v * global_shape[1] + gx_v)
            all_z.append(sampled_z[valid])
            all_xn.append(sampled_xn[valid])
            all_yn.append(sampled_yn[valid])
            all_w.append(sampled_w[valid])

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