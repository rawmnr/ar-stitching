"""Legacy NEOSS port.

The MATLAB version is a three-stage pipeline:
1. build a detector-fixed random/reference map,
2. solve per-subaperture alignment against that map, then
3. stitch corrected tiles onto the global grid.

This port keeps the same structure without trying to improve the legacy math.
"""
from __future__ import annotations

import numpy as np
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.editable._legacy_basis import (
    basis_term_stack,
    evaluate_basis_surface,
    fit_basis_coefficients,
    observed_support_mask,
    remove_low_order_modes,
    rounded_placement_slices,
)


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
                metadata={"method": "neoss_legacy"},
            )

        global_shape = observations[0].global_shape
        tile_shape = observations[0].tile_shape
        support = observed_support_mask(observations, global_shape)

        tp_mode = str(config.metadata.get("neoss_tp_mode", config.metadata.get("truth_basis", "Z"))).upper()
        cs_mode = str(config.metadata.get("neoss_cs_mode", "L" if tp_mode == "L" else "Z")).upper()
        if tp_mode not in {"L", "Z"}:
            tp_mode = "Z"
        if cs_mode not in {"L", "Z"}:
            cs_mode = "L"

        align_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
        tp_terms = tuple(int(v) for v in config.metadata.get("neoss_tp_terms", (0, 1, 2, 3)))
        cs_terms = tuple(int(v) for v in config.metadata.get("neoss_cs_terms", (0, 1, 2, 3, 4, 5)))
        radius_fraction = None
        if tp_mode == "Z" or cs_mode == "Z":
            radius_fraction = float(config.metadata.get("detector_radius_fraction", 0.48))

        detector_cal = self._initial_detector_calibration(observations, tile_shape, tp_mode, tp_terms, radius_fraction)
        detector_cal = remove_low_order_modes(
            detector_cal,
            np.isfinite(detector_cal),
            tp_mode,
            tp_terms[: min(len(tp_terms), 4)],
            radius_fraction=radius_fraction,
        )

        global_map = np.zeros(global_shape, dtype=float)
        coeffs = np.zeros((len(observations), len(align_terms)), dtype=float)

        for _ in range(4):
            global_map, valid_mask, coeffs = self._fuse_with_alignment(
                observations=observations,
                global_shape=global_shape,
                global_map=global_map,
                detector_cal=detector_cal,
                coeffs=coeffs,
                align_terms=align_terms,
                cs_mode=cs_mode,
                cs_terms=cs_terms,
                radius_fraction=radius_fraction,
            )

            new_detector = self._update_detector_calibration(
                observations=observations,
                global_map=global_map,
                detector_cal=detector_cal,
                coeffs=coeffs,
                align_terms=align_terms,
                tp_mode=tp_mode,
                tp_terms=tp_terms,
                radius_fraction=radius_fraction,
            )
            new_detector = remove_low_order_modes(
                new_detector,
                np.isfinite(new_detector),
                tp_mode,
                tp_terms[: min(len(tp_terms), 4)],
                radius_fraction=radius_fraction,
            )

            if np.nanmax(np.abs(np.nan_to_num(new_detector - detector_cal, nan=0.0))) < 1e-6:
                detector_cal = new_detector
                break
            detector_cal = new_detector

        z, valid_mask, final_support, mismatch_rms = self._final_reconstruction(
            observations=observations,
            global_shape=global_shape,
            detector_cal=detector_cal,
            coeffs=coeffs,
            align_terms=align_terms,
            cs_mode=cs_mode,
            cs_terms=cs_terms,
            radius_fraction=radius_fraction,
        )

        cal_mask = np.isfinite(detector_cal)
        cal_map = np.full(tile_shape, np.nan, dtype=float)
        cal_map[cal_mask] = detector_cal[cal_mask]

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=final_support,
            metadata={
                "method": "neoss_legacy",
                "instrument_calibration": cal_map,
                "mismatch_rms": mismatch_rms,
                "alignment_terms": align_terms,
            },
        )

    def _initial_detector_calibration(
        self,
        observations: tuple[SubApertureObservation, ...],
        tile_shape: tuple[int, int],
        mode: str,
        terms: tuple[int, ...],
        radius_fraction: float | None,
    ) -> np.ndarray:
        data_stack = np.stack([np.asarray(obs.z, dtype=float) for obs in observations], axis=0)
        finite = np.isfinite(data_stack)
        counts = np.sum(finite, axis=0)
        cal = np.zeros(tile_shape, dtype=float)
        summed = np.nansum(data_stack, axis=0)
        valid = counts > 0
        cal[valid] = summed[valid] / counts[valid]
        basis_stack, basis_mask = basis_term_stack(mode, terms, tile_shape, radius_fraction=radius_fraction)
        if np.any(basis_mask):
            coeffs = fit_basis_coefficients(cal, basis_stack, basis_mask)
            cal = cal - evaluate_basis_surface(coeffs, basis_stack)
        return cal

    def _fuse_with_alignment(
        self,
        observations: tuple[SubApertureObservation, ...],
        global_shape: tuple[int, int],
        global_map: np.ndarray,
        detector_cal: np.ndarray,
        coeffs: np.ndarray,
        align_terms: tuple[int, ...],
        cs_mode: str,
        cs_terms: tuple[int, ...],
        radius_fraction: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        basis_stack, basis_mask = basis_term_stack("L" if cs_mode == "L" else cs_mode, align_terms, observations[0].tile_shape, radius_fraction=radius_fraction)
        coeffs_new = np.zeros((len(observations), len(align_terms)), dtype=float)

        for i, obs in enumerate(observations):
            gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
            if gy.stop <= gy.start or gx.stop <= gx.start:
                continue
            local_obs = np.asarray(obs.z, dtype=float)[ly, lx]
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
            local_basis_stack = basis_stack[:, ly, lx]
            local_basis_mask = basis_mask[ly, lx]
            if not np.any(local_mask):
                continue

            local_det = np.asarray(detector_cal, dtype=float)[ly, lx]
            global_ref = np.nan_to_num(global_map[gy, gx], nan=0.0)
            residual = local_obs - local_det - global_ref
            coeffs_new[i] = fit_basis_coefficients(residual, local_basis_stack, local_mask & local_basis_mask)
            corrected = local_obs - evaluate_basis_surface(coeffs_new[i], local_basis_stack) - local_det

            global_view_sum = sum_z[gy, gx]
            global_view_count = count[gy, gx]
            global_view_sum[local_mask] += corrected[local_mask]
            global_view_count[local_mask] += 1.0

        valid_mask = count > 0
        fused = np.full(global_shape, np.nan, dtype=float)
        fused[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        return fused, valid_mask, coeffs_new

    def _update_detector_calibration(
        self,
        observations: tuple[SubApertureObservation, ...],
        global_map: np.ndarray,
        detector_cal: np.ndarray,
        coeffs: np.ndarray,
        align_terms: tuple[int, ...],
        tp_mode: str,
        tp_terms: tuple[int, ...],
        radius_fraction: float | None,
    ) -> np.ndarray:
        sum_r = np.zeros_like(detector_cal, dtype=float)
        sum_w = np.zeros_like(detector_cal, dtype=float)
        basis_stack, basis_mask = basis_term_stack(tp_mode, align_terms, observations[0].tile_shape, radius_fraction=radius_fraction)

        for i, obs in enumerate(observations):
            gy, gx, ly, lx = rounded_placement_slices(global_map.shape, obs.tile_shape, obs.center_xy)
            if gy.stop <= gy.start or gx.stop <= gx.start:
                continue
            local_obs = np.asarray(obs.z, dtype=float)[ly, lx]
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
            if not np.any(local_mask):
                continue

            local_basis_stack = basis_stack[:, ly, lx]
            aligned = evaluate_basis_surface(coeffs[i], local_basis_stack)
            local_global = np.nan_to_num(global_map[gy, gx], nan=0.0)
            residual = local_obs - aligned - local_global
            sum_r_view = sum_r[ly, lx]
            sum_w_view = sum_w[ly, lx]
            sum_r_view[local_mask] += residual[local_mask]
            sum_w_view[local_mask] += 1.0

        updated = np.where(sum_w > 0, sum_r / np.maximum(sum_w, 1.0), detector_cal)
        return np.where(np.isfinite(updated), updated, detector_cal)

    def _final_reconstruction(
        self,
        observations: tuple[SubApertureObservation, ...],
        global_shape: tuple[int, int],
        detector_cal: np.ndarray,
        coeffs: np.ndarray,
        align_terms: tuple[int, ...],
        cs_mode: str,
        cs_terms: tuple[int, ...],
        radius_fraction: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)
        mismatch_stack: list[np.ndarray] = []
        basis_stack, basis_mask = basis_term_stack("L" if cs_mode == "L" else cs_mode, align_terms, observations[0].tile_shape, radius_fraction=radius_fraction)

        for i, obs in enumerate(observations):
            gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
            if gy.stop <= gy.start or gx.stop <= gx.start:
                continue
            local_obs = np.asarray(obs.z, dtype=float)[ly, lx]
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
            if not np.any(local_mask):
                continue

            local_basis_stack = basis_stack[:, ly, lx]
            corrected = local_obs - evaluate_basis_surface(coeffs[i], local_basis_stack) - np.asarray(detector_cal, dtype=float)[ly, lx]
            global_view_sum = sum_z[gy, gx]
            global_view_count = count[gy, gx]
            global_view_support = support[gy, gx]
            global_view_sum[local_mask] += corrected[local_mask]
            global_view_count[local_mask] += 1.0
            global_view_support[local_mask] = True
            mismatch_stack.append(corrected[local_mask])

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        mismatch_rms = float(np.sqrt(np.mean(np.square(np.concatenate(mismatch_stack))))) if mismatch_stack else 0.0
        return z, valid_mask, support, mismatch_rms
