"""Legacy NEOSS port with a global MLR-style solve.

The MATLAB legacy fits three blocks jointly:
- per-observation alignment terms,
- global TP terms,
- global CS terms.

This Python version keeps the same block structure and the legacy calibration /
stitching split, but implements the solve with a sparse least-squares system
instead of the MATLAB SVD block assembly.
"""
from __future__ import annotations

import numpy as np

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.editable._legacy_basis import (
    align_tile_to_rounded_grid,
    center_tile_in_canvas,
    basis_term_stack,
    evaluate_basis_surface,
    fit_basis_coefficients,
    observed_support_mask,
    overlap_support_mask,
    remove_low_order_modes,
    project_global_mask_to_tile,
    sample_basis_term_stack_from_coords,
    _zernike_index_pairs,
    _zernike_radial,
    shift_canvas,
)


def _normalize_mode(value: object, fallback: str = "Z") -> str:
    text = str(value).strip().upper()
    if text.startswith("Z"):
        return "Z"
    if text.startswith("L"):
        return "L"
    return fallback


def _resolve_terms(
    configured: object | None,
    start_after: tuple[int, ...],
    default_count: int,
) -> tuple[int, ...]:
    if configured is not None:
        return tuple(int(v) for v in configured)
    start = (max(start_after) + 1) if start_after else 0
    return tuple(range(start, start + default_count))


def _canonicalize_svd_columns(U: np.ndarray) -> np.ndarray:
    """Fix the arbitrary sign of singular vectors using a deterministic rule."""

    U = np.array(U, copy=True, dtype=float)
    if U.size == 0:
        return U
    for col in range(U.shape[1]):
        column = U[:, col]
        if not np.any(np.isfinite(column)):
            continue
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0:
            U[:, col] *= -1.0
    return U


def _legacy_pupil_weights(
    shape: tuple[int, int],
    mode: str,
    radius_fraction: float | None,
    sigma_px: float | None = None,
) -> np.ndarray:
    """Approximate the MATLAB `cartePonderation` weight map."""

    if mode != "Z":
        return np.ones(shape, dtype=float)

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=float)
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    if radius_fraction is None:
        radius_px = 0.5 * min(rows, cols)
    else:
        radius_px = min(rows, cols) * float(radius_fraction)

    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    rr = np.sqrt(rr2)
    if sigma_px is None:
        sigma_px = max(0.72 * radius_px, 1.0)

    weight = np.exp((((radius_px - sigma_px) ** 2) - rr2) / (2.0 * sigma_px**2))
    weight = np.clip(weight, 0.1, 1.0)
    weight[rr > radius_px] = np.nan
    return weight


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
        overlap_support = overlap_support_mask(observations, global_shape)

        tp_mode = _normalize_mode(config.metadata.get("neoss_tp_mode", config.metadata.get("truth_basis", "Z")))
        cs_mode = _normalize_mode(config.metadata.get("neoss_cs_mode", "L" if tp_mode == "L" else "Z"), fallback="Z")
        # Keep the repository's zero-based basis convention for the solver blocks.
        # The MATLAB port logic is mirrored in structure, not in raw index numbering.
        align_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
        tp_default_count = int(config.metadata.get("neoss_tp_default_count", 36))
        cs_default_count = int(config.metadata.get("neoss_cs_default_count", 36))
        tp_terms = _resolve_terms(config.metadata.get("neoss_tp_terms"), align_terms, default_count=tp_default_count)
        cs_terms = _resolve_terms(config.metadata.get("neoss_cs_terms"), align_terms, default_count=cs_default_count)
        tp_terms = tuple(term for term in tp_terms if term not in align_terms)
        cs_terms = tuple(term for term in cs_terms if term not in align_terms and term not in {3, 4, 5})

        if tp_mode == "Z" or cs_mode == "Z":
            radius_fraction = float(config.metadata.get("detector_radius_fraction", 0.48))
            zernike_indexing = str(config.metadata.get("neoss_zernike_indexing", "iso")).lower()
            if zernike_indexing not in {"iso", "noll", "fringe"}:
                zernike_indexing = "iso"
        else:
            radius_fraction = None
            zernike_indexing = "fringe"

        disable_random_map = bool(config.metadata.get("neoss_disable_random_map", True))
        random_map_limit = float(config.metadata.get("neoss_limit", config.metadata.get("limit", 1.0)))
        sigma_px_meta = config.metadata.get("neoss_sigma_px", config.metadata.get("neoss_sigma"))
        sigma_px = float(sigma_px_meta) if sigma_px_meta is not None else None
        if disable_random_map:
            detector_cal = np.zeros(tile_shape, dtype=float)
        else:
            detector_cal = self._initial_detector_calibration(
                observations=observations,
                tile_shape=tile_shape,
                mode=tp_mode,
                terms=tp_terms,
                radius_fraction=radius_fraction,
                zernike_indexing=zernike_indexing,
                limit=random_map_limit,
            )

        solve_result = self._solve_mlr(
            observations=observations,
            detector_cal=detector_cal,
            global_shape=global_shape,
            tile_shape=tile_shape,
            tp_mode=tp_mode,
            tp_terms=tp_terms,
            cs_mode=cs_mode,
            cs_terms=cs_terms,
            align_terms=align_terms,
            radius_fraction=radius_fraction,
            zernike_indexing=zernike_indexing,
            overlap_support=overlap_support,
            coord_system=str(config.metadata.get("neoss_coordinate_system", "IRIDE")),
        )

        global_tp_coeffs = solve_result["tp_coeffs"]
        cs_coeffs = solve_result["cs_coeffs"]
        align_coeffs = solve_result["align_coeffs"]
        detector_map = solve_result["detector_map"]
        solve_rms = solve_result["solve_rms"]

        instrument_calibration = np.array(detector_map, copy=True, dtype=float)

        z, valid_mask, final_support, mismatch_map = self._stitch_observations(
            observations=observations,
            detector_cal=instrument_calibration,
            align_coeffs=align_coeffs,
            tp_mode=tp_mode,
            align_terms=align_terms,
            radius_fraction=radius_fraction,
            zernike_indexing=zernike_indexing,
            overlap_support=overlap_support,
            global_shape=global_shape,
            weight_map=_legacy_pupil_weights(tile_shape, tp_mode, radius_fraction, sigma_px=sigma_px),
        )

        mismatch_rms = float(np.sqrt(np.nanmean(np.square(mismatch_map)))) if np.any(np.isfinite(mismatch_map)) else 0.0
        cal_mask = np.isfinite(instrument_calibration)
        cal_map = np.full(tile_shape, np.nan, dtype=float)
        cal_map[cal_mask] = instrument_calibration[cal_mask]

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={
                "method": "neoss_legacy_mlr",
                "instrument_calibration": cal_map,
                "mismatch_rms": mismatch_rms,
                "solve_rms": solve_rms,
                "alignment_terms": align_terms,
                "tp_terms": tp_terms,
                "cs_terms": cs_terms,
            },
        )

    def _initial_detector_calibration(
        self,
        observations: tuple[SubApertureObservation, ...],
        tile_shape: tuple[int, int],
        mode: str,
        terms: tuple[int, ...],
        radius_fraction: float | None,
        zernike_indexing: str,
        limit: float,
    ) -> np.ndarray:
        data_stack = np.stack([np.asarray(obs.z, dtype=float) for obs in observations], axis=0)
        finite = np.isfinite(data_stack)
        support_ratio = np.sum(finite, axis=(1, 2)) / float(tile_shape[0] * tile_shape[1])
        best_ratio = np.max(support_ratio)
        if best_ratio <= float(limit):
            return np.zeros(tile_shape, dtype=float)
        chosen = support_ratio == best_ratio
        if not np.any(chosen):
            chosen = np.ones(len(observations), dtype=bool)

        selected = data_stack[chosen]
        if selected.size == 0:
            return np.zeros(tile_shape, dtype=float)
        # Match the legacy MATLAB helper: average the best-support tiles as a
        # plain sum/mean over the selected tiles, letting NaNs propagate.
        cal = np.sum(selected, axis=0) / float(selected.shape[0])

        if mode.upper() == "Z":
            # Mirror the legacy MATLAB helper:
            # fit a richer Zernike stack, then remove only the low-order terms.
            fit_terms = tuple(range(0, 51))
            fit_stack, fit_mask = basis_term_stack(
                mode,
                fit_terms,
                tile_shape,
                radius_fraction=radius_fraction,
                zernike_indexing=zernike_indexing,
            )
            if np.any(fit_mask):
                coeffs = fit_basis_coefficients(cal, fit_stack, fit_mask)
                cal = cal - evaluate_basis_surface(coeffs[:4], fit_stack[:4])
        else:
            basis_stack, basis_mask = basis_term_stack(
                mode,
                (0, 1, 2, 3),
                tile_shape,
                radius_fraction=radius_fraction,
                zernike_indexing=zernike_indexing,
            )
            if np.any(basis_mask):
                coeffs = fit_basis_coefficients(cal, basis_stack, basis_mask)
                cal = cal - evaluate_basis_surface(coeffs, basis_stack)
        return cal

    def _legacy_param(
        self,
        obs: SubApertureObservation,
        global_shape: tuple[int, int],
        coord_system: str,
    ) -> tuple[float, float, float, float]:
        """Approximate the MATLAB `calculParametreGrille` output."""

        dx = float(obs.translation_xy[0])
        dy = float(obs.translation_xy[1])
        coord_system = coord_system.lower()
        if coord_system == "polaire":
            return (
                0.0,
                0.0,
                -2.0 * dx / float(max(global_shape[1], 1)),
                -dy,
            )
        return (
            2.0 * dx / float(max(global_shape[1], 1)),
            2.0 * dy / float(max(global_shape[0], 1)),
            0.0,
            0.0,
        )

    def _legacy_grille(
        self,
        carte_shape: tuple[int, int],
        rho: float,
        mode: str,
        param: tuple[float, float, float, float],
        coord_system: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate the MATLAB `grille` helper."""

        rows, cols = carte_shape
        yy, xx = np.indices(carte_shape, dtype=float)
        x = (xx - np.mean(xx)) / max(np.floor(cols / 2.0), 1.0)
        y = (yy - np.mean(yy)) / max(np.floor(cols / 2.0), 1.0)
        if coord_system.lower() == "polaire":
            y = np.flipud(y)

        x = rho * x + float(param[0])
        y = rho * y + float(param[1])

        if mode.upper() == "L":
            return x, y

        if float(param[2]) != 0.0:
            x = x + float(param[2])
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        if float(param[3]) != 0.0:
            theta = np.mod(theta + float(param[3]) * np.pi / 180.0, 2.0 * np.pi)
        r = np.sqrt(x**2 + y**2)
        return theta, r

    def _legacy_term_surface(
        self,
        mode: str,
        term: int,
        coord1: np.ndarray,
        coord2: np.ndarray,
        zernike_indexing: str,
        *,
        block: str = "generic",
    ) -> np.ndarray:
        """Return one legacy basis surface using MATLAB term conventions.

        The MATLAB NEOSS code only applies the Legendre L3/L5 mixing to the
        alignment and TP blocks. CS terms must remain untouched.
        """

        if mode.upper() == "L":
            if block in {"alignment", "tp"} and term in {3, 5}:
                a = sample_basis_term_stack_from_coords("L", [3], coord1, coord2)[0][0]
                b = sample_basis_term_stack_from_coords("L", [5], coord1, coord2)[0][0]
                return a + b if term == 3 else a - b
            return sample_basis_term_stack_from_coords("L", [term], coord1, coord2)[0][0]

        n, m = _zernike_index_pairs(zernike_indexing, term + 1)[term]
        radial = _zernike_radial(n, m, coord2)
        if m == 0:
            return radial
        if m > 0:
            return radial * np.cos(m * coord1)
        return radial * np.sin(abs(m) * coord1)

    def _remplissage_matrice_fit(
        self,
        *,
        tp_mode: str,
        cs_mode: str,
        align_terms: tuple[int, ...],
        tp_terms: tuple[int, ...],
        cs_terms: tuple[int, ...],
        coords: dict[str, np.ndarray],
        zernike_indexing: str,
    ) -> np.ndarray:
        """Construct the local MATLAB-style fit matrix `T`."""

        coord1_tp = coords["coord1_TP"]
        coord2_tp = coords["coord2_TP"]
        coord1_cs = coords["coord1_CS"]
        coord2_cs = coords["coord2_CS"]
        n_pix = coord1_tp.size
        n_cols = len(align_terms) + len(tp_terms) + len(cs_terms)
        if n_pix == 0 or n_cols == 0:
            return np.zeros((0, n_cols), dtype=float)

        T = np.zeros((n_pix, n_cols), dtype=float)
        col = 0

        for term in align_terms:
            T[:, col] = self._legacy_term_surface(
                tp_mode,
                term,
                coord1_tp,
                coord2_tp,
                zernike_indexing,
                block="alignment",
            ).ravel()
            col += 1

        for term in tp_terms:
            T[:, col] = self._legacy_term_surface(
                tp_mode,
                term,
                coord1_tp,
                coord2_tp,
                zernike_indexing,
                block="tp",
            ).ravel()
            col += 1

        for term in cs_terms:
            T[:, col] = self._legacy_term_surface(
                cs_mode,
                term,
                coord1_cs,
                coord2_cs,
                zernike_indexing,
                block="cs",
            ).ravel()
            col += 1

        return T

    def _legacy_align_coefficients(self, coeffs: np.ndarray, mode: str) -> np.ndarray:
        """Mirror MATLAB `calculCarteAlignement` coefficient reshaping."""

        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.size == 0 or mode.upper() != "L":
            return coeffs
        if coeffs.size < 4:
            return coeffs
        if coeffs.size == 4:
            return np.array([coeffs[0], coeffs[1], coeffs[2], coeffs[3], 0.0, coeffs[3]], dtype=float)
        if coeffs.size == 5:
            return np.array([coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[3]], dtype=float)
        transformed = np.array(
            [
                coeffs[0],
                coeffs[1],
                coeffs[2],
                coeffs[3] + coeffs[5],
                coeffs[4],
                coeffs[3] - coeffs[5],
            ],
            dtype=float,
        )
        if coeffs.size > 6:
            transformed = np.concatenate([transformed, coeffs[6:]])
        return transformed

    def _solve_mlr(
        self,
        observations: tuple[SubApertureObservation, ...],
        detector_cal: np.ndarray,
        global_shape: tuple[int, int],
        tile_shape: tuple[int, int],
        tp_mode: str,
        tp_terms: tuple[int, ...],
        cs_mode: str,
        cs_terms: tuple[int, ...],
        align_terms: tuple[int, ...],
        radius_fraction: float | None,
        zernike_indexing: str,
        overlap_support: np.ndarray,
        coord_system: str,
    ) -> dict[str, np.ndarray | float]:
        n_align = len(align_terms)
        n_tp = len(tp_terms)
        n_cs = len(cs_terms)
        n_ha = n_tp + n_cs
        n_obs = len(observations)
        n_elmt_y_i = n_ha + n_align

        if n_elmt_y_i == 0:
            return {
                "tp_coeffs": np.zeros((0,), dtype=float),
                "cs_coeffs": np.zeros((0,), dtype=float),
                "align_coeffs": np.zeros((n_obs, 0), dtype=float),
                "solve_rms": 0.0,
            }

        tp_rho = 1.0
        cs_rho = float(tile_shape[0]) / float(max(global_shape[0], 1))

        coord1_tp, coord2_tp = self._legacy_grille(
            tile_shape,
            tp_rho,
            tp_mode,
            (0.0, 0.0, 0.0, 0.0),
            coord_system,
        )

        M = np.zeros((n_obs * n_elmt_y_i, n_ha + n_align * n_obs), dtype=float)
        y = np.zeros(n_obs * n_elmt_y_i, dtype=float)

        for obs_idx, obs in enumerate(observations):
            block_rows = slice(obs_idx * n_elmt_y_i, (obs_idx + 1) * n_elmt_y_i)
            detector_view = np.asarray(detector_cal, dtype=float)
            local_obs = np.asarray(obs.z, dtype=float) - detector_view
            local_mask = np.asarray(obs.valid_mask, dtype=bool)
            local_overlap = project_global_mask_to_tile(
                overlap_support,
                global_shape,
                obs.tile_shape,
                obs.center_xy,
            )
            fit_mask = local_mask & local_overlap & np.isfinite(local_obs) & np.isfinite(detector_view)
            if not np.any(fit_mask):
                fit_mask = local_mask & np.isfinite(local_obs) & np.isfinite(detector_view)
            if not np.any(fit_mask):
                continue

            param = self._legacy_param(obs, global_shape, coord_system)
            coord1_cs, coord2_cs = self._legacy_grille(
                obs.tile_shape,
                cs_rho,
                cs_mode,
                param,
                coord_system,
            )
            coords = {
                "coord1_TP": coord1_tp[fit_mask],
                "coord2_TP": coord2_tp[fit_mask],
                "coord1_CS": coord1_cs[fit_mask],
                "coord2_CS": coord2_cs[fit_mask],
            }
            T = self._remplissage_matrice_fit(
                tp_mode=tp_mode,
                cs_mode=cs_mode,
                align_terms=align_terms,
                tp_terms=tp_terms,
                cs_terms=cs_terms,
                coords=coords,
                zernike_indexing=zernike_indexing,
            )
            carte = local_obs[fit_mask]
            if T.size == 0 or carte.size == 0:
                continue
            if T.shape[0] < T.shape[1]:
                # The MATLAB solver relies on the local fit being at least
                # reasonably overdetermined; keep the same assumption here.
                continue

            U, _, Vt = np.linalg.svd(T, full_matrices=False)
            U = _canonicalize_svd_columns(U)
            y_i = U.T @ carte
            M_i = U.T @ T

            y[block_rows] = y_i
            M[block_rows, :n_ha] = M_i[:, n_align:]
            align_col = n_ha + obs_idx * n_align
            M[block_rows, align_col : align_col + n_align] = M_i[:, :n_align]

        try:
            x = np.linalg.solve(M, y)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(M, y, rcond=None)
        ha_coeffs = x[:n_ha]
        align_coeffs = x[n_ha:].reshape(n_obs, n_align) if n_align else np.zeros((n_obs, 0), dtype=float)

        tp_basis_stack, tp_mask = basis_term_stack(
            tp_mode,
            tp_terms,
            tile_shape,
            radius_fraction=radius_fraction,
            zernike_indexing=zernike_indexing,
        )
        tp_coeffs = ha_coeffs[:n_tp]
        cs_coeffs = ha_coeffs[n_tp:]
        tp_surface = evaluate_basis_surface(tp_coeffs, tp_basis_stack) if tp_basis_stack.size else np.zeros(tile_shape, dtype=float)
        detector_map = np.asarray(detector_cal, dtype=float) + tp_surface
        if tp_mode == "Z":
            detector_map = np.where(tp_mask, detector_map, np.nan)

        solve_residual = M @ x - y
        solve_rms = float(np.sqrt(np.mean(np.square(solve_residual)))) if solve_residual.size else 0.0

        return {
            "tp_coeffs": tp_coeffs,
            "cs_coeffs": cs_coeffs,
            "align_coeffs": align_coeffs,
            "detector_map": detector_map,
            "solve_rms": solve_rms,
            "global_coeffs": x,
        }

    def _stitch_observations(
        self,
        observations: tuple[SubApertureObservation, ...],
        detector_cal: np.ndarray,
        align_coeffs: np.ndarray,
        tp_mode: str,
        align_terms: tuple[int, ...],
        radius_fraction: float | None,
        zernike_indexing: str,
        overlap_support: np.ndarray,
        global_shape: tuple[int, int],
        weight_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sum_z = np.zeros(global_shape, dtype=float)
        sum_w = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)
        sum_sq = np.zeros(global_shape, dtype=float)

        align_basis_stack, align_mask = basis_term_stack(
            tp_mode,
            align_terms,
            observations[0].tile_shape,
            radius_fraction=radius_fraction,
            zernike_indexing=zernike_indexing,
        )

        for obs_idx, obs in enumerate(observations):
            local_obs = np.asarray(obs.z, dtype=float)
            local_mask = np.asarray(obs.valid_mask, dtype=bool)
            if not np.any(local_mask):
                continue

            local_weight = np.asarray(weight_map, dtype=float)
            if align_coeffs.size:
                align_coeff = self._legacy_align_coefficients(align_coeffs[obs_idx], tp_mode)
                if tp_mode == "L":
                    align_basis_stack_eval, _ = basis_term_stack(
                        tp_mode,
                        tuple(range(len(align_coeff))),
                        observations[0].tile_shape,
                        radius_fraction=radius_fraction,
                        zernike_indexing=zernike_indexing,
                    )
                    align_model = evaluate_basis_surface(align_coeff, align_basis_stack_eval)
                else:
                    align_model = evaluate_basis_surface(align_coeff, align_basis_stack)
            else:
                align_model = np.zeros_like(local_obs)
            corrected = local_obs - align_model - np.asarray(detector_cal, dtype=float)

            corrected_canvas, corrected_mask = center_tile_in_canvas(corrected, local_mask, global_shape)
            weight_canvas, weight_mask = center_tile_in_canvas(local_weight, local_mask, global_shape)

            shift_x = float(obs.center_xy[0]) - (global_shape[1] - 1) / 2.0
            shift_y = float(obs.center_xy[1]) - (global_shape[0] - 1) / 2.0
            shifted_corrected, shifted_corrected_mask = shift_canvas(
                corrected_canvas,
                corrected_mask,
                (shift_x, shift_y),
                order=3,
            )
            shifted_weight, shifted_weight_mask = shift_canvas(
                weight_canvas,
                weight_mask,
                (shift_x, shift_y),
                order=1,
            )

            update_mask = shifted_corrected_mask & shifted_weight_mask
            if not np.any(update_mask):
                continue

            values = shifted_corrected[update_mask]
            weights = np.where(np.isfinite(shifted_weight[update_mask]), shifted_weight[update_mask], 0.0)
            values_mask = np.isfinite(values)
            if not np.any(values_mask):
                continue
            update_mask_idx = np.zeros_like(update_mask, dtype=bool)
            update_mask_idx[update_mask] = values_mask
            sum_z[update_mask_idx] += values[values_mask] * weights[values_mask]
            sum_sq[update_mask_idx] += (values[values_mask] ** 2) * weights[values_mask]
            sum_w[update_mask_idx] += weights[values_mask]
            support[update_mask_idx] = True

        valid_mask = sum_w > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / sum_w[valid_mask]

        mismatch = np.full(global_shape, np.nan, dtype=float)
        mismatch[valid_mask] = np.sqrt(np.maximum(sum_sq[valid_mask] / sum_w[valid_mask] - z[valid_mask] ** 2, 0.0))
        mismatch[mismatch < 0.001] = np.nan
        return z, valid_mask, support, mismatch



