"""Legacy Subaper port with a coupled global solve."""
from __future__ import annotations

import numpy as np

from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.editable._legacy_basis import (
    align_tile_to_rounded_grid,
    basis_term_stack,
    evaluate_basis_surface,
    fit_basis_coefficients,
    observed_support_mask,
    place_tile_in_global_frame,
)


def _normalize_mode(value: object, fallback: str = "L") -> str:
    text = str(value).strip().upper()
    if text.startswith("LM"):
        return "LM"
    if text.startswith("Z"):
        return "Z"
    if text.startswith("L"):
        return "L"
    return fallback


def _apply_basis_scale(stack: np.ndarray, scale: object) -> np.ndarray:
    arr = np.asarray(scale, dtype=float)
    if arr.ndim == 0:
        return stack * float(arr)
    if arr.ndim == 1 and arr.shape[0] == stack.shape[0]:
        return stack * arr[:, None, None]
    return stack


def _basis_stack_for_mode(
    mode: str,
    basis_terms: tuple[int, ...],
    shape: tuple[int, int],
    *,
    radius_fraction: float | None,
    basis_scale: object = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the legacy basis stack, including LM mixing when requested."""

    term_list = tuple(int(v) for v in basis_terms)
    if not term_list:
        empty = np.zeros((0, shape[0], shape[1]), dtype=float)
        return empty, np.zeros(shape, dtype=bool)

    if mode == "LM":
        base_terms = tuple(dict.fromkeys(list(term_list) + ([5] if 3 in term_list else [])))
        base_stack, base_mask = basis_term_stack(
            "L",
            base_terms,
            shape,
            radius_fraction=radius_fraction,
        )
        term_to_idx = {term: idx for idx, term in enumerate(base_terms)}
        selected = []
        for term in term_list:
            surface = np.array(base_stack[term_to_idx[term]], copy=True, dtype=float)
            if term == 3 and 5 in term_to_idx:
                surface = surface + base_stack[term_to_idx[5]]
            selected.append(surface)
        return _apply_basis_scale(np.stack(selected, axis=0), basis_scale), base_mask

    stack, mask = basis_term_stack(
        mode,
        term_list,
        shape,
        radius_fraction=radius_fraction,
    )
    return _apply_basis_scale(stack, basis_scale), mask


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
                metadata={"method": "subaper_legacy"},
            )

        global_shape = observations[0].global_shape
        tile_shape = observations[0].tile_shape
        physical_support = observed_support_mask(observations, global_shape)
        basis_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
        mode = _normalize_mode(config.metadata.get("subaper_mode", config.metadata.get("truth_basis", "L")))
        if mode not in {"L", "Z", "LM"}:
            mode = "L"

        radius_fraction = None
        if mode == "Z":
            radius_fraction = float(config.metadata.get("detector_radius_fraction", 0.48))

        basis_scale = config.metadata.get("subaper_indice_carte", config.metadata.get("indice_carte", 1.0))
        basis_stack, basis_mask = _basis_stack_for_mode(
            mode,
            basis_terms,
            tile_shape,
            radius_fraction=radius_fraction,
            basis_scale=basis_scale,
        )

        coeffs = self._solve_global_coefficients(observations, basis_stack, basis_mask)
        z, valid_mask, support_internal, mismatch_map = self._stitch_observations(
            observations=observations,
            coeffs=coeffs,
            basis_stack=basis_stack,
            global_shape=global_shape,
        )
        raw_z = np.array(z, copy=True)
        z = self._detrend_global_map(
            z,
            valid_mask,
            mode,
            basis_terms,
            radius_fraction=radius_fraction,
            basis_scale=basis_scale,
        )

        mismatch_rms = float(np.sqrt(np.nanmean(np.square(mismatch_map)))) if np.any(np.isfinite(mismatch_map)) else 0.0

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=physical_support,
            metadata={
                "method": "subaper_legacy_mlr",
                "basis_mode": mode,
                "basis_terms": basis_terms,
                "mismatch_rms": mismatch_rms,
                "raw_global_map": raw_z,
                "instrument_calibration": None,
            },
        )

    def _solve_global_coefficients(
        self,
        observations: tuple[SubApertureObservation, ...],
        basis_stack: np.ndarray,
        basis_mask: np.ndarray,
    ) -> np.ndarray:
        n_obs = len(observations)
        n_coeffs = basis_stack.shape[0]
        coeffs = np.zeros((n_obs, n_coeffs), dtype=float)
        if n_obs <= 1 or n_coeffs == 0:
            return coeffs

        tile_shape = observations[0].tile_shape
        n_pix = tile_shape[0] * tile_shape[1]
        Por = basis_stack.reshape(n_coeffs, n_pix).T
        pup = np.asarray(basis_mask, dtype=bool).reshape(n_pix)

        T = np.zeros((n_pix, n_obs), dtype=float)
        MSK = np.zeros((n_pix, n_obs), dtype=bool)
        for k, obs in enumerate(observations):
            tile = np.asarray(obs.z, dtype=float).reshape(n_pix)
            mask = np.asarray(obs.valid_mask, dtype=bool).reshape(n_pix)
            T[:, k] = np.where(mask, tile, 0.0)
            MSK[:, k] = mask

        eta = MSK.sum(axis=1)
        if not np.any(eta > 0):
            return coeffs

        dim = (n_obs - 1) * n_coeffs
        M = np.zeros((dim, dim), dtype=float)
        b = np.zeros(dim, dtype=float)
        BigM = np.einsum("pm,pn->mnp", Por, Por)

        for k in range(1, n_obs):
            base = MSK[:, k] & pup
            if not np.any(base):
                continue
            row_base = (k - 1) * n_coeffs
            for m in range(n_coeffs):
                row = row_base + m
                for kkk in range(n_obs):
                    temp = MSK[:, kkk] & base
                    if np.any(temp):
                        eta_temp = eta[temp]
                        b[row] += np.sum((1.0 / eta_temp - float(k == kkk)) * Por[temp, m] * T[temp, kkk])
                for kp in range(1, n_obs):
                    temp = MSK[:, kp] & base
                    if not np.any(temp):
                        continue
                    factor = float(k == kp) - 1.0 / eta[temp]
                    col_base = (kp - 1) * n_coeffs
                    for mp in range(n_coeffs):
                        M[row, col_base + mp] = np.sum(factor * BigM[m, mp, temp])

        try:
            c = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            c, *_ = np.linalg.lstsq(M, b, rcond=None)
        coeffs[1:] = c.reshape(n_obs - 1, n_coeffs)
        return coeffs

    def _stitch_observations(
        self,
        *,
        observations: tuple[SubApertureObservation, ...],
        coeffs: np.ndarray,
        basis_stack: np.ndarray,
        global_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sum_z = np.zeros(global_shape, dtype=float)
        sum_sq = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)

        for obs_idx, obs in enumerate(observations):
            local_obs = np.asarray(obs.z, dtype=float)
            local_mask = np.asarray(obs.valid_mask, dtype=bool)
            if not np.any(local_mask):
                continue

            correction = evaluate_basis_surface(coeffs[obs_idx], basis_stack)
            corrected = local_obs + correction

            placed_values, placed_mask = place_tile_in_global_frame(
                corrected,
                local_mask,
                global_shape,
                obs.center_xy,
            )
            sum_z += placed_values
            sum_sq += placed_values ** 2
            count += placed_mask.astype(float)
            support |= placed_mask

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        mismatch = np.full(global_shape, np.nan, dtype=float)
        mismatch[valid_mask] = np.sqrt(np.maximum(sum_sq[valid_mask] / count[valid_mask] - z[valid_mask] ** 2, 0.0))
        mismatch[mismatch < 0.001] = np.nan
        return z, valid_mask, support, mismatch

    def _detrend_global_map(
        self,
        z: np.ndarray,
        valid_mask: np.ndarray,
        mode: str,
        basis_terms: tuple[int, ...],
        *,
        radius_fraction: float | None,
        basis_scale: object,
    ) -> np.ndarray:
        basis_stack, basis_mask = _basis_stack_for_mode(
            mode,
            basis_terms,
            z.shape,
            radius_fraction=radius_fraction,
            basis_scale=basis_scale,
        )
        fit_mask = valid_mask & basis_mask
        if not np.any(fit_mask):
            z = np.array(z, copy=True, dtype=float)
            z[~valid_mask] = np.nan
            return z

        coeffs = fit_basis_coefficients(z, basis_stack, fit_mask)
        detrended = np.array(z, copy=True, dtype=float)
        detrended[fit_mask] = detrended[fit_mask] - evaluate_basis_surface(coeffs, basis_stack)[fit_mask]
        detrended[~valid_mask] = np.nan
        return detrended
