"""Legacy Subaper port.

This keeps the original structure intentionally simple: fit low-order per-tile
alignment terms, subtract them, and fuse the corrected sub-apertures.
"""
from __future__ import annotations

import numpy as np
from stitching.contracts import ReconstructionSurface, ScenarioConfig, SubApertureObservation
from stitching.editable._legacy_basis import (
    basis_term_stack,
    evaluate_basis_surface,
    fit_basis_coefficients,
    observed_support_mask,
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
                metadata={"method": "subaper_legacy"},
            )

        global_shape = observations[0].global_shape
        basis_terms = tuple(int(v) for v in config.metadata.get("alignment_term", (0, 1, 2)))
        mode = str(config.metadata.get("subaper_mode", config.metadata.get("truth_basis", "L"))).upper()
        if mode not in {"L", "Z"}:
            mode = "L"
        radius_fraction = None
        if mode == "Z":
            radius_fraction = float(config.metadata.get("detector_radius_fraction", 0.48))

        basis_stack, basis_mask = basis_term_stack(mode, basis_terms, observations[0].tile_shape, radius_fraction=radius_fraction)
        global_map = np.zeros(global_shape, dtype=float)
        final_coeffs = np.zeros((len(observations), len(basis_terms)), dtype=float)

        for _ in range(3):
            sum_z = np.zeros(global_shape, dtype=float)
            count = np.zeros(global_shape, dtype=float)
            support = np.zeros(global_shape, dtype=bool)
            coeffs = np.zeros((len(observations), len(basis_terms)), dtype=float)

            for i, obs in enumerate(observations):
                gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
                if gy.stop <= gy.start or gx.stop <= gx.start:
                    continue

                local_obs = np.asarray(obs.z, dtype=float)[ly, lx]
                local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
                local_basis_stack = basis_stack[:, ly, lx]
                local_basis_mask = basis_mask[ly, lx]
                fit_mask = local_mask & local_basis_mask
                if not np.any(local_mask):
                    continue

                local_global = np.zeros_like(local_obs, dtype=float)
                local_global[fit_mask] = global_map[gy, gx][fit_mask]
                residual = local_obs - local_global
                coeffs[i] = fit_basis_coefficients(residual, local_basis_stack, fit_mask)
                correction = evaluate_basis_surface(coeffs[i], local_basis_stack)
                corrected = local_obs - correction

                global_view_sum = sum_z[gy, gx]
                global_view_count = count[gy, gx]
                global_view_support = support[gy, gx]
                global_view_sum[local_mask] += corrected[local_mask]
                global_view_count[local_mask] += 1.0
                global_view_support[local_mask] = True

            valid_mask = count > 0
            updated_map = np.full(global_shape, np.nan, dtype=float)
            updated_map[valid_mask] = sum_z[valid_mask] / count[valid_mask]

            if np.any(np.isfinite(global_map)) and np.nanmax(np.abs(np.nan_to_num(updated_map - global_map, nan=0.0))) < 1e-6:
                global_map = updated_map
                final_coeffs = coeffs
                break
            global_map = updated_map
            final_coeffs = coeffs

        z, valid_mask, support, mismatch_rms = self._final_fusion(
            observations=observations,
            global_shape=global_shape,
            coeffs=final_coeffs,
            mode=mode,
            basis_terms=basis_terms,
            radius_fraction=radius_fraction,
            global_map=global_map,
        )

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(o.observation_id for o in observations),
            observed_support_mask=support,
            metadata={
                "method": "subaper_legacy",
                "basis_mode": mode,
                "basis_terms": basis_terms,
                "mismatch_rms": mismatch_rms,
            },
        )

    def _final_fusion(
        self,
        observations: tuple[SubApertureObservation, ...],
        global_shape: tuple[int, int],
        coeffs: np.ndarray,
        mode: str,
        basis_terms: tuple[int, ...],
        radius_fraction: float | None,
        global_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        sum_z = np.zeros(global_shape, dtype=float)
        count = np.zeros(global_shape, dtype=float)
        support = np.zeros(global_shape, dtype=bool)
        mismatch_stack = []
        basis_stack, basis_mask = basis_term_stack(mode, basis_terms, observations[0].tile_shape, radius_fraction=radius_fraction)

        for i, obs in enumerate(observations):
            gy, gx, ly, lx = rounded_placement_slices(global_shape, obs.tile_shape, obs.center_xy)
            if gy.stop <= gy.start or gx.stop <= gx.start:
                continue

            local_obs = np.asarray(obs.z, dtype=float)[ly, lx]
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
            local_basis_stack = basis_stack[:, ly, lx]
            local_basis_mask = basis_mask[ly, lx]
            fit_mask = local_mask & local_basis_mask
            if not np.any(local_mask):
                continue

            local_global = np.zeros_like(local_obs, dtype=float)
            local_global[local_mask] = np.nan_to_num(global_map[gy, gx][local_mask], nan=0.0)
            correction = evaluate_basis_surface(coeffs[i], local_basis_stack)
            corrected = local_obs - correction

            global_view_sum = sum_z[gy, gx]
            global_view_count = count[gy, gx]
            global_view_support = support[gy, gx]
            global_view_sum[local_mask] += corrected[local_mask]
            global_view_count[local_mask] += 1.0
            global_view_support[local_mask] = True

            mismatch_stack.append((corrected - local_global)[local_mask])

        valid_mask = count > 0
        z = np.full(global_shape, np.nan, dtype=float)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]
        mismatch_rms = float(np.sqrt(np.mean(np.square(np.concatenate(mismatch_stack))))) if mismatch_stack else 0.0
        return z, valid_mask, support, mismatch_rms
