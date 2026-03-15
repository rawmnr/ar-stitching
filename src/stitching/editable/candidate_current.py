"""Current candidate stitching algorithm – EDITABLE BY AGENT.

This file is the 'genome' that the autoresearch loop optimizes.
It MUST define a `CandidateStitcher` class implementing the
`CandidateAlgorithm` protocol:

    class CandidateStitcher:
        def reconstruct(
            self,
            observations: tuple[SubApertureObservation, ...],
            config: ScenarioConfig,
        ) -> ReconstructionSurface: ...

Starting point: simple mean baseline.
Agent should evolve this toward GLS / CS / SC / Huber / hybrid solutions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from stitching.contracts import (
    ReconstructionSurface,
    ScenarioConfig,
    SubApertureObservation,
)
from stitching.trusted.scan.transforms import placement_slices


class CandidateStitcher:
    """Baseline stitcher: weighted mean with piston removal in overlap zones.

    This is the starting genome for autoresearch optimization.
    The agent should evolve this into progressively more sophisticated algorithms:
    - Phase 1: Piston-corrected mean → reduce DC offsets
    - Phase 2: GLS solver with tip/tilt/piston estimation
    - Phase 3: Robust GLS with Huber M-estimator (IRLS)
    - Phase 4: CS or SC auto-calibration
    - Phase 5: Hybrid CS/SC + robust GLS
    """

    def reconstruct(
        self,
        observations: tuple[SubApertureObservation, ...],
        config: ScenarioConfig,
    ) -> ReconstructionSurface:
        if not observations:
            raise ValueError("No observations provided.")

        global_shape = observations[0].global_shape
        sum_z = np.zeros(global_shape, dtype=np.float64)
        count = np.zeros(global_shape, dtype=np.int32)
        observed_support = np.zeros(global_shape, dtype=bool)
        source_ids: list[str] = []

        # Phase 1: Estimate piston offsets from overlaps
        pistons = self._estimate_pistons(observations, global_shape)

        # Phase 2: Place corrected tiles
        for idx, obs in enumerate(observations):
            gy, gx, ly, lx = placement_slices(
                obs.global_shape, obs.tile_shape, obs.center_xy,
            )
            local_z = np.asarray(obs.z, dtype=np.float64)[ly, lx]
            local_mask = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]

            # Apply piston correction
            corrected_z = local_z - pistons[idx]

            sum_z[gy, gx][local_mask] += corrected_z[local_mask]
            count[gy, gx][local_mask] += 1
            observed_support[gy, gx][local_mask] = True
            source_ids.append(obs.observation_id)

        valid_mask = count > 0
        z = np.zeros(global_shape, dtype=np.float64)
        z[valid_mask] = sum_z[valid_mask] / count[valid_mask]

        return ReconstructionSurface(
            z=z,
            valid_mask=valid_mask,
            source_observation_ids=tuple(source_ids),
            observed_support_mask=observed_support,
            metadata={"algorithm": "piston_corrected_mean", "pistons": pistons.tolist()},
        )

    def _estimate_pistons(
        self,
        observations: tuple[SubApertureObservation, ...],
        global_shape: tuple[int, int],
    ) -> np.ndarray:
        """Estimate per-observation piston offsets from overlap differences.

        Uses a simple least-squares solve on pairwise overlap mean differences:
            For each overlapping pair (i, j): piston_i - piston_j ≈ mean(z_i - z_j)
        Constraint: sum(pistons) = 0
        """
        n = len(observations)
        if n <= 1:
            return np.zeros(n, dtype=np.float64)

        # Place all observations into global frame for overlap detection
        placed: list[tuple[np.ndarray, np.ndarray, slice, slice]] = []
        for obs in observations:
            gy, gx, ly, lx = placement_slices(obs.global_shape, obs.tile_shape, obs.center_xy)
            z_local = np.asarray(obs.z, dtype=np.float64)[ly, lx]
            m_local = np.asarray(obs.valid_mask, dtype=bool)[ly, lx]
            z_full = np.full(global_shape, np.nan, dtype=np.float64)
            m_full = np.zeros(global_shape, dtype=bool)
            z_full[gy, gx][m_local] = z_local[m_local]
            m_full[gy, gx][m_local] = True
            placed.append((z_full, m_full, gy, gx))

        # Build pairwise overlap equations
        rows_A: list[list[float]] = []
        rhs: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                overlap = placed[i][1] & placed[j][1]
                if np.sum(overlap) < 4:
                    continue
                diff_mean = float(np.nanmean(placed[i][0][overlap] - placed[j][0][overlap]))
                row = [0.0] * n
                row[i] = 1.0
                row[j] = -1.0
                rows_A.append(row)
                rhs.append(diff_mean)

        if not rows_A:
            return np.zeros(n, dtype=np.float64)

        # Add constraint: sum(pistons) = 0
        rows_A.append([1.0] * n)
        rhs.append(0.0)

        A = np.array(rows_A, dtype=np.float64)
        b = np.array(rhs, dtype=np.float64)

        # Solve in least-squares sense
        pistons, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pistons
