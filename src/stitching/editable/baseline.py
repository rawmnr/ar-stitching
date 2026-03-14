"""Editable baseline kept intentionally simple during repository foundation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stitching.contracts import SubApertureObservation


def baseline_identity(
    observations: Iterable[SubApertureObservation],
) -> SubApertureObservation:
    """Return the first observation unchanged as a placeholder baseline."""

    first = next(iter(observations))
    return SubApertureObservation(
        observation_id=first.observation_id,
        z=np.array(first.z, copy=True),
        valid_mask=np.array(first.valid_mask, copy=True),
        translation_xy=first.translation_xy,
        rotation_deg=first.rotation_deg,
        reference_bias=first.reference_bias,
        nuisance_terms=dict(first.nuisance_terms),
        metadata=dict(first.metadata),
    )
