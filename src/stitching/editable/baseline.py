"""Editable baseline kept intentionally simple during repository foundation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stitching.contracts import ReconstructionSurface, SubApertureObservation


def baseline_identity(
    observations: Iterable[SubApertureObservation],
) -> ReconstructionSurface:
    """Return the first observation unchanged as a placeholder baseline."""

    first = next(iter(observations))
    return ReconstructionSurface(
        z=np.array(first.z, copy=True),
        valid_mask=np.array(first.valid_mask, copy=True),
        source_observation_ids=(first.observation_id,),
        metadata={"baseline": "identity", **dict(first.metadata)},
    )
