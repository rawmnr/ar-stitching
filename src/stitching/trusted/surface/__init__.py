"""Surface generation and footprint primitives."""

from .footprint import circular_pupil_mask
from .generation import generate_identity_surface

__all__ = ["circular_pupil_mask", "generate_identity_surface"]
