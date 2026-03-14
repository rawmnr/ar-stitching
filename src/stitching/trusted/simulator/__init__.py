"""Simulation orchestrators that combine surfaces, masks, transforms, and nuisance terms."""

from .identity import simulate_identity_observations

__all__ = ["simulate_identity_observations"]
