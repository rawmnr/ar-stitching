"""Editable stitching implementations."""

from .baseline import baseline_identity, baseline_integer_unshift_mean, baseline_integer_unshift_median

__all__ = ["baseline_integer_unshift_mean", "baseline_integer_unshift_median", "baseline_identity"]
