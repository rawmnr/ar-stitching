import numpy as np

from stitching.trusted.surface.footprint import circular_pupil_mask


def test_centered_circular_footprint_is_symmetric_and_centered() -> None:
    mask = circular_pupil_mask((9, 9), radius_fraction=0.34)

    assert mask[4, 4]
    assert np.array_equal(mask, np.flipud(mask))
    assert np.array_equal(mask, np.fliplr(mask))
    assert 0 < mask.sum() < mask.size
