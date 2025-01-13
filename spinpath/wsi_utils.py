from __future__ import annotations

import tiffslide
from tiffslide.tiffslide import PROPERTY_NAME_MPP_X
from tiffslide.tiffslide import PROPERTY_NAME_MPP_Y


def get_avg_mpp(tslide: tiffslide.TiffSlide) -> float:
    """Return the average slide spacing in microns per pixel."""
    mpp_x: float | None = tslide.properties[PROPERTY_NAME_MPP_X]
    mpp_y: float | None = tslide.properties[PROPERTY_NAME_MPP_Y]
    if mpp_x is None:
        raise ValueError("MPP-X is None")
    if mpp_y is None:
        raise ValueError("MPP-Y is None")
    return (mpp_x + mpp_y) / 2
