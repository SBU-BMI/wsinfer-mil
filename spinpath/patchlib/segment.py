"""Segment thumbnail of a whole slide image."""

from __future__ import annotations

import cv2 as cv
import numpy as np
import numpy.typing as npt
import tiffslide
from skimage.morphology import binary_closing
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects

from spinpath.wsi_utils import get_avg_mpp


def _segment_tissue_from_array(
    im_arr: npt.NDArray,
    median_filter_size: int,
    binary_threshold: int,
    closing_kernel_size: int,
    min_object_size_px: int,
    min_hole_size_px: int,
) -> npt.NDArray[np.bool_]:
    """Create a binary tissue mask from an image.

    Parameters
    ----------
    im_arr : array-like
        RGB image array (uint8) with shape (rows, cols, 3).
    median_filter_size : int
        The kernel size for median filtering. Must be odd and greater than one.
    binary_threshold : int
        The pixel threshold for image binarization.
    closing_kernel_size : int
        The kernel size for morphological closing (in pixel units).
    min_object_size_px : int
        The minimum area of an object in pixels. If an object is smaller than this area,
        it is removed and is made into background.
    min_hole_size_px : int
        The minimum area of a hole in pixels. If a hole is smaller than this area, it is
        filled and is made into foreground.

    Returns
    -------
    mask
        Boolean array, where True values indicate presence of tissue.
    """
    im_arr = np.asarray(im_arr)
    assert im_arr.ndim == 3
    assert im_arr.shape[2] == 3

    # Convert to HSV color space.
    im_arr = cv.cvtColor(im_arr, cv.COLOR_RGB2HSV)
    im_arr = im_arr[:, :, 1]  # Keep saturation channel only.

    # Use median blurring to smooth the image.
    if median_filter_size <= 1 or median_filter_size % 2 == 0:
        raise ValueError(
            "median_filter_size must be greater than 1 and odd, but got"
            f" {median_filter_size}"
        )

    # We use opencv here instead of PIL because opencv is _much_ faster. We use skimage
    # further down for artifact removal (hole filling, object removal) because skimage
    # provides easy to use methods for those.
    im_arr = cv.medianBlur(im_arr, median_filter_size)

    # Binarize image.
    _, im_arr = cv.threshold(
        im_arr, thresh=binary_threshold, maxval=255, type=cv.THRESH_BINARY
    )

    # Convert to boolean dtype. This helps with static type analysis because at this
    # point, im_arr is a uint8 array.
    im_arr_binary: npt.NDArray[np.bool_] = im_arr > 0

    # Closing. This removes small holes. It might not be entirely necessary because
    # we have hole removal below.
    im_arr_binary = binary_closing(
        im_arr_binary, footprint=np.ones((closing_kernel_size, closing_kernel_size))
    )

    # Remove small objects.
    im_arr_binary = remove_small_objects(im_arr_binary, min_size=min_object_size_px)

    # Remove small holes.
    im_arr_binary = remove_small_holes(im_arr_binary, area_threshold=min_hole_size_px)

    return im_arr_binary


def segment_tissue(
    tslide: tiffslide.TiffSlide,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_um2: float = 200**2,
    min_hole_size_um2: float = 190**2,
) -> npt.NDArray[np.bool_]:
    """Get non-overlapping patch coordinates in tissue regions of a whole slide image.

    Patch coordinates are saved to an HDF5 file in `{save_dir}/patches/`, and a tissue
    detection image is saved to `{save_dir}/masks/` for quality control.

    In general, this function takes the following steps:

    1. Get a low-resolution thumbnail of the image.
    2. Binarize the image to identify tissue regions.
    3. Process this binary image to remove artifacts.
    4. Create a regular grid of non-overlapping patches of specified size.
    5. Keep patches whose centroids are in tissue regions.

    Parameters
    ----------
    tslide : tiffslide.TiffSlide
        Whole slide image object.
    thumbsize : tuple of two integers
        The size of the thumbnail to use for tissue detection. This specifies the
        largest possible bounding box of the thumbnail, and a thumbnail is taken to fit
        this space while maintaining the original aspect ratio of the whole slide image.
        Larger thumbnails will take longer to process but will result in better tissue
        masks.
    median_filter_size : int
        The size of the kernel for median filtering. This value must be odd and greater
        than one. This is in units of pixels in the thumbnail.
    binary_threshold: int
        The value at which the image in binarized. A higher value will keep less tissue.
    closing_kernel_size : int
        The size of the kernel for a morphological closing operation. This is in units
        of pixels in the thumbnail.
    min_object_size_um2 : float
        The minimum area of an object to keep, in units of micrometers squared. Any
        disconnected objects smaller than this area will be removed.
    min_hole_size_um2 : float
        The minimum size of a hole to keep, in units of micrometers squared. Any hole
        smaller than this area will be filled and be considered tissue.

    Returns
    -------
    array, binary mask of tissue, where True indicates presence of tissue.
    """
    mpp = get_avg_mpp(tslide)
    thumb = tslide.get_thumbnail(thumbsize)

    # thumb has ~12 MPP.
    # (pixels^2 / micron^2) * micron^2 = pixels^2
    thumb_mpp = (mpp * (np.array(tslide.dimensions) / thumb.size)).mean()
    thumb_mpp_squared: float = thumb_mpp**2
    min_object_size_px: int = round(min_object_size_um2 / thumb_mpp_squared)
    min_hole_size_px: int = round(min_hole_size_um2 / thumb_mpp_squared)

    return _segment_tissue_from_array(
        np.asarray(thumb),
        median_filter_size=median_filter_size,
        binary_threshold=binary_threshold,
        closing_kernel_size=closing_kernel_size,
        min_object_size_px=min_object_size_px,
        min_hole_size_px=min_hole_size_px,
    )
