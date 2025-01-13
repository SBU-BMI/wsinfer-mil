from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def write_patch_coords(
    path: str | Path,
    coords: npt.NDArray[np.int_],
    patch_size_um: float,
    compression: str | None = "gzip",
) -> None:
    """Write patch coordinates to HDF5 file.

    This is designed to be interoperable with HDF5 files created by CLAM.

    Parameters
    ----------
    path : str or Path
        Path to save the HDF5 file.
    coords : array
        Nx4 array of coordinates, where N is the number of patches. Each row of the
        array must be [minx, miny, width, height].
    patch_size_um : int
        The size of patches in micrometers.
    compression: str, optional
        Compression to use for storing coordinates. Default is "gzip".

    Returns
    -------
    None
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must have 2 dimensions but got {coords.ndim}")
    if coords.shape[1] != 4:
        raise ValueError(f"length of second axis must be 4 but got {coords.shape[1]}")

    # Get the patch size in pixels.
    wh = coords[:, 2:]
    unique_sizes = np.unique(wh)
    if not unique_sizes.size == 1:
        raise ValueError(f"Found multiple values for width and height: {unique_sizes}")
    patch_size_px = unique_sizes.item()

    coords_only_minx_miny = coords[:, :2]
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(
            "/coords",
            data=coords_only_minx_miny,
            compression=compression,
        )
        dset.attrs["patch_size_px"] = patch_size_px
        dset.attrs["patch_size_um"] = patch_size_um
        dset.attrs["patch_level"] = 0


def read_patch_coords(path: str | Path) -> npt.NDArray[np.int_]:
    """Read HDF5 file of patch coordinates as a numpy array.

    Parameters
    ----------
    path : str or Path
        Path to save HDF5 file.

    Returns
    -------
    Array of int-like dtype and shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    with h5py.File(path) as f:
        coords: npt.NDArray[np.int_] = f["/coords"][:]
        if not np.issubdtype(coords.dtype, np.integer):
            raise TypeError(
                f"Dtype of coordinates should be integer-like, but got {coords.dtype}"
            )
        coords_metadata = f["/coords"].attrs
        if "patch_level" not in coords_metadata.keys():
            raise KeyError(
                "Could not find required key 'patch_level' in hdf5 of patch "
                "coordinates. Has the version of CLAM been updated?"
            )
        patch_level = coords_metadata["patch_level"]
        if patch_level != 0:
            raise NotImplementedError(
                f"This function is designed for patch_level=0 but got {patch_level}"
            )
        if coords.ndim != 2:
            raise ValueError(f"expected coords to have 2 dimensions, got {coords.ndim}")
        if coords.shape[1] != 2:
            raise ValueError(
                f"expected second dim of coords to have len 2 but got {coords.shape[1]}"
            )

        # Append width and height values to the coords, so now each row is
        # [minx, miny, width, height]
        if "patch_size_px" not in coords_metadata.keys():
            raise KeyError("expected key 'patch_size_px' in attrs of coords dataset")
        patch_size = coords_metadata["patch_size_px"]
        wh = np.full_like(coords, patch_size)
        coords = np.concatenate((coords, wh), axis=1)
        assert np.issubdtype(coords.dtype, np.integer)
    return coords
