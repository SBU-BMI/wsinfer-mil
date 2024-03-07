"""Cache for extracted features.

The features extracted from a slide may be reused for multiple MIL models.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from wsinfer_mil.defaults import WSINFER_MIL_CACHE_DIR
from wsinfer_mil.extractors import PatchFeatureExtractor
from wsinfer_mil.patchlib import read_patch_coords
from wsinfer_mil.patchlib import write_patch_coords

TISSUE_MASK_FILENAME = "tissue.png"
PATCH_COORDS_FILENAME = "patches.h5"

logger = logging.getLogger(__name__)


def _get_embedding_filename(extractor: PatchFeatureExtractor) -> str:
    return f"{extractor.name}.npy"


def log_setter(func):  # type: ignore
    """Wrapper to log the status of a cache 'get_*' method."""

    @functools.wraps(func)
    def wrapper(self: Cache, *args, **kwargs):  # type: ignore
        key = func.__name__[4:]  # trim off 'set_'
        logger.debug(f"Attempting to set {key} in {self.slide_cache_dir}")
        result = func(self, *args, **kwargs)
        logger.debug(f"Set {key}")
        return result

    return wrapper


def log_getter(func):  # type: ignore
    """Wrapper to log the status of a cache 'get_*' method."""

    @functools.wraps(func)
    def wrapper(self: Cache, *args, **kwargs):  # type: ignore
        key = func.__name__[4:]  # trim off 'get_'
        logger.debug(f"Attempting to get {key} from {self.slide_cache_dir}")
        result = func(self, *args, **kwargs)
        if result is None:
            logger.debug(f"No entry found for {key}")
        else:
            logger.debug(f"Found entry for {key}")
        return result

    return wrapper


class Cache:
    """Cache for slide tissue masks, patch coordinates, and embeddings."""

    def __init__(
        self,
        slide_path: str | Path,
        slide_quickhash: str,
        patch_size_um: float,
        cache_dir: str | None = None,
    ) -> None:
        self.slide_path = slide_path
        self.slide_quickhash = slide_quickhash
        self.patch_size_um = patch_size_um
        if cache_dir is None:
            self.cache_dir = WSINFER_MIL_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)

        self.slide_name = Path(slide_path).name

        logger.debug(
            f"Instantiating cache object for slide {self.slide_name}"
            f" with patches of size {self.patch_size_um} microns."
        )
        logger.debug(f"Cache directory is {self.slide_cache_dir}")
        if self.slide_cache_dir.exists():
            logger.debug("Cache directory exists")

    @property
    def slide_cache_dir(self) -> Path:
        slide_cache_dir = (
            self.cache_dir
            / f"{self.patch_size_um}um"
            / f"{self.slide_name}_md5-{self.slide_quickhash}"
        )
        return slide_cache_dir

    @log_setter
    def set_tissue_mask(self, tissue_mask: Image.Image) -> None:
        if not isinstance(tissue_mask, Image.Image):
            raise TypeError(f"tissue_mask must be Image but got {type(tissue_mask)}")
        path = self.slide_cache_dir / TISSUE_MASK_FILENAME
        path.parent.mkdir(exist_ok=True, parents=True)
        tissue_mask.save(path)

    @log_getter
    def get_tissue_mask(self) -> Image.Image | None:
        path = self.slide_cache_dir / TISSUE_MASK_FILENAME
        if path.exists():
            # Open the image in this way to close the file handle.
            with Image.open(path) as img:
                img.load()
                return img
        return None

    @log_setter
    def set_patch_coordinates(
        self,
        patch_coordinates: npt.NDArray[np.int_],
        patch_size_um: float,
    ) -> None:
        """Set patch coordinates.

        Parameters
        ----------
        patch_coordinates : array
            A Nx2 array, where each row contains [minx, miny.]
        patch_spacing_um_px : float
            The physical spacing of one pixel in micrometers per pixel.

        Returns
        -------
        None
        """
        if not isinstance(patch_coordinates, np.ndarray):
            raise TypeError(
                f"patch_coordinates must be Image but got {type(patch_coordinates)}"
            )
        if not np.issubdtype(patch_coordinates.dtype, np.int_):
            raise TypeError(f"must be int dtype but got {patch_coordinates.dtype}")
        path = self.slide_cache_dir / PATCH_COORDS_FILENAME
        path.parent.mkdir(exist_ok=True, parents=True)
        write_patch_coords(
            path=path,
            coords=patch_coordinates,
            patch_size_um=patch_size_um,
            compression="gzip",
        )

    @log_getter
    def get_patch_coordinates(self) -> npt.NDArray[np.int_] | None:
        """Read a Nx4 array of patch coordinates.

        Each row is [minx, miny, width, height] of the patch.
        """
        path = self.slide_cache_dir / PATCH_COORDS_FILENAME
        if path.exists():
            return read_patch_coords(path)
        return None

    @log_setter
    def set_embedding(
        self,
        extractor: PatchFeatureExtractor,
        embedding: npt.NDArray[np.float32],
    ) -> None:
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"embedding must be a numpy array, got {type(embedding)}")
        if embedding.dtype != np.float32:
            raise TypeError(
                f"dtype of embedding must be float32 but got {embedding.dtype}"
            )
        filename = _get_embedding_filename(extractor)
        path = self.slide_cache_dir / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        np.save(path, embedding)

    @log_getter
    def get_embedding(
        self, extractor: PatchFeatureExtractor
    ) -> npt.NDArray[np.float32] | None:
        filename = _get_embedding_filename(extractor)
        path = self.slide_cache_dir / filename
        if path.exists():
            res = np.load(path)
            assert isinstance(res, np.ndarray)
            assert res.dtype == np.float32
            return res
        return None
