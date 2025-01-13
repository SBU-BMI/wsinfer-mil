"""Cache for extracted features.

The features extracted from a slide may be reused for multiple MIL models.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from spinpath.defaults import SPINPATH_CACHE_DIR

logger = logging.getLogger(__name__)


def _hash_image(image: Image.Image) -> str:
    """Generate a hash for an image."""
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()


def _hash_array(array: npt.NDArray) -> str:
    """Generate a hash for a numpy array."""
    array_bytes = array.tobytes()
    return hashlib.md5(array_bytes).hexdigest()


class EmbeddingsCache:
    """Cache for slide embeddings."""

    def __init__(
        self,
        slide_path: str | Path,
        *,
        slide_quickhash: str,
        tissue_mask: Image.Image,
        patch_coordinates: npt.NDArray,
        embedding_model_name: str,
        cache_dir: Path | None = None,
    ) -> None:
        self.slide_path = slide_path
        self.slide_quickhash = slide_quickhash
        # self.patch_size_um = patch_size_um
        if cache_dir is None:
            self.cache_dir = SPINPATH_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)

        self.slide_name = Path(slide_path).name

        hash_parts = [
            slide_quickhash,
            _hash_image(tissue_mask),
            _hash_array(patch_coordinates),
            embedding_model_name,
        ]
        combined_string = "_".join(hash_parts)
        self.cache_key = hashlib.md5(combined_string.encode("utf-8")).hexdigest()
        self.embedding_filename = self.cache_dir / f"{self.cache_key}.npy"

    def save(self, embedding: npt.NDArray[np.float32]) -> None:
        logger.debug(
            f"[Cache: {self.cache_key}] Saving embedding to {self.embedding_filename}"
        )
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"embedding must be a numpy array, got {type(embedding)}")
        if embedding.dtype != np.float32:
            raise TypeError(
                f"dtype of embedding must be float32 but got {embedding.dtype}"
            )
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be 2D, got {embedding.ndim}")
        self.embedding_filename.parent.mkdir(exist_ok=True, parents=True)
        np.save(self.embedding_filename, embedding)
        self._embedding = embedding

    def load(self) -> npt.NDArray[np.float32] | None:
        logger.debug(
            f"[Cache: {self.cache_key}] Attempting to load embedding {self.embedding_filename}"
        )
        if self.embedding_filename.exists():
            logger.debug(f"[Cache: {self.cache_key}] Loading {self.embedding_filename}")
            res = np.load(self.embedding_filename)
            assert isinstance(res, np.ndarray)
            assert res.dtype == np.float32
            assert res.ndim == 2
            return res
        logger.debug(
            f"[Cache: {self.cache_key}] Embedding does not exist for {self.embedding_filename}"
        )
        return None
