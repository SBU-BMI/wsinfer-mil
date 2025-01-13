"""Read patches from whole slide images."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from tiffslide import TiffSlide
from torch.utils.data import Dataset


class WSIPatches(Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to npy file with coordinates of input image.
    um_px : float
        Scale of the resulting patches. For example, 0.5 for ~20x magnification.
    patch_size : int
        The size of patches in pixels.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        patch_coordindates: npt.NDArray[np.int_],
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        self.wsi_path = wsi_path
        self.patch_coordindates = patch_coordindates
        self.transform = transform

        if not Path(wsi_path).exists():
            raise FileNotFoundError(f"WSI path not found: {wsi_path}")

        assert self.patch_coordindates.ndim == 2, (
            "expected 2D array of patch coordinates"
        )
        # x, y, width, height
        assert self.patch_coordindates.shape[1] == 4, (
            "expected second dimension to have len 4"
        )

    def worker_init(self, worker_id: int | None = None) -> None:
        del worker_id
        self.slide = TiffSlide(self.wsi_path)

    def __len__(self) -> int:
        return len(self.patch_coordindates)

    def __getitem__(self, idx: int) -> torch.Tensor:
        coords: Sequence[int] = self.patch_coordindates[idx]
        assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
        minx, miny, width, height = coords
        patch_im: Image.Image = self.slide.read_region(
            location=(minx, miny), level=0, size=(width, height)
        )
        patch_im = patch_im.convert("RGB")
        patch_tensor = self.transform(patch_im)

        return patch_tensor
