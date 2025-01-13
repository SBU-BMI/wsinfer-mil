from __future__ import annotations

import abc
import logging
import os
from functools import cached_property

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _find_fastest_device() -> str:
    if os.getenv("SPINPATH_FORCE_CPU", "false").lower() in ("true", "1", "t"):
        return "cpu"
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return "mps"
    elif torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"

    return "cpu"


class PatchFeatureExtractor(abc.ABC):
    """Patch feature extraction base class.

    Parameters
    ----------
    allow_cpu : bool
        If True, allow inference on CPU. If False (default),
        raise a `RuntimeError` if a GPU is not available.
    """

    def __init__(self) -> None:
        logger.debug("Finding best device for PyTorch inference...")
        self.device = torch.device(_find_fastest_device())
        logger.debug(f"PyTorch device: {self.device}")
        if self.device == "cpu":
            logger.warning("Using CPU for inference. This may be sloooow.")

    @abc.abstractmethod
    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def transform(self) -> transforms.Compose:
        raise NotImplementedError()

    @cached_property
    def model(self) -> torch.nn.Module:
        return self.load_model()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    def get_batch_embeddings(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        t: torch.Tensor = self.model(batch)
        return t.detach().cpu().numpy()

    def run(self, loader: DataLoader, progbar: bool = True) -> npt.NDArray[np.float32]:
        """Get embeddings from a loader of patches."""
        patches: torch.Tensor
        embeddings: list[npt.NDArray[np.float32]] = []
        with torch.inference_mode():
            for patches in tqdm(
                loader, desc="Embedding patches", unit="batch", disable=not progbar
            ):
                patches = patches.to(self.device)
                e = self.get_batch_embeddings(patches)
                embeddings.append(e)
        embeddings_np = np.concatenate(embeddings)
        assert embeddings_np.dtype == np.float32
        assert embeddings_np.ndim == 2
        return embeddings_np
