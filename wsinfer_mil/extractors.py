from __future__ import annotations

import abc
import logging
from functools import cached_property

import huggingface_hub
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PatchFeatureExtractor(abc.ABC):
    """Patch feature extraction base class.

    Parameters
    ----------
    allow_cpu : bool
        If True, allow inference on CPU. If False (default),
        raise a `RuntimeError` if a GPU is not available.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            logger.warn("GPU is not available! Falling back to (much much slower) CPU")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.debug(f"PyTorch device: {self.device}")

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

    def get_embeddings(
        self, loader: DataLoader, progbar: bool = True
    ) -> npt.NDArray[np.float32]:
        """Get embeddings from a loader of patches."""
        patches: torch.Tensor
        embeddings: list[npt.NDArray[np.float32]] = []
        with torch.inference_mode():
            for patches in tqdm(
                loader, desc="Embedding patches", unit="batch", disable=not progbar
            ):
                patches = patches.to(self.device)
                e = self.model(patches).detach().cpu().numpy()
                embeddings.append(e)
        embeddings_np = np.concatenate(embeddings)
        assert embeddings_np.dtype == np.float32
        assert embeddings_np.ndim == 2
        return embeddings_np


class CTransPath(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "ctranspath"

    def load_model(self) -> torch.nn.Module:
        model_path = huggingface_hub.hf_hub_download(
            repo_id="kaczmarj/CTransPath", filename="torchscript_model.pt"
        )
        model: torch.nn.Module = torch.jit.load(model_path, map_location="cpu")
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


EXTRACTORS: dict[str, type[PatchFeatureExtractor]] = {
    "ctranspath": CTransPath,
}


def get_extractor_by_name(name: str) -> type[PatchFeatureExtractor]:
    if name not in EXTRACTORS:
        keys = ", ".join(EXTRACTORS.keys())
        raise KeyError(f"unknown extractor: '{name}'. Options are {keys}")
    return EXTRACTORS[name]
