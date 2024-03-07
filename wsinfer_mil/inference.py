"""Main interface to WSInfer MIL.

1. Read slide.
2. Check cache for tissue segmentation.
2. Segment tissue.
3. Create patches.
4. Check cache for embeddings.
5. If it doesn't exist, run patches through feature extractor and save to cache.
6. Run MIL on extracted features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiffslide
import torch
from PIL import Image
from torch.utils.data import DataLoader

from wsinfer_mil.cache import Cache
from wsinfer_mil.client.localmodel import Model
from wsinfer_mil.data import WSIPatches
from wsinfer_mil.extractors import get_extractor_by_name
from wsinfer_mil.output_container import ModelInferenceOutput
from wsinfer_mil.patchlib.patch import patch_tissue
from wsinfer_mil.patchlib.segment import segment_tissue
from wsinfer_mil.quickhash import quickhash

logger = logging.getLogger(__name__)


def get_model_outputs(
    model: torch.nn.Module, embedding: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    with torch.inference_mode():
        logits, attention = model(torch.from_numpy(embedding))
    logits = logits.detach().cpu()
    attention = attention.detach().cpu().numpy()
    softmax_probs = logits.softmax(1).numpy()
    logits = logits.numpy()
    return logits, softmax_probs, attention


def infer_one_slide(
    slide_path: str | Path,
    model: Model,
    tissue_mask: Image.Image | None,
    num_workers: int = 0,
) -> ModelInferenceOutput:
    """Run MIL inference on one slide.

    Parameters
    ----------
    """

    tslide = tiffslide.TiffSlide(slide_path)

    slide_quickhash = quickhash(tslide)

    cache = Cache(
        slide_path=slide_path,
        slide_quickhash=slide_quickhash,
        patch_size_um=model.config.patch_size_um,
    )

    # Segment tissue.
    if tissue_mask is None:
        tissue_mask = cache.get_tissue_mask()
        if tissue_mask is None:
            tissue_mask_arr = segment_tissue(
                tslide=tslide,
                thumbsize=(2048, 2048),
                median_filter_size=7,
                binary_threshold=7,
                closing_kernel_size=6,
                min_object_size_um2=200**2,
                min_hole_size_um2=190**2,
            )
            tissue_mask = Image.fromarray(tissue_mask_arr).convert("1")
            cache.set_tissue_mask(tissue_mask)
    else:
        tissue_mask = tissue_mask.convert("1")
        cache.set_tissue_mask(tissue_mask)

    tissue_mask_aspect = tissue_mask.size[0] / tissue_mask.size[1]
    slide_aspect = tslide.dimensions[0] / tslide.dimensions[1]
    if not np.isclose(slide_aspect, tissue_mask_aspect, atol=1e-2):
        raise ValueError(
            "aspect of slide and tissue mask are not close:"
            f" {slide_aspect}, {tissue_mask_aspect}"
        )

    # Get patch coordinates.
    # Nx4 coords (minx, miny, width, height)
    coords = cache.get_patch_coordinates()
    if coords is None:
        binary_tissue_mask = np.asarray(tissue_mask) > 0
        coords = patch_tissue(
            tslide,
            binary_tissue_mask=binary_tissue_mask,
            patch_size_um=model.config.patch_size_um,
        )
        cache.set_patch_coordinates(
            patch_coordinates=coords, patch_size_um=model.config.patch_size_um
        )

    # Run through feature extractor.
    extractor_contructor = get_extractor_by_name(model.config.feature_extractor)
    extractor = extractor_contructor()
    embedding = cache.get_embedding(extractor)
    if embedding is None:
        dataset = WSIPatches(
            wsi_path=slide_path,
            patch_coordindates=coords,
            transform=extractor.transform,
        )
        # The worker_init_fn does not seem to be used when num_workers=0
        # so we call it manually to finish setting up the dataset.
        if num_workers == 0:
            dataset.worker_init()
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=dataset.worker_init,
        )
        embedding = extractor.get_embeddings(loader)
        cache.set_embedding(extractor, embedding)

    model_jit = torch.jit.load(model.model_path, map_location="cpu")
    if not isinstance(model_jit, torch.nn.Module):
        raise TypeError(
            f"expected loaded model to be torch.nn.Module but got {type(model_jit)}"
        )
    logits, softmax_probs, attention = get_model_outputs(model_jit, embedding)
    output = ModelInferenceOutput(
        logits=logits,
        softmax_probs=softmax_probs,
        attention=attention,
        class_names=model.config.class_names,
        patch_coordinates=coords,
    )
    return output
