"""WSInfer MIL is a toolkit for specimen-level inference on whole slide images."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.unknown"

from wsinfer_mil.client.hfmodel import load_torchscript_model_from_hf
from wsinfer_mil.client.localmodel import load_torchscript_model_from_filesystem
from wsinfer_mil.inference import infer_one_slide

__all__ = [
    "infer_one_slide",
    "load_torchscript_model_from_filesystem",
    "load_torchscript_model_from_hf",
]


# Patch Zarr. See:
# https://github.com/bayer-science-for-a-better-life/tiffslide/issues/72#issuecomment-1627918238
# https://github.com/zarr-developers/zarr-python/pull/1454
def _patch_zarr_kvstore() -> None:
    from zarr.storage import KVStore

    def _zarr_KVStore___contains__(self, key):  # type: ignore
        return key in self._mutable_mapping

    if "__contains__" not in KVStore.__dict__:
        KVStore.__contains__ = _zarr_KVStore___contains__


_patch_zarr_kvstore()
