"""SpinPath is a toolkit for specimen-level inference on whole slide images."""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.unknown"

from spinpath.client.hfmodel import load_torchscript_model_from_hf
from spinpath.client.localmodel import load_torchscript_model_from_filesystem
from spinpath.inference import infer_one_slide

__all__ = [
    "infer_one_slide",
    "load_torchscript_model_from_filesystem",
    "load_torchscript_model_from_hf",
]
