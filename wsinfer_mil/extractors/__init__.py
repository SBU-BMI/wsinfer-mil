from .base import PatchFeatureExtractor
from .ctranspath import CTransPath
from .hoptimus0 import HOptimus0
from .uni import UNI
from .virchow2 import Virchow2

EXTRACTORS: dict[str, type[PatchFeatureExtractor]] = {
    "ctranspath": CTransPath,
    "hoptimus0": HOptimus0,
    "uni": UNI,
    "virchow2": Virchow2,
}


def get_extractor_by_name(name: str) -> type[PatchFeatureExtractor]:
    if name not in EXTRACTORS:
        keys = ", ".join(EXTRACTORS.keys())
        raise KeyError(f"unknown extractor: '{name}'. Options are {keys}")
    return EXTRACTORS[name]
