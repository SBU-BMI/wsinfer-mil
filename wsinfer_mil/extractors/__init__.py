from .base import PatchFeatureExtractor
from .ctranspath import CTransPath
from .uni import UNI

EXTRACTORS: dict[str, type[PatchFeatureExtractor]] = {
    "ctranspath": CTransPath,
    "uni": UNI,
}


def get_extractor_by_name(name: str) -> type[PatchFeatureExtractor]:
    if name not in EXTRACTORS:
        keys = ", ".join(EXTRACTORS.keys())
        raise KeyError(f"unknown extractor: '{name}'. Options are {keys}")
    return EXTRACTORS[name]
