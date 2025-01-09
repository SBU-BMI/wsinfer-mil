from .base import PatchFeatureExtractor
from .ctranspath import CTransPath
from .hibou import HibouL
from .hoptimus0 import HOptimus0
from .kaiko import KaikoL
from .phikon import Phikon
from .phikonv2 import PhikonV2
from .provgigapath import ProvGigaPath
from .uni import UNI
from .virchow import Virchow
from .virchow2 import Virchow2

EXTRACTORS: dict[str, type[PatchFeatureExtractor]] = {
    "ctranspath": CTransPath,
    "hiboul": HibouL,
    "hoptimus0": HOptimus0,
    "kaikol": KaikoL,
    "phikon": Phikon,
    "phikonv2": PhikonV2,
    "provgigapath": ProvGigaPath,
    "uni": UNI,
    "virchow": Virchow,
    "virchow2": Virchow2,
}


def get_extractor_by_name(name: str) -> type[PatchFeatureExtractor]:
    if name not in EXTRACTORS:
        keys = ", ".join(EXTRACTORS.keys())
        raise KeyError(f"unknown extractor: '{name}'. Options are {keys}")
    return EXTRACTORS[name]
