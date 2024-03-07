"""Hash parts of a whole slide image.

This implementation is heavily inspired by OpenSlide's quickhash1:
https://github.com/openslide/openslide/blob/549e81b6662efe2b2285f11a5bcb31ccd7b95655/src/openslide-decode-tifflike.c#L996-L1143
"""

from __future__ import annotations

import hashlib

import tiffslide
from PIL import Image
from tiffslide.tiffslide import PROPERTY_NAME_COMMENT
from tiffslide.tiffslide import PROPERTY_NAME_VENDOR


def _read_smallest_level(tslide: tiffslide.TiffSlide) -> Image.Image:
    smallest_level = tslide.level_count - 1
    size = tslide.level_dimensions[smallest_level]
    return tslide.read_region((0, 0), level=smallest_level, size=size)


def _hash_str_and_property(
    hasher: hashlib._Hash, tslide: tiffslide.TiffSlide, name: str
) -> None:
    value = tslide.properties.get(name)
    if value is not None:
        hasher.update(name.encode())
        hasher.update(str(value).encode())


def quickhash(tslide: tiffslide.TiffSlide) -> str:
    """Return a quick MD5 hash of a whole slide image."""
    m = hashlib.md5()
    _hash_str_and_property(m, tslide, PROPERTY_NAME_COMMENT)
    _hash_str_and_property(m, tslide, PROPERTY_NAME_VENDOR)
    smallest_level_bytes = _read_smallest_level(tslide).tobytes()
    m.update(smallest_level_bytes)
    return m.hexdigest()
