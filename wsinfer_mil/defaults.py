from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir

# Where we keep all files related to WSInfer MIL.
WSINFER_MIL_DIR = Path(user_cache_dir(appname="wsinfer-mil"))

# Cache for tissue masks, patch coordinates, and feature embeddings.
WSINFER_MIL_CACHE_DIR = WSINFER_MIL_DIR / "cache"

# JSON file with list of registered WSInfer MIL models.
WSINFER_MIL_REGISTRY_PATH = WSINFER_MIL_DIR / "registry.json"
