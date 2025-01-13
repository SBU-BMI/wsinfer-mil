from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir

# Where we keep all files related to WSInfer MIL.
SPINPATH_DIR = Path(user_cache_dir(appname="wsinfer-mil"))

# Cache for tissue masks, patch coordinates, and feature embeddings.
SPINPATH_CACHE_DIR = SPINPATH_DIR / "cache"

# JSON file with list of registered WSInfer MIL models.
SPINPATH_REGISTRY_PATH = SPINPATH_DIR / "registry.json"
