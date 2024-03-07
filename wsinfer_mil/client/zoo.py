from __future__ import annotations

import dataclasses
import functools
import json
import logging
from pathlib import Path

import jsonschema
from huggingface_hub import hf_hub_download

from wsinfer_mil.client.hfmodel import HFModel
from wsinfer_mil.client.hfmodel import load_torchscript_model_from_hf
from wsinfer_mil.defaults import WSINFER_MIL_REGISTRY_PATH
from wsinfer_mil.errors import InvalidRegistryConfiguration

logger = logging.getLogger(__name__)


def validate_model_zoo_json(instance: object) -> bool:
    """Raise an error if the model zoo registry JSON is invalid. Otherwise return
    True.
    """
    _here = Path(__file__).parent.resolve()
    schema_path = _here / ".." / "schemas" / "wsinfer-mil-zoo-registry.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"JSON schema for wsinfer zoo not found: {schema_path}")
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance, schema=schema)
    except jsonschema.ValidationError as e:
        raise InvalidRegistryConfiguration(
            "Invalid model zoo registry configuration. See traceback above for details."
        ) from e
    return True


@dataclasses.dataclass
class RegisteredModel:
    """Container with information about where to find a single model."""

    name: str
    description: str
    hf_repo_id: str
    hf_revision: str

    def load_model(self) -> HFModel:
        return load_torchscript_model_from_hf(
            repo_id=self.hf_repo_id, revision=self.hf_revision
        )

    def __str__(self) -> str:
        return f"{self.name} ({self.hf_repo_id} @ {self.hf_revision})"


@dataclasses.dataclass
class ModelRegistry:
    """Registry of models that can be used with WSInfer."""

    models: dict[str, RegisteredModel]

    def get_model_by_name(self, name: str) -> RegisteredModel:
        try:
            return self.models[name]
        except KeyError as e:
            raise KeyError(f"model not found with name '{name}'.") from e

    @classmethod
    def from_dict(cls, config: dict) -> ModelRegistry:
        """Create a new ModelRegistry instance from a dictionary."""
        validate_model_zoo_json(config)
        models = {
            name: RegisteredModel(
                name=name,
                description=kwds["description"],
                hf_repo_id=kwds["hf_repo_id"],
                hf_revision=kwds["hf_revision"],
            )
            for name, kwds in config["models"].items()
        }

        return cls(models=models)


@functools.lru_cache()
def load_registry(registry_file: str | Path | None = None) -> ModelRegistry:
    """Load model registry.

    This downloads the registry JSON file to a cache and reuses it if
    the remote file is the same as the cached file.

    If registry_file is not None, it should be a path to a JSON file. This will be
    preferred over the remote registry file on HuggingFace.
    """
    logger.info("Loading model registry")
    if registry_file is None:
        logger.debug("Downloading model registry from HuggingFace Hub")
        path = hf_hub_download(
            repo_id="kaczmarj/wsinfer-mil-model-zoo-json",
            filename="wsinfer-mil-zoo-registry.json",
            revision="main",
            repo_type="dataset",
            local_dir=WSINFER_MIL_REGISTRY_PATH.parent,
        )
        if not Path(WSINFER_MIL_REGISTRY_PATH).exists():
            raise FileNotFoundError(
                "Expected registry to be saved to"
                f" {WSINFER_MIL_REGISTRY_PATH} but was saved instead to {path}"
            )
    else:
        if not Path(registry_file).exists():
            raise FileNotFoundError(f"registry file not found at {registry_file}")
        path = registry_file

    with open(path) as f:
        registry = json.load(f)
    try:
        logger.debug("Validating model registry JSON")
        validate_model_zoo_json(registry)
    except InvalidRegistryConfiguration as e:
        raise InvalidRegistryConfiguration(
            "Registry schema is invalid. Please contact the developer by"
            " creating a new issue on our GitHub page:"
            " https://github.com/kaczmarj/wsinfer-mil/issues/new."
        ) from e

    return ModelRegistry.from_dict(registry)
