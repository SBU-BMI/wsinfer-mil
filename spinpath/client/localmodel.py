"""API to load local models."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Sequence

import jsonschema

from spinpath.errors import InvalidModelConfiguration


@dataclasses.dataclass
class ModelConfiguration:
    """Container for the configuration of a single model.

    This is from the contents of 'config.json'.
    """

    num_classes: int
    class_names: Sequence[str]
    feature_extractor: str
    patch_size_um: float

    def __post_init__(self) -> None:
        if len(self.class_names) != self.num_classes:
            raise InvalidModelConfiguration()

    @classmethod
    def from_dict(cls, config: dict) -> ModelConfiguration:
        validate_config_json(config)
        num_classes = config["num_classes"]
        class_names = config["class_names"]
        feature_extractor = config["feature_extractor"]
        patch_size_um = config["patch_size_um"]
        return cls(
            num_classes=num_classes,
            class_names=class_names,
            feature_extractor=feature_extractor,
            patch_size_um=patch_size_um,
        )


@dataclasses.dataclass
class Model:
    config: ModelConfiguration
    model_path: str | Path

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")


def validate_config_json(instance: object) -> bool:
    """Raise an error if the model configuration JSON is invalid. Otherwise return
    True.
    """
    schema_path = Path(__file__).parent / ".." / "schemas" / "model-config.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"JSON schema for model configurations not found: {schema_path}"
        )
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance, schema=schema)
    except jsonschema.ValidationError as e:
        raise InvalidModelConfiguration(
            "Invalid model configuration. See traceback above for details."
        ) from e

    return True


def load_torchscript_model_from_filesystem(
    model_path: str | Path, config_path: str | Path
) -> Model:
    """Load a TorchScript model from local filesystem."""
    with open(config_path) as f:
        config_dict = json.load(f)
    if not isinstance(config_dict, dict):
        raise TypeError(
            f"Expected configuration to be a dict but got {type(config_dict)}"
        )
    config = ModelConfiguration.from_dict(config_dict)
    model = Model(config=config, model_path=model_path)
    return model
