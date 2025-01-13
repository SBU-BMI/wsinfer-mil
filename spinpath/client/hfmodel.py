"""API to interact with SpinPath models on HuggingFace Hub."""

from __future__ import annotations

import dataclasses
import json

from huggingface_hub import hf_hub_download

from spinpath.client.localmodel import Model
from spinpath.client.localmodel import ModelConfiguration

HF_CONFIG_NAME = "config.json"
HF_TORCHSCRIPT_NAME = "torchscript_model.pt"


@dataclasses.dataclass
class HFInfo:
    """Container for information on model's location on HuggingFace Hub."""

    repo_id: str
    revision: str | None = None


@dataclasses.dataclass
class HFModel(Model):
    """Container for a TorchScript model hosted on HuggingFace."""

    hf_info: HFInfo


def load_torchscript_model_from_hf(
    repo_id: str, revision: str | None = None
) -> HFModel:
    """Load a TorchScript model from HuggingFace."""
    model_path = hf_hub_download(repo_id, HF_TORCHSCRIPT_NAME, revision=revision)
    config_path = hf_hub_download(repo_id, HF_CONFIG_NAME, revision=revision)
    with open(config_path) as f:
        config_dict = json.load(f)
    if not isinstance(config_dict, dict):
        raise TypeError(
            f"Expected configuration to be a dict but got {type(config_dict)}"
        )
    config = ModelConfiguration.from_dict(config_dict)
    hf_info = HFInfo(repo_id=repo_id, revision=revision)
    model = HFModel(config=config, model_path=model_path, hf_info=hf_info)
    return model
