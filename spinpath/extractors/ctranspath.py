import huggingface_hub
import torch
from torchvision import transforms

from .base import PatchFeatureExtractor


class CTransPath(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "CTransPath"

    def load_model(self) -> torch.nn.Module:
        model_path = huggingface_hub.hf_hub_download(
            repo_id="kaczmarj/CTransPath", filename="torchscript_model.pt"
        )
        model: torch.nn.Module = torch.jit.load(model_path, map_location="cpu")
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
