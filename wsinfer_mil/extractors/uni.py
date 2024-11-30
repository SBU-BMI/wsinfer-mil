import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose

from .base import PatchFeatureExtractor


class UNI(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "UNI"

    def load_model(self) -> torch.nn.Module:
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> Compose:
        return create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
