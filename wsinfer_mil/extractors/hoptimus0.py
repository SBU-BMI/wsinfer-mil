import timm
import torch
from torchvision import transforms

from .base import PatchFeatureExtractor


class HOptimus0(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "H-Optimus-0"

    def load_model(self) -> torch.nn.Module:
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
