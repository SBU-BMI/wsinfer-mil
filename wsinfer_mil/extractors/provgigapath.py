import timm
import torch
from torchvision import transforms

from .base import PatchFeatureExtractor


class ProvGigaPath(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Prov-GigaPath"

    def load_model(self) -> torch.nn.Module:
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
