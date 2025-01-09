import torch
from torchvision import transforms

from .base import PatchFeatureExtractor


class KaikoL(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Kaiko-vitl14"

    def load_model(self) -> torch.nn.Module:
        model = torch.hub.load(
            "kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True
        )
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
