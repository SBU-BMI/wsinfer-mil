import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms
from transformers import AutoImageProcessor
from transformers import AutoModel

from .base import PatchFeatureExtractor


class HibouL(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Hibou-L"

    def load_model(self) -> torch.nn.Module:
        model = AutoModel.from_pretrained(
            "histai/hibou-L",
            trust_remote_code=True,
        )
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> transforms.Compose:
        processor = AutoImageProcessor.from_pretrained(
            "histai/hibou-L", trust_remote_code=True
        )
        return lambda image: processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

    def get_batch_embeddings(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        output = self.model(pixel_values=batch).pooler_output
        return output.detach().cpu().numpy()  # type: ignore
