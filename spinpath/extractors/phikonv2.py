import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import Compose
from transformers import AutoImageProcessor
from transformers import AutoModel

from .base import PatchFeatureExtractor


class PhikonV2(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Phikon-v2"

    def load_model(self) -> torch.nn.Module:
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> Compose:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        return lambda image: processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

    def get_batch_embeddings(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        output = self.model(batch)
        return output.last_hidden_state[:, 0, :].detach().cpu().numpy()  # type: ignore
