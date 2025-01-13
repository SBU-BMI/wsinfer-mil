import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import Compose
from transformers import AutoImageProcessor
from transformers import ViTModel

from .base import PatchFeatureExtractor


class Phikon(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Phikon"

    def load_model(self) -> torch.nn.Module:
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> Compose:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        return lambda image: processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

    def get_batch_embeddings(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        output = self.model(batch)
        return output.last_hidden_state[:, 0, :].detach().cpu().numpy()  # type: ignore
