import numpy as np
import numpy.typing as npt
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision.transforms import Compose

from .base import PatchFeatureExtractor


class Virchow2(PatchFeatureExtractor):
    @property
    def name(self) -> str:
        return "Virchow2"

    def load_model(self) -> torch.nn.Module:
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        assert isinstance(model, torch.nn.Module)
        model = model.eval().to(self.device)
        return model

    @property
    def transform(self) -> Compose:
        return create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def get_batch_embeddings(self, batch: torch.Tensor) -> npt.NDArray[np.float32]:
        output = self.model(batch)  # size: b x 261 x 1280
        class_token = output[:, 0]  # size: b x 1280
        # size: b x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        patch_tokens = output[:, 5:]
        # concatenate class token and average pool of patch tokens
        # size: b x 2560
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding.detach().cpu().numpy()
