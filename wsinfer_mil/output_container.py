from __future__ import annotations

import dataclasses
import json
import logging
from typing import Protocol
from typing import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt
from tabulate import tabulate

__all__ = ["ModelInferenceOutput"]

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""

    def default(self, obj):  # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SupportsWrite(Protocol):
    def write(self, data: str) -> None: ...


@dataclasses.dataclass
class ModelInferenceOutput:
    logits: npt.NDArray[np.float32]
    softmax_probs: npt.NDArray[np.float32]
    attention: npt.NDArray[np.float32]
    class_names: Sequence[str]
    patch_coordinates: npt.NDArray[np.int_]

    @overload
    def to_json(self, fp: None = None) -> str: ...

    @overload
    def to_json(self, fp: SupportsWrite) -> None: ...

    def to_json(self, fp: SupportsWrite | None = None) -> str | None:
        d = dataclasses.asdict(self)
        if fp is None:
            return json.dumps(d, cls=_NumpyEncoder)
        else:
            json.dump(d, fp=fp, cls=_NumpyEncoder)
        return None

    def to_str_table(self, tablefmt: str = "simple") -> str:
        probs: list[float]
        if self.softmax_probs.ndim > 2:
            raise ValueError(
                "Expected softmax probs to be 1- or 2-dim but got"
                f" {self.softmax_probs.ndim}-dim"
            )
        if self.softmax_probs.ndim == 2:
            if self.softmax_probs.shape[0] == 1:
                probs = self.softmax_probs.squeeze(0).tolist()  # type: ignore
            else:
                raise ValueError(
                    "In the case of 2-dim softmax probabiltiies, expected first axis to"
                    f" have length 1 but got {self.softmax_probs.shape[0]}"
                )
        else:
            probs = self.softmax_probs.tolist()  # type: ignore

        if len(probs) != len(self.class_names):
            raise ValueError(
                "Expected the number of output probabiltiies to be equal to the number"
                f" of class names but got {len(probs)} vs {len(self.class_names)}."
            )
        headers = ["Class", "Probability"]
        rows = list(zip(self.class_names, probs))
        return tabulate(rows, headers=headers, tablefmt=tablefmt)
