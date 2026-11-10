"""Load CoBas checkpoints (.pth or .pt) and run ResNet-18 with PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .specs import CLASS_NAMES

STATE_KEY = "model_state_dict"


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint dict (weights plus the notebook's metadata)."""
    import torch

    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # metadata may hold objects outside the safe allowlist
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or STATE_KEY not in checkpoint:
        raise ValueError(f"{path} is not a CoBas checkpoint (missing {STATE_KEY!r})")
    return dict(checkpoint)


def checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != STATE_KEY}


def class_names_from_checkpoint(checkpoint: Mapping[str, Any]) -> list[str]:
    index_to_label = checkpoint.get("index_to_label")
    if isinstance(index_to_label, Mapping) and index_to_label:
        names = []
        for index in range(len(index_to_label)):
            if index in index_to_label:
                names.append(str(index_to_label[index]))
            elif str(index) in index_to_label:
                names.append(str(index_to_label[str(index)]))
            else:
                raise ValueError(f"index_to_label is missing index {index}")
        return names
    label_to_index = checkpoint.get("label_to_index")
    if isinstance(label_to_index, Mapping) and label_to_index:
        names = [""] * len(label_to_index)
        for label, index in label_to_index.items():
            names[int(index)] = str(label)
        return names
    num_classes = int(checkpoint[STATE_KEY]["fc.weight"].shape[0])
    if num_classes == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [str(i) for i in range(num_classes)]


def build_resnet18(num_classes: int):
    import torch
    from torchvision import models

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(path: str | Path):
    """Return (eval-mode ResNet-18, class names, metadata) for a .pth checkpoint."""
    checkpoint = load_checkpoint(path)
    state = checkpoint[STATE_KEY]
    num_classes = int(state["fc.weight"].shape[0])
    model = build_resnet18(num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_names_from_checkpoint(checkpoint), checkpoint_metadata(checkpoint)


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #

class TorchRunner:
    """Run a torch model on float32 NCHW numpy input; returns logits as numpy."""

    name = "torch"

    def __init__(self, model, threads: int | None = None):
        import torch

        if threads:
            torch.set_num_threads(int(threads))
        self.model = model.eval()
        self._torch = torch

    def __call__(self, x: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        with self._torch.inference_mode():
            return self.model(tensor).cpu().numpy()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
