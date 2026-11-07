"""Load CoBas checkpoints and run ResNet-18 with PyTorch or ONNX Runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .specs import CLASS_NAMES, MODEL_INPUT_SIZE

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


class OnnxRunner:
    """Run an exported ONNX graph with ONNX Runtime on the CPU."""

    name = "onnx"

    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: x})[0]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# ONNX export with a JSON sidecar carrying the checkpoint metadata
# --------------------------------------------------------------------------- #

def sidecar_path(onnx_path: str | Path) -> Path:
    return Path(onnx_path).with_suffix(".json")


def export_onnx(
    model,
    onnx_path: str | Path,
    class_names: Sequence[str],
    metadata: Mapping[str, Any],
    opset: int = 17,
) -> Path:
    import torch

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=torch.float32)
    kwargs: dict[str, Any] = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    model.eval()
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **kwargs)

    sidecar = {
        "class_names": list(class_names),
        "metadata": _json_safe(metadata),
        "input": {"name": "input", "shape": [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]},
    }
    sidecar_path(onnx_path).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return onnx_path


def load_sidecar(onnx_path: str | Path) -> tuple[list[str], dict[str, Any]]:
    path = sidecar_path(onnx_path)
    if not path.is_file():
        return list(CLASS_NAMES), {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(name) for name in data.get("class_names", CLASS_NAMES)], dict(data.get("metadata", {}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)
