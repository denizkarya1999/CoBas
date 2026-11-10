"""High-level API: one chirp triplet in, battery-level probabilities out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import preprocess
from .model import (
    OnnxRunner,
    TorchRunner,
    load_model,
    load_sidecar,
    softmax,
)
from .session import ChirpTriplet
from .specs import VOICE, FusionSpec, resolve_spec


@dataclass(frozen=True)
class Prediction:
    probabilities: np.ndarray
    class_names: tuple[str, ...]
    elapsed_ms: float = 0.0

    @property
    def index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def label(self) -> str:
        return self.class_names[self.index]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.index])

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(self.probabilities)[::-1][:k]
        return [(self.class_names[i], float(self.probabilities[i])) for i in order]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "probabilities": {
                name: round(float(p), 6) for name, p in zip(self.class_names, self.probabilities)
            },
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class Predictor:
    """Load a CoBas fusion model once, then classify chirp triplets."""

    def __init__(self, runner, spec: FusionSpec, class_names: Sequence[str], model_path: Path, metadata: Mapping[str, Any]):
        self.runner = runner
        self.spec = spec
        self.class_names = tuple(class_names)
        self.model_path = Path(model_path)
        self.metadata = dict(metadata)

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        *,
        weights_override: Sequence[float] | None = None,
        threads: int | None = None,
    ) -> "Predictor":
        model_path = Path(model_path)
        if model_path.suffix.lower() == ".onnx":
            class_names, metadata = load_sidecar(model_path)
            runner = OnnxRunner(model_path, threads=threads)
        else:
            model, class_names, metadata = load_model(model_path)
            runner = TorchRunner(model, threads=threads)
        spec = resolve_spec(model_path, metadata, weights_override=weights_override)
        return cls(runner, spec, class_names, model_path, metadata)

    # -- preprocessing -------------------------------------------------------

    def fused_image(self, paths: Mapping[str, str | Path]) -> np.ndarray:
        return preprocess.fused_image(paths, self.spec)

    def model_input(self, paths: Mapping[str, str | Path]) -> np.ndarray:
        return preprocess.model_input(paths, self.spec)

    # -- inference -----------------------------------------------------------

    def predict_input(self, x: np.ndarray) -> Prediction:
        started = time.perf_counter()
        logits = self.runner(x)
        probabilities = softmax(logits)[0]
        return Prediction(probabilities, self.class_names, (time.perf_counter() - started) * 1000.0)

    def predict_paths(self, paths: Mapping[str, str | Path]) -> Prediction:
        started = time.perf_counter()
        x = self.model_input(paths)
        logits = self.runner(x)
        probabilities = softmax(logits)[0]
        return Prediction(probabilities, self.class_names, (time.perf_counter() - started) * 1000.0)

    def predict_file(self, path: str | Path) -> Prediction:
        """Classify one picture or voice recording from disk (see ``preprocess.file_model_input``)."""
        started = time.perf_counter()
        x = preprocess.file_model_input(path, self.spec)
        logits = self.runner(x)
        probabilities = softmax(logits)[0]
        return Prediction(probabilities, self.class_names, (time.perf_counter() - started) * 1000.0)

    def voice_spectrogram(self, path: str | Path) -> np.ndarray:
        """Voice recording -> the spectrogram picture this model reads (voice standalone only)."""
        preprocess.require_voice_model(self.spec)
        return self.fused_image({VOICE: path})

    def predict_picture(self, picture: np.ndarray) -> Prediction:
        """Classify a picture already in memory, e.g. the result of ``voice_spectrogram``."""
        return self.predict_input(
            preprocess.to_model_input(picture, self.spec.tensor_norm, self.spec.tensor_pipeline)
        )

    def predict_triplet(self, triplet: ChirpTriplet) -> Prediction:
        paths = triplet.paths()
        missing = [name for name in self.spec.modalities if name not in paths]
        if missing:
            raise FileNotFoundError(
                f"Chirp {triplet.name} is missing {', '.join(missing)} for model {self.spec.family}"
            )
        return self.predict_paths(paths)

    def requires(self) -> tuple[str, ...]:
        return self.spec.modalities


@dataclass
class VoteAggregator:
    """Combine several chirp predictions by averaging log-probabilities."""

    class_names: tuple[str, ...]
    log_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    count: int = 0
    labels: list[str] = field(default_factory=list)

    def add(self, prediction: Prediction) -> None:
        if self.log_sum.size == 0:
            self.log_sum = np.zeros(len(self.class_names), dtype=np.float64)
        self.log_sum += np.log(np.clip(prediction.probabilities, 1e-12, 1.0))
        self.count += 1
        self.labels.append(prediction.label)

    def result(self) -> Prediction | None:
        if self.count == 0:
            return None
        mean_log = self.log_sum / self.count
        probabilities = np.exp(mean_log - mean_log.max())
        probabilities /= probabilities.sum()
        return Prediction(probabilities, self.class_names)

    def consistency(self) -> float:
        """Fraction of individual chirps that agree with the aggregated label."""
        result = self.result()
        if result is None:
            return 0.0
        return sum(1 for label in self.labels if label == result.label) / self.count
