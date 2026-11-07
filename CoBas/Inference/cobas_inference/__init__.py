"""On-device inference for the CoBas battery-level fusion models.

The trained checkpoints are torchvision ResNet-18 classifiers that consume one
fused 256x256 RGB image built from the per-chirp mmWave, thermal and voice
captures. Each model family was trained with a slightly different fusion and
normalization recipe; ``specs`` records those recipes, ``preprocess`` applies
them, ``model`` loads and runs the network, ``session`` finds chirp triplets in
a capture session, and ``predictor`` ties it together.
"""

from .specs import CLASS_NAMES, FusionSpec, resolve_spec, spec_for_model_file
from .predictor import Predictor, Prediction, VoteAggregator

__all__ = [
    "CLASS_NAMES",
    "FusionSpec",
    "Predictor",
    "Prediction",
    "VoteAggregator",
    "resolve_spec",
    "spec_for_model_file",
]
