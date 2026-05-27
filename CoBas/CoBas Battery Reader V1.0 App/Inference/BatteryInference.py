from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASS_NAMES = ["0%", "50%", "100%"]


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    segment_count: int
    class_confidences: list[tuple[str, float]]


class SpectrogramCNN(nn.Module):
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.projection(x)


class MultiModalFusionNet(nn.Module):
    def __init__(self, num_classes: int, image_features: int = 256, audio_features: int = 256):
        super().__init__()
        image_model = models.resnet18(weights=None)
        image_in = image_model.fc.in_features
        image_model.fc = nn.Linear(image_in, image_features)

        self.image_model = image_model
        self.audio_model = SpectrogramCNN(out_features=audio_features)
        self.classifier = nn.Sequential(
            nn.Linear(image_features + audio_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        image_features = self.image_model(image)
        audio_features = self.audio_model(spectrogram)
        fused = torch.cat([audio_features, image_features], dim=1)
        return self.classifier(fused)


class BatteryPercentagePredictor:
    def __init__(self, app_root: Path):
        self.app_root = Path(app_root)
        self.model_path = self.app_root / "Models" / "CoBas_Multimodal_Fusion_Model.pth"
        self.device = torch.device("cpu")
        self.model = None
        self.index_to_label = {index: label for index, label in enumerate(CLASS_NAMES)}
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_from_video(self, video_path: str | Path) -> PredictionResult:
        output_folder = self._pipeline_output_folder(video_path)
        pairs = self._paired_inputs(output_folder)

        if not pairs:
            raise RuntimeError(f"No paired frame/spectrogram inputs found in {output_folder}")

        self._load_model()
        probabilities = []

        with torch.no_grad():
            for image_path, spectrogram_path in pairs:
                image = self._load_image_tensor(image_path)
                spectrogram = self._load_spectrogram_tensor(spectrogram_path)
                logits = self.model(image, spectrogram)
                probabilities.append(torch.softmax(logits, dim=1).squeeze(0))

        mean_probabilities = torch.stack(probabilities).mean(dim=0)
        predicted_index = int(mean_probabilities.argmax().item())
        class_confidences = [
            (
                self.index_to_label.get(index, f"{index}%"),
                float(mean_probabilities[index].item()),
            )
            for index in range(len(mean_probabilities))
        ]

        return PredictionResult(
            label=self.index_to_label.get(predicted_index, f"{predicted_index}%"),
            confidence=float(mean_probabilities[predicted_index].item()),
            segment_count=len(pairs),
            class_confidences=class_confidences,
        )

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        self.index_to_label = {
            int(index): label
            for index, label in checkpoint.get("index_to_label", self.index_to_label).items()
        }

        self.model = MultiModalFusionNet(num_classes=len(self.index_to_label)).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _pipeline_output_folder(self, video_path: str | Path) -> Path:
        video_path = Path(video_path)
        return self.app_root / "Captures" / f"{video_path.stem}_Image_and_Video"

    def _paired_inputs(self, output_folder: Path) -> list[tuple[Path, Path]]:
        frames_folder = output_folder / "Frames"
        spectrogram_dirs = sorted(output_folder.glob("*_Spectogram"))

        if not frames_folder.exists() or not spectrogram_dirs:
            return []

        frames_by_index = {
            self._segment_index(path): path
            for path in sorted(frames_folder.glob("*.jpg"))
            if self._segment_index(path) is not None
        }
        pairs = []

        for spectrogram_path in sorted(spectrogram_dirs[0].glob("*.npy")):
            segment_index = self._segment_index(spectrogram_path)
            if segment_index is None:
                continue

            frame_path = frames_by_index.get(segment_index)
            if frame_path is not None:
                pairs.append((frame_path, spectrogram_path))

        return pairs

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image).unsqueeze(0).to(self.device)

    def _load_spectrogram_tensor(self, spectrogram_path: Path) -> torch.Tensor:
        spectrogram = np.load(spectrogram_path).astype(np.float32)
        spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)

    @staticmethod
    def _segment_index(path: Path) -> int | None:
        match = re.search(r"_(?:seg|frame)(\d{3})", path.stem)
        if match is None:
            return None

        return int(match.group(1))
