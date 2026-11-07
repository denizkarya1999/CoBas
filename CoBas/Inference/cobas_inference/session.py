"""Discover chirp triplets in a CoBas Battery Reader capture session.

A session folder written by the app looks like::

    Session_<timestamp>_<id>/
        mmWave Frames/frame_000001.jpg ...            (one per chirp, numbered globally)
        Thermal Frames/Thermal_Image_<pct>_Percent_Position_<p>_<k>.png
        Voices/CoBas_V1_PulseVoice_<ts>_Position_<p>_<k>.wav
        References/...

The training notebooks paired modalities ordinally inside each class and
position, which for one session means: the k-th thermal frame and k-th voice
recording of position p go with the mmWave frame whose global number is the
running count of chirps before position p plus k.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .specs import CLASS_NAMES, MMWAVE, THERMAL, VOICE

_POSITION_INDEX = re.compile(r"Position_(\d+)_(\d+)\.(?:png|wav)$", re.IGNORECASE)
_MMWAVE_FRAME = re.compile(r"frame_(\d+)\.jpe?g$", re.IGNORECASE)
_LABEL_DIR = re.compile(r"^(\d+)_?\s*Percent", re.IGNORECASE)

MMWAVE_DIR = "mmWave Frames"
THERMAL_DIR = "Thermal Frames"
VOICE_DIR = "Voices"


@dataclass(frozen=True)
class ChirpTriplet:
    position: int
    index: int  # 1-based chirp number within the position
    global_index: int  # 1-based chirp number within the session
    mmwave: Path | None
    thermal: Path | None
    voice: Path | None

    def paths(self) -> dict[str, Path]:
        present = {MMWAVE: self.mmwave, THERMAL: self.thermal, VOICE: self.voice}
        return {name: path for name, path in present.items() if path is not None}

    @property
    def name(self) -> str:
        return f"P{self.position}-{self.index:03d}"


def label_from_path(path: Path) -> str | None:
    """Read '50%' from a parent folder such as '50_Percent_Battery' or '50_ Percent'."""
    for part in reversed(path.resolve().parts):
        match = _LABEL_DIR.match(part)
        if match:
            label = f"{int(match.group(1))}%"
            return label if label in CLASS_NAMES else None
    return None


def _indexed(files: Iterable[Path]) -> dict[tuple[int, int], Path]:
    indexed: dict[tuple[int, int], Path] = {}
    for file in files:
        match = _POSITION_INDEX.search(file.name)
        if match:
            indexed[(int(match.group(1)), int(match.group(2)))] = file
    return indexed


def discover_session(session_dir: str | Path) -> list[ChirpTriplet]:
    """List every chirp of a session with whichever modality files exist for it."""
    session_dir = Path(session_dir)
    thermal = _indexed((session_dir / THERMAL_DIR).glob("*.png")) if (session_dir / THERMAL_DIR).is_dir() else {}
    voice = _indexed((session_dir / VOICE_DIR).glob("*.wav")) if (session_dir / VOICE_DIR).is_dir() else {}
    mmwave: dict[int, Path] = {}
    if (session_dir / MMWAVE_DIR).is_dir():
        for file in (session_dir / MMWAVE_DIR).iterdir():
            match = _MMWAVE_FRAME.search(file.name)
            if match:
                mmwave[int(match.group(1))] = file

    keys = sorted(set(thermal) | set(voice))
    if not keys and not mmwave:
        raise FileNotFoundError(f"No chirp files found in {session_dir}")

    per_position: dict[int, int] = {}
    for position, index in keys:
        per_position[position] = max(per_position.get(position, 0), index)

    triplets: list[ChirpTriplet] = []
    offset = 0
    for position in sorted(per_position):
        count = per_position[position]
        for index in range(1, count + 1):
            global_index = offset + index
            triplets.append(
                ChirpTriplet(
                    position=position,
                    index=index,
                    global_index=global_index,
                    mmwave=mmwave.get(global_index),
                    thermal=thermal.get((position, index)),
                    voice=voice.get((position, index)),
                )
            )
        offset += count
    return triplets


def find_sessions(root: str | Path) -> list[Path]:
    """Find session folders under a dataset root (any depth)."""
    root = Path(root)
    if (root / THERMAL_DIR).is_dir() or (root / VOICE_DIR).is_dir():
        return [root]
    sessions = [
        path
        for path in root.rglob("*")
        if path.is_dir() and ((path / THERMAL_DIR).is_dir() or (path / VOICE_DIR).is_dir())
    ]
    return sorted(sessions)
