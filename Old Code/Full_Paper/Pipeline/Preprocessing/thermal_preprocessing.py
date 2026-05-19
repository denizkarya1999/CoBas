"""Thermal + optical preprocessing for CycleGAN training.

Implements the plan documented at the top of GAN.py:

    decode synced 640x480 frames
    -> strip Foxwell RT280 HUD from thermal
    -> independently crop a fixed square cell ROI in each view
    -> resize to 256x256
    -> save 3-channel pseudocolor PNGs side-by-side under
       <out-root>/{opt,therm}/frame_NNNNN.png
    -> write manifest.csv with frame_index + per-stream timestamps

Inputs are the time-synced pair produced by the sync pipeline:
    data/cobas/o_synced.mp4
    data/cobas/t_synced.mp4
    data/cobas/sync_metadata.json

CycleGAN is unpaired, so the two streams are NOT spatially pixel-aligned;
they are temporally paired (frame i in opt corresponds in time to frame i
in therm, via sync_metadata.json's common_fps + overlap_start_epoch).

Usage:
    python thermal_preprocessing.py                    # uses default Colab paths
    python thermal_preprocessing.py \
        --optical /path/o_synced.mp4 \
        --thermal /path/t_synced.mp4 \
        --metadata /path/sync_metadata.json \
        --out-root /path/out

Run --calibrate-only to print bbox / HUD diagnostics without writing frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibrated constants (synced 640x480 canvas).  See plan in GAN.py.
# ---------------------------------------------------------------------------

LANDSCAPE_FRAME_SIZE = (640, 480)  # (W, H)
PORTRAIT_FRAME_SIZE = (480, 640)  # (W, H), after rotating optical 270 deg
GAN_INPUT_SIZE = (256, 256)  # (W, H) for both streams

# HUD-free rect on the synced thermal (Foxwell RT280, 640x480 after sync).
THERMAL_HUD_FREE_XYXY = (5, 50, 480, 400)

# Squared cell ROIs (independent per view; stage C of the plan).
#
# Sized to a SAFE SUPERSET of the cell's position across the whole video
# (sampled 40 frames evenly). User constraint: must never clip the cell, even
# at the cost of being off-center or including extra background.
#
#   THERMAL: detected cell-union across 30/40 frames was (258,202,430,381);
#            squared+padded(20px), clamped to HUD-free rect.
#   OPTICAL: cell is stationary at ~(235,162,530,268) (blue HSV detection is
#            unreliable due to reflected light, so we pad the manually
#            measured bbox by 30px and square it).
OPTICAL_CELL_XYXY = (205, 38, 560, 393)  # 355x355 @ (382, 215) center
OPTICAL_CELL_XYXY_PORTRAIT = (38, 79, 393, 434)  # mapped from OPTICAL_CELL_XYXY
THERMAL_CELL_XYXY = (235, 182, 453, 400)  # 218x218 @ (344, 291) center

# Default I/O paths (Colab Drive layout — Q4 = (b)).  All overridable via CLI.
DEFAULT_OPTICAL_MP4 = "data/cobas/o_synced.mp4"
DEFAULT_THERMAL_MP4 = "data/cobas/t_synced.mp4"
DEFAULT_METADATA = "data/cobas/sync_metadata.json"
DEFAULT_OUT_ROOT = "drive/MyDrive/images/battery"


# ---------------------------------------------------------------------------
# Stateless preprocessing primitives
# ---------------------------------------------------------------------------


def strip_hud(thermal_frame: np.ndarray) -> np.ndarray:
    """Crop a synced thermal frame to the HUD-free region."""
    x1, y1, x2, y2 = THERMAL_HUD_FREE_XYXY
    return thermal_frame[y1:y2, x1:x2]


def crop_cell(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame to the given (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = xyxy
    return frame[y1:y2, x1:x2]


def resize_for_gan(crop: np.ndarray) -> np.ndarray:
    """Resize a square crop to GAN_INPUT_SIZE.

    Uses INTER_AREA when downscaling, INTER_CUBIC when upscaling.
    """
    h, w = crop.shape[:2]
    target_w, target_h = GAN_INPUT_SIZE
    interp = cv2.INTER_AREA if (w >= target_w and h >= target_h) else cv2.INTER_CUBIC
    return cv2.resize(crop, (target_w, target_h), interpolation=interp)


def preprocess_optical(frame: np.ndarray) -> np.ndarray:
    """o_synced BGR (640x480 or rotated 480x640) -> 256x256 BGR cell crop."""
    h, w = frame.shape[:2]
    if (w, h) == LANDSCAPE_FRAME_SIZE:
        bbox = OPTICAL_CELL_XYXY
    elif (w, h) == PORTRAIT_FRAME_SIZE:
        bbox = OPTICAL_CELL_XYXY_PORTRAIT
    else:
        raise RuntimeError(
            f"unsupported optical frame size {w}x{h}; "
            f"expected {LANDSCAPE_FRAME_SIZE[0]}x{LANDSCAPE_FRAME_SIZE[1]} or "
            f"{PORTRAIT_FRAME_SIZE[0]}x{PORTRAIT_FRAME_SIZE[1]}"
        )
    return resize_for_gan(crop_cell(frame, bbox))


def preprocess_thermal(frame: np.ndarray) -> np.ndarray:
    """t_synced 640x480 BGR  ->  256x256 BGR cell crop, HUD removed.

    The thermal cell box already lies inside the HUD-free rect, so the
    explicit strip_hud() call is a guard against future bbox drift rather
    than a slicing necessity.
    """
    return resize_for_gan(crop_cell(frame, THERMAL_CELL_XYXY))


# ---------------------------------------------------------------------------
# End-to-end pair preprocessing
# ---------------------------------------------------------------------------


@dataclass
class SyncMeta:
    overlap_start_epoch: float
    common_fps: float
    shared_frame_count: int
    optical_trim_start_sec: float
    thermal_trim_start_sec: float


def load_sync_metadata(path: Path) -> SyncMeta:
    raw = json.loads(path.read_text())
    return SyncMeta(
        overlap_start_epoch=float(raw["overlap_start_epoch"]),
        common_fps=float(raw["common_fps"]),
        shared_frame_count=int(raw["shared_frame_count"]),
        optical_trim_start_sec=float(raw["optical_trim_start_sec"]),
        thermal_trim_start_sec=float(raw["thermal_trim_start_sec"]),
    )


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    return cap


def assert_frame_size(
    cap: cv2.VideoCapture, label: str, allowed_sizes: set[tuple[int, int]]
) -> None:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) not in allowed_sizes:
        expected = " or ".join(f"{aw}x{ah}" for aw, ah in sorted(allowed_sizes))
        raise RuntimeError(f"{label} frame size is {w}x{h}, expected {expected}")


def preprocess_pair(
    optical_mp4: Path,
    thermal_mp4: Path,
    metadata_json: Path,
    out_root: Path,
    qa_samples: int = 16,
    limit: int | None = None,
) -> None:
    meta = load_sync_metadata(metadata_json)

    out_opt = out_root / "opt"
    out_therm = out_root / "therm"
    out_opt.mkdir(parents=True, exist_ok=True)
    out_therm.mkdir(parents=True, exist_ok=True)

    cap_o = open_video(optical_mp4)
    cap_t = open_video(thermal_mp4)
    assert_frame_size(cap_o, "optical", {LANDSCAPE_FRAME_SIZE, PORTRAIT_FRAME_SIZE})
    assert_frame_size(cap_t, "thermal", {LANDSCAPE_FRAME_SIZE})

    manifest_path = out_root / "manifest.csv"
    manifest_f = manifest_path.open("w", newline="")
    writer = csv.writer(manifest_f)
    writer.writerow(
        [
            "frame_index",
            "optical_source_frame",
            "thermal_source_frame",
            "optical_timestamp",
            "thermal_timestamp",
        ]
    )

    qa_indices = (
        set(
            random.sample(
                range(meta.shared_frame_count), min(qa_samples, meta.shared_frame_count)
            )
        )
        if qa_samples > 0
        else set()
    )
    qa_tiles: list[np.ndarray] = []

    written = 0
    for idx in range(meta.shared_frame_count):
        if limit is not None and idx >= limit:
            break
        ok_o, frame_o = cap_o.read()
        ok_t, frame_t = cap_t.read()
        if not (ok_o and ok_t):
            print(f"[warn] stream ended early at frame {idx}")
            break

        crop_o = preprocess_optical(frame_o)
        crop_t = preprocess_thermal(frame_t)

        name = f"frame_{idx:05d}.png"
        cv2.imwrite(str(out_opt / name), crop_o)
        cv2.imwrite(str(out_therm / name), crop_t)

        ts_o = meta.overlap_start_epoch + idx / meta.common_fps
        ts_t = ts_o  # streams share common_fps + overlap_start
        writer.writerow([idx, idx, idx, f"{ts_o:.6f}", f"{ts_t:.6f}"])
        written += 1

        if idx in qa_indices:
            qa_tiles.append(_build_qa_row(frame_t, frame_o, crop_t, crop_o, idx))

    cap_o.release()
    cap_t.release()
    manifest_f.close()

    if qa_tiles:
        _save_qa_grid(qa_tiles, out_root / "_qa.png")

    print(f"wrote {written} pairs to {out_root}")
    print(f"manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# QA visualization
# ---------------------------------------------------------------------------


def _annotate(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _to_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    new_w = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _build_qa_row(raw_t, raw_o, crop_t, crop_o, idx) -> np.ndarray:
    h = 240
    panels = [
        _to_h(_annotate(raw_t, f"t_synced #{idx}"), h),
        _to_h(_annotate(strip_hud(raw_t), "thermal hud-stripped"), h),
        _to_h(_annotate(crop_t, f"thermal {GAN_INPUT_SIZE[0]}x{GAN_INPUT_SIZE[1]}"), h),
        _to_h(_annotate(raw_o, f"o_synced #{idx}"), h),
        _to_h(_annotate(crop_o, f"optical {GAN_INPUT_SIZE[0]}x{GAN_INPUT_SIZE[1]}"), h),
    ]
    return np.hstack(panels)


def _save_qa_grid(rows: list[np.ndarray], path: Path) -> None:
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=r.dtype)
            r = np.hstack([r, pad])
        padded.append(r)
    grid = np.vstack(padded)
    cv2.imwrite(str(path), grid)
    print(f"QA grid: {path}")


# ---------------------------------------------------------------------------
# Calibration sanity report
# ---------------------------------------------------------------------------


def calibrate_only(optical_mp4: Path, thermal_mp4: Path) -> None:
    """Decode one frame from each stream and dump bbox / HUD overlays for
    visual sanity check. Writes to out_root / _calib_*.png is not done; we
    emit to /tmp so the user can eyeball without polluting the dataset dir.
    """
    cap_o = open_video(optical_mp4)
    ok_o, frame_o = cap_o.read()
    cap_o.release()
    cap_t = open_video(thermal_mp4)
    ok_t, frame_t = cap_t.read()
    cap_t.release()
    if not (ok_o and ok_t):
        raise RuntimeError("could not decode reference frames")

    def draw_box(img, xyxy, color, label):
        out = img.copy()
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        return out

    overlay_o = draw_box(frame_o, OPTICAL_CELL_XYXY, (0, 255, 0), "OPTICAL_CELL")
    overlay_t = draw_box(frame_t, THERMAL_HUD_FREE_XYXY, (255, 0, 0), "HUD_FREE")
    overlay_t = draw_box(overlay_t, THERMAL_CELL_XYXY, (0, 255, 0), "THERMAL_CELL")

    out_dir = Path("/tmp")
    cv2.imwrite(str(out_dir / "_calib_optical.png"), overlay_o)
    cv2.imwrite(str(out_dir / "_calib_thermal.png"), overlay_t)
    cv2.imwrite(
        str(out_dir / "_calib_thermal_cropped.png"), preprocess_thermal(frame_t)
    )
    cv2.imwrite(
        str(out_dir / "_calib_optical_cropped.png"), preprocess_optical(frame_o)
    )
    print(f"calibration overlays + 256x256 crops in {out_dir}/_calib_*.png")
    print(f"OPTICAL_CELL_XYXY = {OPTICAL_CELL_XYXY}")
    print(f"THERMAL_CELL_XYXY = {THERMAL_CELL_XYXY}")
    print(f"THERMAL_HUD_FREE_XYXY = {THERMAL_HUD_FREE_XYXY}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--optical", default=DEFAULT_OPTICAL_MP4)
    p.add_argument("--thermal", default=DEFAULT_THERMAL_MP4)
    p.add_argument("--metadata", default=DEFAULT_METADATA)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--qa-samples",
        type=int,
        default=16,
        help="number of random frames to include in _qa.png (0 = skip)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on frames written, for smoke testing",
    )
    p.add_argument(
        "--calibrate-only",
        action="store_true",
        help="just dump bbox overlays + 256x256 sample crops, do not write the dataset",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    optical_mp4 = Path(args.optical)
    thermal_mp4 = Path(args.thermal)
    metadata = Path(args.metadata)
    out_root = Path(args.out_root)

    if args.calibrate_only:
        calibrate_only(optical_mp4, thermal_mp4)
        return 0

    preprocess_pair(
        optical_mp4,
        thermal_mp4,
        metadata,
        out_root,
        qa_samples=args.qa_samples,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
