#!/usr/bin/env python3
"""Detect battery side edges as two line segments (4 points total) and visualize them.

This script focuses on locating the two long battery side edges in each input image.
It returns the segment endpoints and can draw thick overlay lines for easy inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectionResult:
    image: str
    ok: bool
    message: str
    # Side edge A endpoints
    x1a: Optional[float] = None
    y1a: Optional[float] = None
    x2a: Optional[float] = None
    y2a: Optional[float] = None
    # Side edge B endpoints
    x1b: Optional[float] = None
    y1b: Optional[float] = None
    x2b: Optional[float] = None
    y2b: Optional[float] = None


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_u8


def battery_roi(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return ROI around likely battery and ROI bounds (x1,y1,x2,y2)."""
    mag = sobel_magnitude(gray)

    thr = int(np.percentile(mag, 84))
    thr = max(28, min(185, thr))
    edge = (mag >= thr).astype(np.uint8) * 255

    edge = cv2.morphologyEx(
        edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2
    )
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    if not contours:
        return gray.copy(), (0, 0, w, h)

    cx0, cy0 = w * 0.5, h * 0.5
    best_rect = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 0.0015 * (w * h):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        cx, cy = x + cw * 0.5, y + ch * 0.5
        dist = math.hypot(cx - cx0, cy - cy0)
        dist_pen = 1.0 - min(1.0, dist / (0.8 * math.hypot(w, h)))
        score = area * (0.6 + 0.4 * dist_pen)
        if score > best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        return gray.copy(), (0, 0, w, h)

    x, y, cw, ch = best_rect
    pad_x = int(0.18 * cw)
    pad_y = int(0.20 * ch)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    return gray[y1:y2, x1:x2], (x1, y1, x2, y2)


def _line_angle_deg(line: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in line]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in line]
    return math.hypot(x2 - x1, y2 - y1)


def _line_midpoint(line: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in line]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _point_to_line_distance(pt: tuple[float, float], line: np.ndarray) -> float:
    px, py = pt
    x1, y1, x2, y2 = [float(v) for v in line]
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return math.hypot(px - x1, py - y1)
    cross = abs(vx * wy - vy * wx)
    return cross / math.sqrt(denom)


def _signed_offset_from_line(line_ref: np.ndarray, line_other: np.ndarray) -> float:
    """Signed perpendicular offset of line_other midpoint from line_ref."""
    x1, y1, x2, y2 = [float(v) for v in line_ref]
    ox, oy = _line_midpoint(line_other)
    vx, vy = x2 - x1, y2 - y1
    norm = math.hypot(vx, vy)
    if norm <= 1e-6:
        return 0.0
    # left normal of ref direction
    nx, ny = -vy / norm, vx / norm
    dx, dy = ox - x1, oy - y1
    return dx * nx + dy * ny


def _select_best_pair(
    cand: np.ndarray,
    min_len: float,
    w: int,
    h: int,
    max_angle_diff: float,
    min_sep_ratio: float,
    max_sep_ratio: float,
    angle_penalty: float,
    sep_weight: float,
    off_weight: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    best_pair = None
    best_score = -1e9

    for i in range(len(cand)):
        li = cand[i]
        len_i = _line_length(li)
        if len_i < min_len:
            continue
        ai = _line_angle_deg(li)

        for j in range(i + 1, len(cand)):
            lj = cand[j]
            len_j = _line_length(lj)
            if len_j < min_len:
                continue
            aj = _line_angle_deg(lj)

            d = abs(ai - aj) % 180.0
            d = min(d, 180.0 - d)
            if d > max_angle_diff:
                continue

            sep = _point_to_line_distance(_line_midpoint(lj), li)
            min_sep = max(8.0, min_sep_ratio * min(w, h))
            max_sep = max(min_sep + 2.0, max_sep_ratio * min(w, h))
            if sep < min_sep or sep > max_sep:
                continue

            off = _signed_offset_from_line(li, lj)
            score = (len_i + len_j) - angle_penalty * d + sep_weight * sep + off_weight * abs(off)

            if score > best_score:
                best_score = score
                best_pair = (li.copy(), lj.copy())

    return best_pair


def detect_side_segments(roi_gray: np.ndarray, profile: str = "auto") -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Detect two long, near-parallel side edges in ROI.

    Returns two line segments as arrays [x1,y1,x2,y2] in ROI coordinates.
    Uses filename-based threshold profiles:
    - profile="o": stricter optical thresholds
    - profile="t": older/relaxed thermal thresholds
    - profile="auto": strict pass then relaxed fallback
    """
    mag = sobel_magnitude(roi_gray)
    h, w = roi_gray.shape[:2]
    scale = max(w, h)

    def run_pass(
        percentile: float,
        t_min: int,
        t_max: int,
        open_k: int,
        close_k: int,
        min_len_ratio: float,
        hough_thresh_ratio: float,
        hough_thresh_min: int,
        max_gap_ratio: float,
        max_gap_min: int,
        max_angle_diff: float,
        min_sep_ratio: float,
        max_sep_ratio: float,
        angle_penalty: float,
        sep_weight: float,
        off_weight: float,
        top_k: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        t = int(np.percentile(mag, percentile))
        t = max(t_min, min(t_max, t))
        edge = (mag >= t).astype(np.uint8) * 255
        edge = cv2.morphologyEx(
            edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k)), iterations=1
        )
        edge = cv2.morphologyEx(
            edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)), iterations=1
        )

        min_len = max(20, int(min_len_ratio * scale))
        lines = cv2.HoughLinesP(
            edge,
            rho=1,
            theta=np.pi / 180,
            threshold=max(hough_thresh_min, int(hough_thresh_ratio * scale)),
            minLineLength=min_len,
            maxLineGap=max(max_gap_min, int(max_gap_ratio * scale)),
        )
        if lines is None or len(lines) < 2:
            return None

        cand = np.array([l[0] for l in lines], dtype=np.float32)
        lengths = np.array([_line_length(l) for l in cand], dtype=np.float32)
        order = np.argsort(-lengths)
        cand = cand[order[: min(top_k, len(order))]]

        return _select_best_pair(
            cand=cand,
            min_len=min_len,
            w=w,
            h=h,
            max_angle_diff=max_angle_diff,
            min_sep_ratio=min_sep_ratio,
            max_sep_ratio=max_sep_ratio,
            angle_penalty=angle_penalty,
            sep_weight=sep_weight,
            off_weight=off_weight,
        )

    # Optical profile (current strict thresholds)
    if profile == "o":
        return run_pass(
            percentile=88,
            t_min=35,
            t_max=205,
            open_k=3,
            close_k=5,
            min_len_ratio=0.45,
            hough_thresh_ratio=0.14,
            hough_thresh_min=55,
            max_gap_ratio=0.04,
            max_gap_min=10,
            max_angle_diff=7.0,
            min_sep_ratio=0.12,
            max_sep_ratio=0.45,
            angle_penalty=3.2,
            sep_weight=0.25,
            off_weight=0.08,
            top_k=40,
        )

    # Thermal profile (old relaxed thresholds)
    if profile == "t":
        return run_pass(
            percentile=82,
            t_min=24,
            t_max=185,
            open_k=3,
            close_k=7,
            min_len_ratio=0.30,
            hough_thresh_ratio=0.10,
            hough_thresh_min=36,
            max_gap_ratio=0.06,
            max_gap_min=14,
            max_angle_diff=12.0,
            min_sep_ratio=0.06,
            max_sep_ratio=0.65,
            angle_penalty=2.1,
            sep_weight=0.30,
            off_weight=0.06,
            top_k=60,
        )

    # Auto mode: strict then relaxed fallback
    strict_pair = run_pass(
        percentile=88,
        t_min=35,
        t_max=205,
        open_k=3,
        close_k=5,
        min_len_ratio=0.45,
        hough_thresh_ratio=0.14,
        hough_thresh_min=55,
        max_gap_ratio=0.04,
        max_gap_min=10,
        max_angle_diff=7.0,
        min_sep_ratio=0.12,
        max_sep_ratio=0.45,
        angle_penalty=3.2,
        sep_weight=0.25,
        off_weight=0.08,
        top_k=40,
    )
    if strict_pair is not None:
        return strict_pair

    return run_pass(
        percentile=82,
        t_min=24,
        t_max=185,
        open_k=3,
        close_k=7,
        min_len_ratio=0.30,
        hough_thresh_ratio=0.10,
        hough_thresh_min=36,
        max_gap_ratio=0.06,
        max_gap_min=14,
        max_angle_diff=12.0,
        min_sep_ratio=0.06,
        max_sep_ratio=0.65,
        angle_penalty=2.1,
        sep_weight=0.30,
        off_weight=0.06,
        top_k=60,
    )


def detect_battery_edges(path: Path, debug_dir: Optional[Path]) -> DetectionResult:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return DetectionResult(image=str(path), ok=False, message="failed to read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi, (rx1, ry1, rx2, ry2) = battery_roi(gray)

    stem_l = path.stem.lower()
    if stem_l.startswith("o"):
        profile = "o"
    elif stem_l.startswith("t"):
        profile = "t"
    else:
        profile = "auto"

    pair = detect_side_segments(roi, profile=profile)
    if pair is None:
        return DetectionResult(image=str(path), ok=False, message="failed to find two side edge segments")

    la, lb = pair

    # Convert ROI -> global coordinates
    x1a, y1a, x2a, y2a = [float(v) for v in la]
    x1b, y1b, x2b, y2b = [float(v) for v in lb]

    x1a += rx1
    y1a += ry1
    x2a += rx1
    y2a += ry1

    x1b += rx1
    y1b += ry1
    x2b += rx1
    y2b += ry1

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = img.copy()

        # ROI rectangle
        cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

        # Thick line overlays
        cv2.line(
            dbg,
            (int(round(x1a)), int(round(y1a))),
            (int(round(x2a)), int(round(y2a))),
            (0, 0, 255),
            8,
            cv2.LINE_AA,
        )
        cv2.line(
            dbg,
            (int(round(x1b)), int(round(y1b))),
            (int(round(x2b)), int(round(y2b))),
            (0, 255, 0),
            8,
            cv2.LINE_AA,
        )

        # Endpoints (4 points)
        for (x, y), color in [
            ((x1a, y1a), (0, 0, 255)),
            ((x2a, y2a), (0, 0, 255)),
            ((x1b, y1b), (0, 255, 0)),
            ((x2b, y2b), (0, 255, 0)),
        ]:
            cv2.circle(dbg, (int(round(x)), int(round(y))), 6, color, -1)

        cv2.putText(
            dbg,
            f"Detected battery side edges ({profile}-profile)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (50, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / f"{path.stem}_battery_edges_debug.jpg"), dbg)

    return DetectionResult(
        image=str(path),
        ok=True,
        message="ok",
        x1a=x1a,
        y1a=y1a,
        x2a=x2a,
        y2a=y2a,
        x1b=x1b,
        y1b=y1b,
        x2b=x2b,
        y2b=y2b,
    )


def collect_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            out.append(p)
        else:
            out.extend(sorted(Path(".").glob(item)))

    uniq: list[Path] = []
    seen = set()
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect battery as two side-edge line segments (4 points) and visualize thick overlay lines"
    )
    parser.add_argument("inputs", nargs="+", help="image files or glob patterns")
    parser.add_argument("--debug-dir", default=None, help="save visualization images here")
    parser.add_argument("--json", action="store_true", help="print JSON one object per line")
    args = parser.parse_args()

    images = collect_inputs(args.inputs)
    if not images:
        print("No matching images found")
        return 1

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    rc = 0
    for p in images:
        res = detect_battery_edges(p, debug_dir)
        if args.json:
            print(json.dumps(res.__dict__, ensure_ascii=True))
        else:
            if res.ok:
                print(
                    f"{res.image}: "
                    f"A=({res.x1a:.1f},{res.y1a:.1f})->({res.x2a:.1f},{res.y2a:.1f}) "
                    f"B=({res.x1b:.1f},{res.y1b:.1f})->({res.x2b:.1f},{res.y2b:.1f})"
                )
            else:
                print(f"{res.image}: ERROR - {res.message}")
                rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
