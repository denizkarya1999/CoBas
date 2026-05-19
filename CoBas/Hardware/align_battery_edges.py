#!/usr/bin/env python3
"""Align images by battery side edges so batteries overlay as closely as possible.

This script reuses detection logic from `detect_anode.py` to find the two battery
side-edge segments, then applies:
1) isotropic scaling
2) translation
3) cropping to common overlap

The result is a set of aligned images with battery side edges in near-perfect
correspondence, suitable for visual overlay.

Usage examples:
    python3 align_battery_edges.py o0_copy.jpg t2.jpg --output-dir aligned
    python3 align_battery_edges.py "frames/*.jpg" --reference o0_copy.jpg --output-dir aligned --debug-dir aligned_debug
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reuse battery edge detection from existing script
import detect_anode


@dataclass
class EdgePair:
    # side A
    x1a: float
    y1a: float
    x2a: float
    y2a: float
    # side B
    x1b: float
    y1b: float
    x2b: float
    y2b: float


@dataclass
class AlignMeta:
    image: str
    ok: bool
    message: str
    reference: str
    scale: Optional[float] = None
    tx: Optional[float] = None
    ty: Optional[float] = None
    out_path: Optional[str] = None


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


def edge_pair_from_result(res: detect_anode.DetectionResult) -> Optional[EdgePair]:
    fields = [res.x1a, res.y1a, res.x2a, res.y2a, res.x1b, res.y1b, res.x2b, res.y2b]
    if any(v is None for v in fields):
        return None
    return EdgePair(
        x1a=float(res.x1a), y1a=float(res.y1a), x2a=float(res.x2a), y2a=float(res.y2a),
        x1b=float(res.x1b), y1b=float(res.y1b), x2b=float(res.x2b), y2b=float(res.y2b),
    )


def _line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_midpoint(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def pair_center(ep: EdgePair) -> tuple[float, float]:
    m1 = _line_midpoint(ep.x1a, ep.y1a, ep.x2a, ep.y2a)
    m2 = _line_midpoint(ep.x1b, ep.y1b, ep.x2b, ep.y2b)
    return (0.5 * (m1[0] + m2[0]), 0.5 * (m1[1] + m2[1]))


def pair_mean_length(ep: EdgePair) -> float:
    l1 = _line_length(ep.x1a, ep.y1a, ep.x2a, ep.y2a)
    l2 = _line_length(ep.x1b, ep.y1b, ep.x2b, ep.y2b)
    return 0.5 * (l1 + l2)


def transform_points(pts: np.ndarray, s: float, tx: float, ty: float) -> np.ndarray:
    out = pts.copy().astype(np.float32)
    out[:, 0] = out[:, 0] * s + tx
    out[:, 1] = out[:, 1] * s + ty
    return out


def transform_edge_pair(ep: EdgePair, s: float, tx: float, ty: float) -> EdgePair:
    pts = np.array(
        [
            [ep.x1a, ep.y1a],
            [ep.x2a, ep.y2a],
            [ep.x1b, ep.y1b],
            [ep.x2b, ep.y2b],
        ],
        dtype=np.float32,
    )
    t = transform_points(pts, s, tx, ty)
    return EdgePair(
        x1a=float(t[0, 0]), y1a=float(t[0, 1]),
        x2a=float(t[1, 0]), y2a=float(t[1, 1]),
        x1b=float(t[2, 0]), y1b=float(t[2, 1]),
        x2b=float(t[3, 0]), y2b=float(t[3, 1]),
    )


def compute_similarity_from_edges(src: EdgePair, ref: EdgePair) -> tuple[float, float, float]:
    """Compute isotropic scale + translation using edge lengths and pair centers."""
    src_len = max(1e-6, pair_mean_length(src))
    ref_len = max(1e-6, pair_mean_length(ref))
    s = ref_len / src_len

    src_cx, src_cy = pair_center(src)
    ref_cx, ref_cy = pair_center(ref)

    tx = ref_cx - s * src_cx
    ty = ref_cy - s * src_cy
    return float(s), float(tx), float(ty)


def warp_scale_translate(
    img: np.ndarray,
    s: float,
    tx: float,
    ty: float,
    out_w: Optional[int] = None,
    out_h: Optional[int] = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    if out_w is None:
        out_w = w
    if out_h is None:
        out_h = h
    m = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float32)
    out = cv2.warpAffine(
        img,
        m,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return out


def image_transformed_bounds(w: int, h: int, s: float, tx: float, ty: float) -> tuple[float, float, float, float]:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    tc = transform_points(corners, s, tx, ty)
    xmin = float(np.min(tc[:, 0]))
    ymin = float(np.min(tc[:, 1]))
    xmax = float(np.max(tc[:, 0]))
    ymax = float(np.max(tc[:, 1]))
    return xmin, ymin, xmax, ymax


def compute_common_crop(
    ref_shape: tuple[int, int],
    bounds_list: list[tuple[float, float, float, float]],
) -> Optional[tuple[int, int, int, int]]:
    """Intersection of reference frame with all transformed-image valid bounds."""
    ref_h, ref_w = ref_shape[:2]
    x1, y1, x2, y2 = 0.0, 0.0, float(ref_w), float(ref_h)

    for bx1, by1, bx2, by2 in bounds_list:
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
        x2 = min(x2, bx2)
        y2 = min(y2, by2)

    ix1 = int(np.ceil(x1))
    iy1 = int(np.ceil(y1))
    ix2 = int(np.floor(x2))
    iy2 = int(np.floor(y2))

    if ix2 - ix1 < 16 or iy2 - iy1 < 16:
        return None
    return ix1, iy1, ix2, iy2


def draw_edges(img: np.ndarray, ep: EdgePair, label: str) -> np.ndarray:
    dbg = img.copy()
    cv2.line(dbg, (int(round(ep.x1a)), int(round(ep.y1a))), (int(round(ep.x2a)), int(round(ep.y2a))), (0, 0, 255), 6, cv2.LINE_AA)
    cv2.line(dbg, (int(round(ep.x1b)), int(round(ep.y1b))), (int(round(ep.x2b)), int(round(ep.y2b))), (0, 255, 0), 6, cv2.LINE_AA)
    for x, y, c in [
        (ep.x1a, ep.y1a, (0, 0, 255)),
        (ep.x2a, ep.y2a, (0, 0, 255)),
        (ep.x1b, ep.y1b, (0, 255, 0)),
        (ep.x2b, ep.y2b, (0, 255, 0)),
    ]:
        cv2.circle(dbg, (int(round(x)), int(round(y))), 5, c, -1)
    cv2.putText(dbg, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 60), 2, cv2.LINE_AA)
    return dbg


def main() -> int:
    parser = argparse.ArgumentParser(description="Align images by battery side edges detected from detect_anode.py")
    parser.add_argument("inputs", nargs="+", help="Image files or glob patterns")
    parser.add_argument("--reference", default=None, help="Reference image path (default: first matched input)")
    parser.add_argument("--output-dir", required=True, help="Directory for aligned output images")
    parser.add_argument("--debug-dir", default=None, help="Optional debug directory")
    parser.add_argument("--json", action="store_true", help="Print per-image JSON metadata")
    args = parser.parse_args()

    images = collect_inputs(args.inputs)
    if not images:
        print("No matching images found")
        return 1

    ref_path = Path(args.reference) if args.reference else images[0]
    if not ref_path.is_file():
        print(f"Reference image not found: {ref_path}")
        return 1

    # Ensure reference appears in processing list
    if str(ref_path.resolve()) not in {str(p.resolve()) for p in images}:
        images = [ref_path] + images

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dbg_dir = Path(args.debug_dir) if args.debug_dir else None
    if dbg_dir is not None:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    # Detect reference edges
    ref_det = detect_anode.detect_battery_edges(ref_path, dbg_dir)
    ref_ep = edge_pair_from_result(ref_det)
    if not ref_det.ok or ref_ep is None:
        print(f"{ref_path}: ERROR - reference edge detection failed: {ref_det.message}")
        return 2

    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"{ref_path}: ERROR - failed to read reference image")
        return 2

    # Store transformed images and metadata before final common crop
    transformed: dict[str, np.ndarray] = {}
    metas: list[AlignMeta] = []
    bounds: list[tuple[float, float, float, float]] = []

    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            metas.append(
                AlignMeta(
                    image=str(p),
                    ok=False,
                    message="failed to read image",
                    reference=str(ref_path),
                )
            )
            continue

        det = detect_anode.detect_battery_edges(p, dbg_dir)
        ep = edge_pair_from_result(det)
        if not det.ok or ep is None:
            metas.append(
                AlignMeta(
                    image=str(p),
                    ok=False,
                    message=f"edge detection failed: {det.message}",
                    reference=str(ref_path),
                )
            )
            continue

        if str(p.resolve()) == str(ref_path.resolve()):
            s, tx, ty = 1.0, 0.0, 0.0
        else:
            s, tx, ty = compute_similarity_from_edges(ep, ref_ep)

        warped = warp_scale_translate(
            img,
            s,
            tx,
            ty,
            out_w=ref_img.shape[1],
            out_h=ref_img.shape[0],
        )
        transformed[str(p)] = warped

        h, w = img.shape[:2]
        b = image_transformed_bounds(w, h, s, tx, ty)
        bounds.append(b)

        metas.append(
            AlignMeta(
                image=str(p),
                ok=True,
                message="ok",
                reference=str(ref_path),
                scale=float(s),
                tx=float(tx),
                ty=float(ty),
            )
        )

        if dbg_dir is not None:
            det_dbg = draw_edges(img, ep, f"detected: {p.name}")
            cv2.imwrite(str(dbg_dir / f"{p.stem}_detected_edges.jpg"), det_dbg)

            tep = transform_edge_pair(ep, s, tx, ty)
            warp_dbg = draw_edges(warped, tep, f"transformed-to-ref: {p.name}")
            if warp_dbg.shape[:2] != ref_img.shape[:2]:
                warp_dbg = cv2.resize(
                    warp_dbg,
                    (ref_img.shape[1], ref_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            ref_overlay = cv2.addWeighted(warp_dbg, 0.65, ref_img, 0.35, 0.0)
            cv2.imwrite(str(dbg_dir / f"{p.stem}_transformed_overlay.jpg"), ref_overlay)

    # Compute common crop from valid transformed images + reference frame
    if not bounds:
        print("No images could be aligned.")
        return 2

    crop = compute_common_crop(ref_img.shape[:2], bounds)
    if crop is None:
        print("Failed to compute common crop region after alignment.")
        return 2

    cx1, cy1, cx2, cy2 = crop

    rc = 0
    for m in metas:
        if not m.ok:
            rc = 2
            if args.json:
                print(json.dumps(m.__dict__, ensure_ascii=True))
            else:
                print(f"{m.image}: ERROR - {m.message}")
            continue

        src_key = m.image
        if src_key not in transformed:
            rc = 2
            m.ok = False
            m.message = "missing transformed image buffer"
            if args.json:
                print(json.dumps(m.__dict__, ensure_ascii=True))
            else:
                print(f"{m.image}: ERROR - {m.message}")
            continue

        cropped = transformed[src_key][cy1:cy2, cx1:cx2]
        out_path = out_dir / f"{Path(m.image).stem}_aligned.png"
        cv2.imwrite(str(out_path), cropped)
        m.out_path = str(out_path)

        if args.json:
            print(json.dumps(m.__dict__, ensure_ascii=True))
        else:
            print(
                f"{m.image}: aligned -> {out_path} "
                f"(scale={m.scale:.5f}, tx={m.tx:.2f}, ty={m.ty:.2f}, crop=({cx1},{cy1})-({cx2},{cy2}))"
            )

    if dbg_dir is not None:
        ref_dbg = draw_edges(ref_img, ref_ep, f"reference: {ref_path.name}")
        ref_dbg_crop = ref_dbg[cy1:cy2, cx1:cx2]
        cv2.imwrite(str(dbg_dir / f"{ref_path.stem}_reference_edges.jpg"), ref_dbg)
        cv2.imwrite(str(dbg_dir / f"{ref_path.stem}_reference_edges_cropped.jpg"), ref_dbg_crop)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
