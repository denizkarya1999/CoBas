"""Command line for the CoBas inference package.

    python -m cobas_inference inspect   MODEL.pth
    python -m cobas_inference predict   --model MODEL.pth --mmwave f.jpg --thermal f.png --voice f.wav
    python -m cobas_inference replay    --model MODEL.pth SESSION_OR_DATASET_DIR [--csv out.csv] [--as-images]
    python -m cobas_inference export-onnx --model MODEL.pth --out MODEL.onnx
    python -m cobas_inference bench     --model MODEL.pth|MODEL.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import preprocess
from .model import checkpoint_metadata, load_checkpoint, load_model, parameter_count, export_onnx
from .predictor import Predictor, VoteAggregator
from .session import discover_session, find_sessions, label_from_path
from .specs import MMWAVE, THERMAL, VOICE, resolve_spec


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="checkpoint (.pth) or exported graph (.onnx)")
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help="override fusion weights, in the model's modality order (e.g. --weights 1 1 2)",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for inference")


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.model)
    if path.suffix.lower() == ".onnx":
        predictor = Predictor.from_model(path, weights_override=args.weights, threads=args.threads)
        print(f"ONNX model: {path}")
        print(f"classes: {', '.join(predictor.class_names)}")
        print(predictor.spec.describe())
        return 0
    checkpoint = load_checkpoint(path)
    metadata = checkpoint_metadata(checkpoint)
    state = checkpoint["model_state_dict"]
    print(f"checkpoint: {path}")
    print(f"tensors: {len(state)}; fc.weight: {tuple(state['fc.weight'].shape)}")
    print("metadata:")
    print(json.dumps(_json_safe(metadata), indent=2))
    print("resolved spec:")
    print(resolve_spec(path, metadata, weights_override=args.weights).describe())
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = Predictor.from_model(args.model, weights_override=args.weights, threads=args.threads)
    paths = {}
    if args.mmwave:
        paths[MMWAVE] = Path(args.mmwave)
    if args.thermal:
        paths[THERMAL] = Path(args.thermal)
    if args.voice:
        paths[VOICE] = Path(args.voice)
    missing = [m for m in predictor.requires() if m not in paths]
    if missing:
        print(f"error: model {predictor.spec.family} needs --{' --'.join(missing)}", file=sys.stderr)
        return 2
    if args.save_fused:
        Image.fromarray(predictor.fused_image(paths)).save(args.save_fused)
        print(f"fused image saved to {args.save_fused}")
    prediction = predictor.predict_paths(paths)
    if args.json:
        print(json.dumps(prediction.as_dict(), indent=2))
    else:
        print(f"model:      {predictor.spec.family} ({predictor.runner.name})")
        print(f"estimate:   {prediction.label}  (confidence {prediction.confidence:.3f}, {prediction.elapsed_ms:.0f} ms)")
        print("top-3:      " + ", ".join(f"{name} {p:.3f}" for name, p in prediction.top(3)))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    predictor = Predictor.from_model(args.model, weights_override=args.weights, threads=args.threads)
    sessions = find_sessions(args.path)
    if not sessions:
        print(f"error: no capture sessions found under {args.path}", file=sys.stderr)
        return 2
    print(f"model: {predictor.spec.family} ({predictor.runner.name}); weights {predictor.spec.weights_source}")
    print(f"sessions: {len(sessions)}")
    scratch = tempfile.TemporaryDirectory(prefix="cobas-replay-") if args.as_images else None
    if scratch:
        classify = image_classifier(predictor, Path(scratch.name))
        print("input: one picture per chirp, opened as Local Image Inference opens it")
    else:
        classify = predictor.predict_triplet

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["session", "true_label", "position", "index", "global_index", "predicted", "confidence", "elapsed_ms"]
            + [f"p_{name}" for name in predictor.class_names]
        )

    overall_total = 0
    overall_correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    for session in sessions:
        true_label = label_from_path(session)
        try:
            triplets = discover_session(session)
        except FileNotFoundError:
            # Aborted captures leave the folder skeleton behind with no chirp files.
            print(f"{session.parent.name}/{session.name}: skipped (no chirp files)")
            continue
        usable = select_chirps(
            [t for t in triplets if all(m in t.paths() for m in predictor.requires())],
            stride=args.stride,
            limit=args.limit,
        )
        aggregator = VoteAggregator(predictor.class_names)
        correct = 0
        elapsed = []
        for triplet in usable:
            prediction = classify(triplet)
            aggregator.add(prediction)
            elapsed.append(prediction.elapsed_ms)
            if true_label is not None and prediction.label == true_label:
                correct += 1
            if true_label is not None:
                confusion[true_label][prediction.label] += 1
            if writer:
                writer.writerow(
                    [session.name, true_label or "", triplet.position, triplet.index, triplet.global_index,
                     prediction.label, f"{prediction.confidence:.6f}", f"{prediction.elapsed_ms:.1f}"]
                    + [f"{p:.6f}" for p in prediction.probabilities]
                )
        vote = aggregator.result()
        summary = f"{session.parent.name}/{session.name}: chirps={len(usable)}/{len(triplets)}"
        if vote is not None:
            summary += f" vote={vote.label} ({vote.confidence:.3f}, consistency {aggregator.consistency():.2f})"
        if true_label is not None and usable:
            summary += f" true={true_label} acc={correct / len(usable):.3f}"
            overall_total += len(usable)
            overall_correct += correct
        if elapsed:
            summary += f" {np.mean(elapsed):.0f} ms/chirp"
        print(summary)

    if scratch:
        scratch.cleanup()
    if csv_file:
        csv_file.close()
        print(f"per-chirp predictions written to {args.csv}")
    if overall_total:
        print(f"overall per-chirp accuracy: {overall_correct}/{overall_total} = {overall_correct / overall_total:.3f}")
        print("confusion (true -> predicted counts):")
        for true_label in sorted(confusion, key=_label_key):
            row = ", ".join(f"{p}:{n}" for p, n in sorted(confusion[true_label].items(), key=lambda kv: _label_key(kv[0])))
            print(f"  {true_label:>5}: {row}")
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    model, class_names, metadata = load_model(args.model)
    out = Path(args.out) if args.out else Path(args.model).with_suffix(".onnx")
    export_onnx(model, out, class_names, metadata, opset=args.opset)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB) with sidecar {out.with_suffix('.json').name}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    predictor = Predictor.from_model(args.model, threads=args.threads)
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(3):
        predictor.predict_input(x)
    times = []
    for _ in range(args.runs):
        started = time.perf_counter()
        predictor.predict_input(x)
        times.append((time.perf_counter() - started) * 1000.0)
    print(f"{predictor.runner.name}: median {np.median(times):.1f} ms, min {min(times):.1f} ms, max {max(times):.1f} ms over {args.runs} runs")
    if args.model.lower().endswith(".pth"):
        model, _, _ = load_model(args.model)
        print(f"parameters: {parameter_count(model):,}")
    return 0


def image_classifier(predictor: Predictor, scratch: Path):
    """Classify a chirp the way Local Image Inference does: from one picture on disk.

    A thermal or mmWave standalone model opens the saved frame itself. A capture
    session holds no ready-made picture for the other models, so their fused
    image (or voice spectrogram) is written to a PNG first and that file is opened.
    """
    raw_frame = preprocess.is_raw_frame_model(predictor.spec)
    picture = scratch / "picture.png"

    def classify(triplet):
        paths = triplet.paths()
        if raw_frame:
            return predictor.predict_image(paths[predictor.spec.modalities[0]])
        Image.fromarray(predictor.fused_image(paths)).save(picture)
        return predictor.predict_image(picture)

    return classify


def select_chirps(chirps: list, stride: int | None = 1, limit: int | None = None) -> list:
    """Every ``stride``-th chirp (so a short run still spans all positions), then at most ``limit``."""
    if stride and stride > 1:
        chirps = chirps[::stride]
    if limit:
        chirps = chirps[:limit]
    return chirps


def _label_key(label: str) -> int:
    try:
        return int(label.rstrip("%"))
    except ValueError:
        return 10**6


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cobas_inference", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print checkpoint metadata and the resolved preprocessing spec")
    _add_model_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("predict", help="classify one chirp from explicit files")
    _add_model_args(p)
    p.add_argument("--mmwave", help="mmWave frame JPG")
    p.add_argument("--thermal", help="thermal frame PNG")
    p.add_argument("--voice", help="voice reflection WAV")
    p.add_argument("--save-fused", help="write the fused 256x256 image here")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("replay", help="run every chirp of a session or dataset folder")
    _add_model_args(p)
    p.add_argument("path", help="Session_... folder or a dataset root containing <N>_Percent_Battery/Session_...")
    p.add_argument("--stride", type=int, default=1, help="use every N-th chirp (spreads a limited run over all positions)")
    p.add_argument("--limit", type=int, default=None, help="max chirps per session (applied after --stride)")
    p.add_argument("--csv", help="write per-chirp predictions to this CSV")
    p.add_argument(
        "--as-images", action="store_true",
        help="classify one picture per chirp as Local Image Inference does (the saved frame, "
             "or the fused image / spectrogram written to a PNG first)",
    )
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("export-onnx", help="export a .pth checkpoint to ONNX with a JSON sidecar")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("bench", help="time the network forward pass")
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--runs", type=int, default=20)
    p.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
