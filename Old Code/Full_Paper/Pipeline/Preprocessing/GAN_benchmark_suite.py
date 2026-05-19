#!/usr/bin/env python3
"""
Standalone benchmark suite for evaluating GAN generator (.pt) accuracy on paired optical/thermal frames.

This script benchmarks a trained generator against paired ground-truth targets and reports:
- Pixel-space metrics: MAE, MSE, RMSE, PSNR
- Perceptual/structure metrics: SSIM
- Optional LPIPS (if installed)
- Optional FID-style proxy using lightweight feature embeddings (if enabled)

Primary use case:
- Evaluate optical->thermal generator on paired (optical, thermal) frames.

Assumptions:
- The .pt file is a state_dict for the generator architecture defined below.
- Paired data is aligned/synchronized by sorted filename order.
- Images are RGB-compatible files.

Example:
    python3 GAN_benchmark_suite.py \
      --model /path/to/G_optical_to_thermal.pt \
      --input-dir /data/frames/optical \
      --target-dir /data/frames/thermal \
      --direction o2t \
      --image-size 256 \
      --batch-size 8 \
      --output-json benchmark_results.json \
      --save-preds-dir preds_preview

Optional split benchmark:
    python3 GAN_benchmark_suite.py ... --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.utils import save_image
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import lpips  # type: ignore
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def sorted_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p)])


def denorm(t: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (t.clamp(-1, 1) + 1.0) * 0.5


def clamp01(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


def to_numpy_img01(t: torch.Tensor) -> np.ndarray:
    # CHW [0,1] -> HWC [0,1], float32
    arr = t.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def compute_psnr_from_mse(mse: float, max_i: float = 1.0) -> float:
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(max_i) - 10.0 * math.log10(mse)


# ----------------------------
# Dataset
# ----------------------------

class PairedFrameDataset(Dataset):
    """
    Paired dataset by sorted filename order:
      input[i] <-> target[i]
    """

    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        index_subset: Optional[Sequence[int]] = None,
    ):
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for transforms/model utilities.")

        self.input_paths = sorted_images(input_dir)
        self.target_paths = sorted_images(target_dir)

        if len(self.input_paths) == 0:
            raise RuntimeError(f"No input images found in: {input_dir}")
        if len(self.target_paths) == 0:
            raise RuntimeError(f"No target images found in: {target_dir}")

        n = min(len(self.input_paths), len(self.target_paths))
        self.input_paths = self.input_paths[:n]
        self.target_paths = self.target_paths[:n]

        if index_subset is not None:
            self.input_paths = [self.input_paths[i] for i in index_subset]
            self.target_paths = [self.target_paths[i] for i in index_subset]

        if max_samples is not None:
            n2 = min(max_samples, len(self.input_paths))
            self.input_paths = self.input_paths[:n2]
            self.target_paths = self.target_paths[:n2]

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(self, idx: int):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return {
            "input": self.tf(inp),
            "target": self.tf(tgt),
            "input_path": str(self.input_paths[idx]),
            "target_path": str(self.target_paths[idx]),
        }


# ----------------------------
# Generator architecture
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, use_norm=True, use_relu=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=2, p=1, out_p=1, use_norm=True, use_relu=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, out_p, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, n_res=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, 64, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            ConvBlock(64, 128, k=3, s=2, p=1),
            ConvBlock(128, 256, k=3, s=2, p=1),
        ]
        for _ in range(n_res):
            layers.append(ResBlock(256))
        layers += [
            DeconvBlock(256, 128, k=3, s=2, p=1, out_p=1),
            DeconvBlock(128, 64, k=3, s=2, p=1, out_p=1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_nc, 7, 1, 0),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class SampleMetrics:
    mae: float
    mse: float
    rmse: float
    psnr: float
    ssim: float
    lpips: Optional[float] = None


@dataclass
class AggregateMetrics:
    n_samples: int
    mae_mean: float
    mse_mean: float
    rmse_mean: float
    psnr_mean: float
    ssim_mean: float
    lpips_mean: Optional[float]
    mae_std: float
    mse_std: float
    rmse_std: float
    psnr_std: float
    ssim_std: float
    lpips_std: Optional[float]


def _safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def _safe_std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(statistics.pstdev(vals))


def ssim_batch(pred_01: torch.Tensor, tgt_01: torch.Tensor) -> List[float]:
    """
    Compute SSIM per sample (RGB).
    Fallback to simple luminance-channel SSIM approximation if skimage unavailable.
    pred_01/tgt_01 are in [0,1], shape NCHW.
    """
    out = []
    n = pred_01.shape[0]
    for i in range(n):
        p = to_numpy_img01(pred_01[i])
        t = to_numpy_img01(tgt_01[i])

        if _HAS_SKIMAGE:
            # channel_axis for modern skimage
            val = float(skimage_ssim(t, p, data_range=1.0, channel_axis=2))
        else:
            # Fallback: crude similarity from MSE -> pseudo-SSIM-ish score
            mse = float(np.mean((p - t) ** 2))
            val = max(0.0, 1.0 - mse * 10.0)
        out.append(val)
    return out


def compute_basic_metrics(pred_01: torch.Tensor, tgt_01: torch.Tensor) -> Dict[str, List[float]]:
    """
    pred_01, tgt_01: NCHW in [0,1]
    """
    diff = pred_01 - tgt_01
    mae = torch.mean(torch.abs(diff), dim=(1, 2, 3)).detach().cpu().tolist()
    mse = torch.mean(diff * diff, dim=(1, 2, 3)).detach().cpu().tolist()
    rmse = [math.sqrt(max(0.0, m)) for m in mse]
    psnr = [compute_psnr_from_mse(m, 1.0) for m in mse]
    ssim = ssim_batch(pred_01, tgt_01)
    return {
        "mae": [float(v) for v in mae],
        "mse": [float(v) for v in mse],
        "rmse": [float(v) for v in rmse],
        "psnr": [float(v) for v in psnr],
        "ssim": [float(v) for v in ssim],
    }


def aggregate(metrics_list: List[SampleMetrics]) -> AggregateMetrics:
    mae = [m.mae for m in metrics_list]
    mse = [m.mse for m in metrics_list]
    rmse = [m.rmse for m in metrics_list]
    psnr = [m.psnr for m in metrics_list]
    ssim = [m.ssim for m in metrics_list]
    lpips_vals = [m.lpips for m in metrics_list if m.lpips is not None]

    return AggregateMetrics(
        n_samples=len(metrics_list),
        mae_mean=_safe_mean(mae),
        mse_mean=_safe_mean(mse),
        rmse_mean=_safe_mean(rmse),
        psnr_mean=_safe_mean(psnr),
        ssim_mean=_safe_mean(ssim),
        lpips_mean=_safe_mean(lpips_vals) if lpips_vals else None,
        mae_std=_safe_std(mae),
        mse_std=_safe_std(mse),
        rmse_std=_safe_std(rmse),
        psnr_std=_safe_std(psnr),
        ssim_std=_safe_std(ssim),
        lpips_std=_safe_std(lpips_vals) if lpips_vals else None,
    )


# ----------------------------
# Optional feature/FID proxy
# ----------------------------

def compute_feature_stats(
    imgs_01: torch.Tensor,
    feat_extractor: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    imgs_01: NCHW [0,1], RGB
    Returns mean/cov of features as numpy arrays.
    """
    # ImageNet normalization for ResNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    x = (imgs_01.to(device) - mean) / std

    with torch.no_grad():
        f = feat_extractor(x).detach().cpu().numpy().astype(np.float64)

    mu = np.mean(f, axis=0)
    sigma = np.cov(f, rowvar=False)
    return mu, sigma


def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    # Numerical PSD sqrt via eigendecomposition
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def fid_from_stats(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = sqrtm_psd(s1 @ s2)
    fid = float(diff @ diff + np.trace(s1 + s2 - 2.0 * covmean))
    return max(fid, 0.0)


# ----------------------------
# Main benchmark
# ----------------------------

def _extract_generator_state_from_checkpoint(
    raw_state: object,
    direction: str,
) -> Dict[str, torch.Tensor]:
    """
    Accept either:
    1) plain generator state_dict
    2) wrapper with 'state_dict'
    3) full training checkpoint from GAN_train_simple.py containing:
       G_o2t, G_t2o, D_o, D_t, opt_G, opt_D_o, opt_D_t, ...
    and extract the correct generator weights based on direction.
    """
    # unwrap common wrapper
    if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]

    if not isinstance(raw_state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected dict-like state.")

    # Full training checkpoint format
    if "G_o2t" in raw_state or "G_t2o" in raw_state:
        key = "G_o2t" if direction == "o2t" else "G_t2o"
        if key not in raw_state:
            raise RuntimeError(
                f"Checkpoint appears to be full training checkpoint but missing required key '{key}'."
            )
        gen_state = raw_state[key]
        if not isinstance(gen_state, dict):
            raise RuntimeError(f"Checkpoint key '{key}' is not a valid state_dict.")
        return gen_state

    # Already a plain generator state_dict
    return raw_state  # type: ignore[return-value]


def build_generator(
    model_path: Path,
    image_size: int,
    device: torch.device,
    n_res: int,
    direction: str,
) -> nn.Module:
    gen = Generator(in_nc=3, out_nc=3, n_res=n_res).to(device)
    raw_state = torch.load(model_path, map_location="cpu")
    state = _extract_generator_state_from_checkpoint(raw_state, direction=direction)
    gen.load_state_dict(state, strict=True)
    gen.eval()
    return gen


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else n
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    return train_idx, val_idx


def benchmark(
    model_path: Path,
    input_dir: Path,
    target_dir: Path,
    direction: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    n_res: int,
    max_samples: Optional[int],
    use_lpips: bool,
    save_preds_dir: Optional[Path],
    val_ratio: float,
    seed: int,
    enable_fid_proxy: bool,
) -> Dict:
    if not _HAS_TORCHVISION:
        raise RuntimeError("torchvision is required but unavailable in this environment.")

    # Base dataset to determine count
    full_ds = PairedFrameDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        image_size=image_size,
        max_samples=max_samples,
        index_subset=None,
    )
    n_all = len(full_ds)

    if val_ratio > 0.0 and n_all >= 2:
        _, val_idx = split_indices(n_all, val_ratio, seed)
    else:
        val_idx = list(range(n_all))

    ds = PairedFrameDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        image_size=image_size,
        max_samples=max_samples,
        index_subset=val_idx,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    gen = build_generator(
        model_path,
        image_size=image_size,
        device=device,
        n_res=n_res,
        direction=direction,
    )

    lpips_net = None
    if use_lpips:
        if not _HAS_LPIPS:
            raise RuntimeError("LPIPS requested but lpips package is not installed.")
        lpips_net = lpips.LPIPS(net="alex").to(device).eval()

    if save_preds_dir is not None:
        save_preds_dir.mkdir(parents=True, exist_ok=True)

    sample_metrics: List[SampleMetrics] = []

    # Optional FID proxy accumulators
    pred_batches_for_fid: List[torch.Tensor] = []
    tgt_batches_for_fid: List[torch.Tensor] = []
    feat_extractor = None
    if enable_fid_proxy:
        if not _HAS_TORCHVISION:
            raise RuntimeError("FID proxy requires torchvision models.")
        weights = ResNet18_Weights.DEFAULT
        feat_extractor = resnet18(weights=weights)
        feat_extractor.fc = nn.Identity()
        feat_extractor = feat_extractor.to(device).eval()

    tic = time.time()
    seen = 0

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            inp = batch["input"].to(device, non_blocking=True)      # [-1,1]
            tgt = batch["target"].to(device, non_blocking=True)     # [-1,1]

            pred = gen(inp)  # [-1,1]

            pred_01 = clamp01(denorm(pred))
            tgt_01 = clamp01(denorm(tgt))

            basic = compute_basic_metrics(pred_01, tgt_01)
            lpips_vals: Optional[List[float]] = None

            if lpips_net is not None:
                # LPIPS expects [-1,1]
                lp = lpips_net(pred, tgt).view(-1).detach().cpu().tolist()
                lpips_vals = [float(v) for v in lp]

            n = pred.shape[0]
            for i in range(n):
                m = SampleMetrics(
                    mae=basic["mae"][i],
                    mse=basic["mse"][i],
                    rmse=basic["rmse"][i],
                    psnr=basic["psnr"][i],
                    ssim=basic["ssim"][i],
                    lpips=(lpips_vals[i] if lpips_vals is not None else None),
                )
                sample_metrics.append(m)

            if enable_fid_proxy:
                pred_batches_for_fid.append(pred_01.detach().cpu())
                tgt_batches_for_fid.append(tgt_01.detach().cpu())

            if save_preds_dir is not None:
                for i in range(n):
                    idx_global = seen + i
                    pred_img = pred_01[i].cpu()
                    tgt_img = tgt_01[i].cpu()
                    inp_img = clamp01(denorm(inp[i].cpu()))

                    save_image(inp_img, str(save_preds_dir / f"{idx_global:06d}_input.png"))
                    save_image(pred_img, str(save_preds_dir / f"{idx_global:06d}_pred.png"))
                    save_image(tgt_img, str(save_preds_dir / f"{idx_global:06d}_target.png"))

            seen += n

    elapsed = time.time() - tic
    agg = aggregate(sample_metrics)

    result = {
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "target_dir": str(target_dir),
        "n_total_pairs_available": n_all,
        "n_pairs_evaluated": agg.n_samples,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_sec": elapsed,
        "throughput_img_per_sec": float(agg.n_samples / max(elapsed, 1e-9)),
        "metrics": asdict(agg),
        "notes": {
            "lpips_used": bool(lpips_net is not None),
            "skimage_ssim_used": bool(_HAS_SKIMAGE),
            "fid_proxy_enabled": bool(enable_fid_proxy),
        },
    }

    if enable_fid_proxy:
        pred_all = torch.cat(pred_batches_for_fid, dim=0) if pred_batches_for_fid else torch.empty(0, 3, image_size, image_size)
        tgt_all = torch.cat(tgt_batches_for_fid, dim=0) if tgt_batches_for_fid else torch.empty(0, 3, image_size, image_size)

        if pred_all.shape[0] >= 2 and tgt_all.shape[0] >= 2 and feat_extractor is not None:
            mu_p, s_p = compute_feature_stats(pred_all, feat_extractor, device)
            mu_t, s_t = compute_feature_stats(tgt_all, feat_extractor, device)
            fid_proxy = fid_from_stats(mu_p, s_p, mu_t, s_t)
            result["metrics"]["fid_proxy_resnet18"] = float(fid_proxy)
        else:
            result["metrics"]["fid_proxy_resnet18"] = None

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark GAN generator .pt accuracy on paired frames."
    )
    p.add_argument("--model", required=True, type=str, help="Path to generator .pt (state_dict)")
    p.add_argument("--input-dir", required=True, type=str, help="Input frame directory (source domain)")
    p.add_argument("--target-dir", required=True, type=str, help="Target frame directory (ground truth domain)")
    p.add_argument(
        "--direction",
        default="o2t",
        choices=["o2t", "t2o"],
        help="Translation direction label for reporting only",
    )

    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--val-ratio", type=float, default=0.0, help="Evaluate on validation split ratio [0,1). 0 = all")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-res", type=int, default=9, help="Generator residual block count used during training")
    p.add_argument("--cpu", action="store_true", help="Force CPU")

    p.add_argument("--use-lpips", action="store_true", help="Enable LPIPS metric (requires lpips package)")
    p.add_argument("--enable-fid-proxy", action="store_true", help="Enable lightweight FID-style proxy metric")

    p.add_argument("--save-preds-dir", type=str, default=None, help="Optional directory to save input/pred/target images")
    p.add_argument("--output-json", type=str, default=None, help="Optional path to write JSON result")
    p.add_argument("--print-per-sample", action="store_true", help="Print per-sample metrics summary line")

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    save_preds_dir = Path(args.save_preds_dir) if args.save_preds_dir else None

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")
    if not target_dir.is_dir():
        raise SystemExit(f"Target dir not found: {target_dir}")
    if args.val_ratio < 0.0 or args.val_ratio >= 1.0:
        raise SystemExit("--val-ratio must be in [0, 1).")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    result = benchmark(
        model_path=model_path,
        input_dir=input_dir,
        target_dir=target_dir,
        direction=args.direction,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        n_res=args.n_res,
        max_samples=args.max_samples,
        use_lpips=args.use_lpips,
        save_preds_dir=save_preds_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        enable_fid_proxy=args.enable_fid_proxy,
    )
    result["direction"] = args.direction

    # Console summary
    m = result["metrics"]
    print("=== GAN Benchmark Summary ===")
    print(f"Direction:              {result['direction']}")
    print(f"Model:                  {result['model_path']}")
    print(f"Pairs evaluated:        {result['n_pairs_evaluated']} / {result['n_total_pairs_available']}")
    print(f"Device:                 {result['device']}")
    print(f"Elapsed (s):            {result['elapsed_sec']:.3f}")
    print(f"Throughput (img/s):     {result['throughput_img_per_sec']:.2f}")
    print(f"MAE mean ± std:         {m['mae_mean']:.6f} ± {m['mae_std']:.6f}")
    print(f"MSE mean ± std:         {m['mse_mean']:.6f} ± {m['mse_std']:.6f}")
    print(f"RMSE mean ± std:        {m['rmse_mean']:.6f} ± {m['rmse_std']:.6f}")
    print(f"PSNR mean ± std:        {m['psnr_mean']:.4f} ± {m['psnr_std']:.4f}")
    print(f"SSIM mean ± std:        {m['ssim_mean']:.6f} ± {m['ssim_std']:.6f}")
    if m.get("lpips_mean") is not None:
        print(f"LPIPS mean ± std:       {m['lpips_mean']:.6f} ± {m['lpips_std']:.6f}")
    if "fid_proxy_resnet18" in m:
        print(f"FID proxy (ResNet18):   {m['fid_proxy_resnet18']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON report:      {out_path}")


if __name__ == "__main__":
    main()
