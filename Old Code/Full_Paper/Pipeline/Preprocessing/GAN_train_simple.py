#!/usr/bin/env python3
"""
Standalone simple CycleGAN trainer for optical <-> thermal translation.

This script is designed for remote execution with minimal setup:
- Accepts either synced videos OR pre-extracted frame folders
- Builds paired datasets by sorted filename/time order
- Trains a CycleGAN with basic logging, sample outputs, and checkpoints

Example usage (videos):
    python3 GAN_train_simple.py \
      --optical-video /path/o_synced.mp4 \
      --thermal-video /path/t_synced.mp4 \
      --work-dir /tmp/cyclegan_run \
      --epochs 50 --batch-size 4

Example usage (frame folders):
    python3 GAN_train_simple.py \
      --optical-frames /path/optical_frames \
      --thermal-frames /path/thermal_frames \
      --work-dir /tmp/cyclegan_run \
      --epochs 50 --batch-size 4

Resume training:
    python3 GAN_train_simple.py ... --resume /tmp/cyclegan_run/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


# ----------------------------
# Utilities
# ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def sorted_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p)])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def denorm(x: torch.Tensor) -> torch.Tensor:
    # from [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


# ----------------------------
# Video frame extraction
# ----------------------------

def extract_synced_frames(
    optical_video: Path,
    thermal_video: Path,
    out_optical_dir: Path,
    out_thermal_dir: Path,
    max_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Extract paired frames by index from synced videos.
    Assumes videos are already synchronized and frame counts are close/equal.
    """
    ensure_dir(out_optical_dir)
    ensure_dir(out_thermal_dir)

    cap_o = cv2.VideoCapture(str(optical_video))
    cap_t = cv2.VideoCapture(str(thermal_video))
    if not cap_o.isOpened():
        raise RuntimeError(f"Cannot open optical video: {optical_video}")
    if not cap_t.isOpened():
        raise RuntimeError(f"Cannot open thermal video: {thermal_video}")

    count = 0
    while True:
        ok_o, frame_o = cap_o.read()
        ok_t, frame_t = cap_t.read()
        if not ok_o or not ok_t:
            break
        if max_frames is not None and count >= max_frames:
            break

        # save as jpg for space/runtime balance
        cv2.imwrite(str(out_optical_dir / f"{count:06d}.jpg"), frame_o, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_thermal_dir / f"{count:06d}.jpg"), frame_t, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    cap_o.release()
    cap_t.release()
    return count, count


# ----------------------------
# Dataset
# ----------------------------

class PairedFrameDataset(Dataset):
    """
    Sorted paired dataset:
    optical[i] corresponds to thermal[i].
    """

    def __init__(self, optical_dir: Path, thermal_dir: Path, image_size: int = 256):
        self.optical_paths = sorted_images(optical_dir)
        self.thermal_paths = sorted_images(thermal_dir)

        if len(self.optical_paths) == 0 or len(self.thermal_paths) == 0:
            raise RuntimeError(
                f"No images found. optical={len(self.optical_paths)} thermal={len(self.thermal_paths)}"
            )

        self.n = min(len(self.optical_paths), len(self.thermal_paths))
        self.optical_paths = self.optical_paths[: self.n]
        self.thermal_paths = self.thermal_paths[: self.n]

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        o = Image.open(self.optical_paths[idx]).convert("RGB")
        t = Image.open(self.thermal_paths[idx]).convert("RGB")
        return {
            "optical": self.tf(o),
            "thermal": self.tf(t),
            "optical_path": str(self.optical_paths[idx]),
            "thermal_path": str(self.thermal_paths[idx]),
        }


# ----------------------------
# CycleGAN Models
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


class DiscBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_nc=3):
        super().__init__()
        self.net = nn.Sequential(
            DiscBlock(in_nc, 64, stride=2, norm=False),
            DiscBlock(64, 128, stride=2, norm=True),
            DiscBlock(128, 256, stride=2, norm=True),
            DiscBlock(256, 512, stride=1, norm=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Training helpers
# ----------------------------

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


def save_checkpoint(
    ckpt_path: Path,
    state: TrainState,
    G_o2t: nn.Module,
    G_t2o: nn.Module,
    D_o: nn.Module,
    D_t: nn.Module,
    opt_G: optim.Optimizer,
    opt_D_o: optim.Optimizer,
    opt_D_t: optim.Optimizer,
) -> None:
    payload = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "G_o2t": G_o2t.state_dict(),
        "G_t2o": G_t2o.state_dict(),
        "D_o": D_o.state_dict(),
        "D_t": D_t.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D_o": opt_D_o.state_dict(),
        "opt_D_t": opt_D_t.state_dict(),
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    G_o2t: nn.Module,
    G_t2o: nn.Module,
    D_o: nn.Module,
    D_t: nn.Module,
    opt_G: optim.Optimizer,
    opt_D_o: optim.Optimizer,
    opt_D_t: optim.Optimizer,
) -> TrainState:
    payload = torch.load(ckpt_path, map_location="cpu")
    G_o2t.load_state_dict(payload["G_o2t"])
    G_t2o.load_state_dict(payload["G_t2o"])
    D_o.load_state_dict(payload["D_o"])
    D_t.load_state_dict(payload["D_t"])
    opt_G.load_state_dict(payload["opt_G"])
    opt_D_o.load_state_dict(payload["opt_D_o"])
    opt_D_t.load_state_dict(payload["opt_D_t"])
    return TrainState(epoch=int(payload["epoch"]), global_step=int(payload["global_step"]))


def make_real_fake_targets(pred_shape: torch.Size, device: torch.device):
    real = torch.ones(pred_shape, device=device)
    fake = torch.zeros(pred_shape, device=device)
    return real, fake


def save_sample_grid(
    out_path: Path,
    optical: torch.Tensor,
    thermal: torch.Tensor,
    fake_t: torch.Tensor,
    fake_o: torch.Tensor,
    rec_o: torch.Tensor,
    rec_t: torch.Tensor,
    max_items: int = 4,
) -> None:
    n = min(max_items, optical.shape[0])
    rows = []
    for i in range(n):
        row = torch.stack(
            [
                denorm(optical[i]),
                denorm(fake_t[i]),
                denorm(rec_o[i]),
                denorm(thermal[i]),
                denorm(fake_o[i]),
                denorm(rec_t[i]),
            ],
            dim=0,
        )
        rows.append(row)
    grid = make_grid(torch.cat(rows, dim=0), nrow=6)
    save_image(grid, str(out_path))


# ----------------------------
# Main train
# ----------------------------

def train(args):
    seed_everything(args.seed)

    work_dir = Path(args.work_dir)
    frames_dir = work_dir / "frames"
    optical_frames_dir = frames_dir / "optical"
    thermal_frames_dir = frames_dir / "thermal"
    ckpt_dir = work_dir / "checkpoints"
    sample_dir = work_dir / "samples"
    ensure_dir(work_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    # Prepare frame inputs
    if args.optical_video and args.thermal_video:
        if args.refresh_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)
        ensure_dir(optical_frames_dir)
        ensure_dir(thermal_frames_dir)
        n_o, n_t = extract_synced_frames(
            optical_video=Path(args.optical_video),
            thermal_video=Path(args.thermal_video),
            out_optical_dir=optical_frames_dir,
            out_thermal_dir=thermal_frames_dir,
            max_frames=args.max_frames,
        )
        print(f"[data] extracted frames: optical={n_o}, thermal={n_t}")
    else:
        optical_frames_dir = Path(args.optical_frames)
        thermal_frames_dir = Path(args.thermal_frames)

    ds = PairedFrameDataset(
        optical_dir=optical_frames_dir,
        thermal_dir=thermal_frames_dir,
        image_size=args.image_size,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[data] pairs={len(ds)} batches/epoch={len(dl)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] {device}")

    # Models
    n_res = 9 if args.image_size >= 256 else 6
    G_o2t = Generator(3, 3, n_res=n_res).to(device)
    G_t2o = Generator(3, 3, n_res=n_res).to(device)
    D_o = Discriminator(3).to(device)
    D_t = Discriminator(3).to(device)

    # Losses
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Opts
    opt_G = optim.Adam(
        list(G_o2t.parameters()) + list(G_t2o.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    opt_D_o = optim.Adam(D_o.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D_t = optim.Adam(D_t.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    state = TrainState(epoch=0, global_step=0)
    if args.resume:
        state = load_checkpoint(
            Path(args.resume), G_o2t, G_t2o, D_o, D_t, opt_G, opt_D_o, opt_D_t
        )
        print(f"[resume] epoch={state.epoch} step={state.global_step} from {args.resume}")

    # Save run config
    run_cfg = vars(args).copy()
    run_cfg["dataset_pairs"] = len(ds)
    run_cfg["device"] = str(device)
    save_json(run_cfg, work_dir / "run_config.json")

    G_o2t.train()
    G_t2o.train()
    D_o.train()
    D_t.train()

    start_epoch = state.epoch + 1
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        loss_G_meter = 0.0
        loss_D_meter = 0.0
        steps = 0

        for batch in dl:
            real_o = batch["optical"].to(device, non_blocking=True)
            real_t = batch["thermal"].to(device, non_blocking=True)

            # --------------------
            # Generators
            # --------------------
            opt_G.zero_grad(set_to_none=True)

            fake_t = G_o2t(real_o)
            rec_o = G_t2o(fake_t)

            fake_o = G_t2o(real_t)
            rec_t = G_o2t(fake_o)

            idt_o = G_t2o(real_o)
            idt_t = G_o2t(real_t)

            pred_fake_t = D_t(fake_t)
            pred_fake_o = D_o(fake_o)
            real_lbl_t, fake_lbl_t = make_real_fake_targets(pred_fake_t.shape, device)
            real_lbl_o, fake_lbl_o = make_real_fake_targets(pred_fake_o.shape, device)

            loss_gan_o2t = criterion_gan(pred_fake_t, real_lbl_t)
            loss_gan_t2o = criterion_gan(pred_fake_o, real_lbl_o)

            loss_cycle_o = criterion_cycle(rec_o, real_o) * args.lambda_cycle
            loss_cycle_t = criterion_cycle(rec_t, real_t) * args.lambda_cycle

            loss_idt_o = criterion_identity(idt_o, real_o) * args.lambda_id
            loss_idt_t = criterion_identity(idt_t, real_t) * args.lambda_id

            loss_G = (
                loss_gan_o2t
                + loss_gan_t2o
                + loss_cycle_o
                + loss_cycle_t
                + loss_idt_o
                + loss_idt_t
            )
            loss_G.backward()
            opt_G.step()

            # --------------------
            # D_t
            # --------------------
            opt_D_t.zero_grad(set_to_none=True)
            pred_real_t = D_t(real_t)
            pred_fake_t_det = D_t(fake_t.detach())
            real_lbl_t2, fake_lbl_t2 = make_real_fake_targets(pred_real_t.shape, device)
            loss_D_t_real = criterion_gan(pred_real_t, real_lbl_t2)
            loss_D_t_fake = criterion_gan(pred_fake_t_det, fake_lbl_t2)
            loss_D_t = 0.5 * (loss_D_t_real + loss_D_t_fake)
            loss_D_t.backward()
            opt_D_t.step()

            # --------------------
            # D_o
            # --------------------
            opt_D_o.zero_grad(set_to_none=True)
            pred_real_o = D_o(real_o)
            pred_fake_o_det = D_o(fake_o.detach())
            real_lbl_o2, fake_lbl_o2 = make_real_fake_targets(pred_real_o.shape, device)
            loss_D_o_real = criterion_gan(pred_real_o, real_lbl_o2)
            loss_D_o_fake = criterion_gan(pred_fake_o_det, fake_lbl_o2)
            loss_D_o = 0.5 * (loss_D_o_real + loss_D_o_fake)
            loss_D_o.backward()
            opt_D_o.step()

            loss_D = 0.5 * (loss_D_t + loss_D_o)

            steps += 1
            state.global_step += 1
            loss_G_meter += float(loss_G.item())
            loss_D_meter += float(loss_D.item())

            if state.global_step % args.log_every == 0:
                print(
                    f"[train] ep={epoch}/{total_epochs} step={state.global_step} "
                    f"loss_G={loss_G.item():.4f} loss_D={loss_D.item():.4f}"
                )

        # epoch end
        state.epoch = epoch
        ep_loss_G = loss_G_meter / max(1, steps)
        ep_loss_D = loss_D_meter / max(1, steps)
        dt = time.time() - t0

        print(
            f"[epoch] {epoch}/{total_epochs} done in {dt:.1f}s "
            f"loss_G={ep_loss_G:.4f} loss_D={ep_loss_D:.4f}"
        )

        # Sample visualization
        with torch.no_grad():
            G_o2t.eval()
            G_t2o.eval()
            sample_batch = next(iter(dl))
            s_o = sample_batch["optical"].to(device)
            s_t = sample_batch["thermal"].to(device)
            s_fake_t = G_o2t(s_o)
            s_rec_o = G_t2o(s_fake_t)
            s_fake_o = G_t2o(s_t)
            s_rec_t = G_o2t(s_fake_o)

            sample_path = sample_dir / f"epoch_{epoch:04d}.jpg"
            save_sample_grid(sample_path, s_o, s_t, s_fake_t, s_fake_o, s_rec_o, s_rec_t)
            print(f"[sample] {sample_path}")
            G_o2t.train()
            G_t2o.train()

        # Checkpoints
        if epoch % args.save_every == 0 or epoch == total_epochs:
            ckpt_last = ckpt_dir / "last.pt"
            ckpt_ep = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_last, state, G_o2t, G_t2o, D_o, D_t, opt_G, opt_D_o, opt_D_t)
            save_checkpoint(ckpt_ep, state, G_o2t, G_t2o, D_o, D_t, opt_G, opt_D_o, opt_D_t)
            print(f"[ckpt] saved {ckpt_last} and {ckpt_ep}")

    # Export inference-only generator weights
    torch.save(G_o2t.state_dict(), work_dir / "G_optical_to_thermal.pt")
    torch.save(G_t2o.state_dict(), work_dir / "G_thermal_to_optical.pt")
    print(f"[done] exported generators to {work_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple standalone CycleGAN trainer using synced videos or frame folders."
    )

    src = p.add_argument_group("Data source (choose one mode)")
    src.add_argument("--optical-video", type=str, default=None, help="Path to synced optical video")
    src.add_argument("--thermal-video", type=str, default=None, help="Path to synced thermal video")
    src.add_argument("--optical-frames", type=str, default=None, help="Path to optical frame folder")
    src.add_argument("--thermal-frames", type=str, default=None, help="Path to thermal frame folder")
    src.add_argument("--max-frames", type=int, default=None, help="Optional cap when extracting from videos")
    src.add_argument("--refresh-frames", action="store_true", help="Delete and re-extract frames in work-dir/frames")

    p.add_argument("--work-dir", type=str, required=True, help="Output run directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-id", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = p.parse_args()

    video_mode = args.optical_video is not None or args.thermal_video is not None
    frame_mode = args.optical_frames is not None or args.thermal_frames is not None

    if video_mode and frame_mode:
        raise SystemExit("Use either video inputs OR frame-folder inputs, not both.")

    if video_mode:
        if not args.optical_video or not args.thermal_video:
            raise SystemExit("Both --optical-video and --thermal-video are required in video mode.")
    elif frame_mode:
        if not args.optical_frames or not args.thermal_frames:
            raise SystemExit("Both --optical-frames and --thermal-frames are required in frame mode.")
    else:
        raise SystemExit(
            "No data source provided. Use --optical-video/--thermal-video or --optical-frames/--thermal-frames."
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
