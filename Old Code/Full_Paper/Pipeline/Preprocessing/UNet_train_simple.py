#!/usr/bin/env python3
"""
Standalone simple Pix2Pix-style U-Net trainer for optical -> thermal translation.

This script mirrors GAN_train_simple.py ergonomics while using paired supervised
training with a conditional GAN (PatchGAN discriminator).

Features:
- Accepts either synced videos OR pre-extracted frame folders
- Builds paired datasets by sorted filename/time order
- Trains Pix2Pix (U-Net generator + PatchGAN discriminator)
- Basic logging, sample outputs, checkpoints, and resume support

Example usage (videos):
    python3 UNet_train_simple.py \
      --optical-video /path/o_synced.mp4 \
      --thermal-video /path/t_synced.mp4 \
      --work-dir /tmp/unet_pix2pix_run \
      --epochs 50 --batch-size 4

Example usage (frame folders):
    python3 UNet_train_simple.py \
      --optical-frames /path/optical_frames \
      --thermal-frames /path/thermal_frames \
      --work-dir /tmp/unet_pix2pix_run \
      --epochs 50 --batch-size 4

Resume training:
    python3 UNet_train_simple.py ... --resume /tmp/unet_pix2pix_run/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
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
    return (x.clamp(-1, 1) + 1.0) * 0.5


def extract_synced_frames(
    optical_video: Path,
    thermal_video: Path,
    out_optical_dir: Path,
    out_thermal_dir: Path,
    max_frames: Optional[int] = None,
) -> Tuple[int, int]:
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

        cv2.imwrite(str(out_optical_dir / f"{count:06d}.jpg"), frame_o, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_thermal_dir / f"{count:06d}.jpg"), frame_t, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    cap_o.release()
    cap_t.release()
    return count, count


class PairedFrameDataset(Dataset):
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


class UNetDown(nn.Module):
    def __init__(self, in_c: int, out_c: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetUp(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """Pix2Pix-style U-Net generator for 256x256."""

    def __init__(self, in_nc: int = 3, out_nc: int = 3, base_c: int = 64):
        super().__init__()
        c = base_c

        self.d1 = UNetDown(in_nc, c, normalize=False)
        self.d2 = UNetDown(c, c * 2)
        self.d3 = UNetDown(c * 2, c * 4)
        self.d4 = UNetDown(c * 4, c * 8)
        self.d5 = UNetDown(c * 8, c * 8)
        self.d6 = UNetDown(c * 8, c * 8)
        self.d7 = UNetDown(c * 8, c * 8)
        self.d8 = UNetDown(c * 8, c * 8, normalize=False)

        self.u1 = UNetUp(c * 8, c * 8, dropout=0.5)
        self.u2 = UNetUp(c * 16, c * 8, dropout=0.5)
        self.u3 = UNetUp(c * 16, c * 8, dropout=0.5)
        self.u4 = UNetUp(c * 16, c * 8)
        self.u5 = UNetUp(c * 16, c * 4)
        self.u6 = UNetUp(c * 8, c * 2)
        self.u7 = UNetUp(c * 4, c)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_nc, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


class DiscBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 2, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator operating on concatenated (input, target) image pair."""

    def __init__(self, in_nc: int = 3):
        super().__init__()
        c_in = in_nc * 2
        self.net = nn.Sequential(
            DiscBlock(c_in, 64, stride=2, norm=False),
            DiscBlock(64, 128, stride=2, norm=True),
            DiscBlock(128, 256, stride=2, norm=True),
            DiscBlock(256, 512, stride=1, norm=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cond, target], dim=1)
        return self.net(x)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


def save_checkpoint(
    ckpt_path: Path,
    state: TrainState,
    G: nn.Module,
    D: nn.Module,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
) -> None:
    payload = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    G: nn.Module,
    D: nn.Module,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
) -> TrainState:
    payload = torch.load(ckpt_path, map_location="cpu")
    G.load_state_dict(payload["G"])
    D.load_state_dict(payload["D"])
    opt_G.load_state_dict(payload["opt_G"])
    opt_D.load_state_dict(payload["opt_D"])
    return TrainState(epoch=int(payload["epoch"]), global_step=int(payload["global_step"]))


def make_real_fake_targets(pred_shape: torch.Size, device: torch.device):
    real = torch.ones(pred_shape, device=device)
    fake = torch.zeros(pred_shape, device=device)
    return real, fake


def save_sample_grid(
    out_path: Path,
    optical: torch.Tensor,
    fake_t: torch.Tensor,
    thermal: torch.Tensor,
    max_items: int = 4,
) -> None:
    n = min(max_items, optical.shape[0])
    rows = []
    for i in range(n):
        row = torch.stack(
            [
                denorm(optical[i]),
                denorm(fake_t[i]),
                denorm(thermal[i]),
            ],
            dim=0,
        )
        rows.append(row)
    grid = make_grid(torch.cat(rows, dim=0), nrow=3)
    save_image(grid, str(out_path))


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

    G = UNetGenerator(in_nc=3, out_nc=3, base_c=args.base_channels).to(device)
    D = PatchDiscriminator(in_nc=3).to(device)

    if args.gan_loss == "bce":
        criterion_gan = nn.BCEWithLogitsLoss()
    else:
        criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    state = TrainState(epoch=0, global_step=0)
    if args.resume:
        state = load_checkpoint(Path(args.resume), G, D, opt_G, opt_D)
        print(f"[resume] epoch={state.epoch} step={state.global_step} from {args.resume}")

    run_cfg = vars(args).copy()
    run_cfg["dataset_pairs"] = len(ds)
    run_cfg["device"] = str(device)
    save_json(run_cfg, work_dir / "run_config.json")

    G.train()
    D.train()

    start_epoch = state.epoch + 1
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        loss_G_meter = 0.0
        loss_D_meter = 0.0
        loss_G_gan_meter = 0.0
        loss_G_l1_meter = 0.0
        steps = 0

        for batch in dl:
            real_o = batch["optical"].to(device, non_blocking=True)
            real_t = batch["thermal"].to(device, non_blocking=True)

            # Train D
            opt_D.zero_grad(set_to_none=True)
            fake_t_detached = G(real_o).detach()

            pred_real = D(real_o, real_t)
            pred_fake = D(real_o, fake_t_detached)
            real_lbl, fake_lbl = make_real_fake_targets(pred_real.shape, device)

            loss_D_real = criterion_gan(pred_real, real_lbl)
            loss_D_fake = criterion_gan(pred_fake, fake_lbl)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad(set_to_none=True)
            fake_t = G(real_o)
            pred_fake_for_g = D(real_o, fake_t)
            real_lbl_g, _ = make_real_fake_targets(pred_fake_for_g.shape, device)

            loss_G_gan = criterion_gan(pred_fake_for_g, real_lbl_g)
            loss_G_l1 = criterion_l1(fake_t, real_t) * args.lambda_l1
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            opt_G.step()

            steps += 1
            state.global_step += 1

            loss_G_meter += float(loss_G.item())
            loss_D_meter += float(loss_D.item())
            loss_G_gan_meter += float(loss_G_gan.item())
            loss_G_l1_meter += float(loss_G_l1.item())

            if state.global_step % args.log_every == 0:
                print(
                    f"[train] ep={epoch}/{total_epochs} step={state.global_step} "
                    f"loss_G={loss_G.item():.4f} loss_D={loss_D.item():.4f} "
                    f"loss_G_gan={loss_G_gan.item():.4f} loss_G_l1={loss_G_l1.item():.4f}"
                )

        state.epoch = epoch
        ep_loss_G = loss_G_meter / max(1, steps)
        ep_loss_D = loss_D_meter / max(1, steps)
        ep_loss_G_gan = loss_G_gan_meter / max(1, steps)
        ep_loss_G_l1 = loss_G_l1_meter / max(1, steps)
        dt = time.time() - t0

        print(
            f"[epoch] {epoch}/{total_epochs} done in {dt:.1f}s "
            f"loss_G={ep_loss_G:.4f} loss_D={ep_loss_D:.4f} "
            f"loss_G_gan={ep_loss_G_gan:.4f} loss_G_l1={ep_loss_G_l1:.4f}"
        )

        with torch.no_grad():
            G.eval()
            sample_batch = next(iter(dl))
            s_o = sample_batch["optical"].to(device)
            s_t = sample_batch["thermal"].to(device)
            s_fake_t = G(s_o)

            sample_path = sample_dir / f"epoch_{epoch:04d}.jpg"
            save_sample_grid(sample_path, s_o, s_fake_t, s_t)
            print(f"[sample] {sample_path}")
            G.train()

        if epoch % args.save_every == 0 or epoch == total_epochs:
            ckpt_last = ckpt_dir / "last.pt"
            ckpt_ep = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_last, state, G, D, opt_G, opt_D)
            save_checkpoint(ckpt_ep, state, G, D, opt_G, opt_D)
            print(f"[ckpt] saved {ckpt_last} and {ckpt_ep}")

    torch.save(G.state_dict(), work_dir / "UNet_optical_to_thermal.pt")
    print(f"[done] exported generator to {work_dir / 'UNet_optical_to_thermal.pt'}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple standalone Pix2Pix trainer (U-Net + PatchGAN) using synced videos or frame folders."
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
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--gan-loss", type=str, default="mse", choices=["mse", "bce"])
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

    if args.image_size != 256:
        print(
            "[warn] current U-Net depth is pix2pix-256 style; non-256 sizes may still run but architecture is tuned for 256."
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
