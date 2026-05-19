# CycleGAN Clean v1 - Optical to Thermal Translation

## Training
- Architecture: CycleGAN (ResNet-9 Generator, PatchGAN Discriminator)
- Epochs: 20
- Image size: 256x256
- Total frames: 407 (train: 326, test: 81)
- Device: Colab T4 GPU

## Best Checkpoint: epoch_0010
  - PSNR: 25.34 +/- 3.20
  - SSIM: 0.8403 +/- 0.0350
  - MAE:  0.0304
  - RMSE: 0.0574
  - Throughput: 0.7 img/s

## Comparison vs Old Model
| Model | Data | HUD removed | PSNR | SSIM | Notes |
|-------|------|-------------|------|------|-------|
| old epoch_0040 | old frames | no | 25.10 | 0.827 | HUD contaminated |
| epoch_0010 (clean_v1) | cleaned crops | yes | 25.34 | 0.8403 | current run |

Pixel metrics are approximate due to unpaired CycleGAN + independent crops.
Visual quality (see comparison_grid.png) is the primary judge.
