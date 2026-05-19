## Optical -> Thermal Ablation (Preprocessing Effect)

Controlled comparison using the same CycleGAN setup (ResNet-9 generator, PatchGAN discriminator, 256x256 images).
Primary changed factor: preprocessing quality (HUD contamination vs cleaned crops).

| Dataset | HUD removed | Alignment mode | PSNR | SSIM | Best checkpoint | Notes |
|---|---|---|---:|---:|---|---|
| old frames | no | partial / legacy | 25.10 | 0.8270 | epoch_0040 | HUD contamination present |
| clean_v1 crops | yes | independent ROI crops (unpaired) | 25.34 | 0.8403 | epoch_0010 | current clean run |

### Delta (clean_v1 - old)

- PSNR: +0.24 dB
- SSIM: +0.0133

### Metric caveat

Pixel-wise metrics are approximate because training and evaluation use unpaired CycleGAN assumptions with independently cropped optical and thermal ROIs. Visual quality should be treated as the primary assessment signal.

### Sources

- `Full_Paper/results/final_clean_v1/metrics.json`
- `Full_Paper/runs/5-5-2026/gan_run/deliverables/cobas_cycle_gan_clean_v1/README.md`
