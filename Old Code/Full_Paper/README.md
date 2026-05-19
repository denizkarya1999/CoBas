# COBAS Full Paper - Optical to Thermal Translation

This folder contains the CycleGAN pipeline for translating optical battery imagery to thermal-like outputs, including preprocessing, training, benchmarking, and packaged run artifacts.

## Current final run (clean_v1)

- Stable artifact location: `Full_Paper/results/final_clean_v1`
- Source run archive: `Full_Paper/runs/5-5-2026/gan_run`
- Best checkpoint (by validation metrics): `epoch_0010`
- Best metrics: PSNR `25.34`, SSIM `0.8403`, MAE `0.0304`, RMSE `0.0574`
- Controlled old-vs-clean comparison table: `Full_Paper/results/ablation_table.md`

## Key outputs

- Metrics JSON: `Full_Paper/results/final_clean_v1/metrics.json`
- Run summary README: `Full_Paper/results/final_clean_v1/README.md`
- Main visual grid: `Full_Paper/results/final_clean_v1/comparison_grid.png`
- Summary grid: `Full_Paper/results/final_clean_v1/summary_grid.png`
- Per-sample comparisons: `Full_Paper/results/final_clean_v1/test_outputs/grids`
- Exported generators:
  - `Full_Paper/results/final_clean_v1/checkpoints/G_optical_to_thermal.pt`
  - `Full_Paper/results/final_clean_v1/checkpoints/G_thermal_to_optical.pt`

## Reproduce pipeline

Run commands from `Full_Paper/`.

### 1) Preprocess synced optical + thermal videos

```bash
python Pipeline/Preprocessing/thermal_preprocessing.py \
  --optical data/cobas/o_synced.mp4 \
  --thermal data/cobas/t_synced.mp4 \
  --metadata data/cobas/sync_metadata.json \
  --out-root data/cobas/preprocessed_frames
```

### 2) Train CycleGAN

```bash
python Pipeline/Preprocessing/GAN_train_simple.py \
  --optical-frames data/cobas/preprocessed_frames/opt \
  --thermal-frames data/cobas/preprocessed_frames/therm \
  --work-dir runs/gan_run \
  --epochs 20 \
  --batch-size 4 \
  --image-size 256
```

### 3) Benchmark a generator checkpoint

```bash
python Pipeline/Preprocessing/GAN_benchmark_suite.py \
  --model runs/gan_run/G_optical_to_thermal.pt \
  --input-dir data/cobas/preprocessed_frames/opt \
  --target-dir data/cobas/preprocessed_frames/therm \
  --direction o2t \
  --image-size 256 \
  --batch-size 8 \
  --output-json runs/gan_run/benchmark_results.json \
  --save-preds-dir runs/gan_run/benchmark_preds
```

## Evaluation caveat

Pixel-wise metrics are approximate because this setup uses unpaired CycleGAN assumptions with independently cropped optical and thermal ROIs. Visual quality is the primary judgment signal.
