# CoBas inference report

Test of the Battery Reader app's inference on the CoBas Latest Dataset, run on
2026-11-09 on the Raspberry Pi 5 (aarch64, Python 3.13.5, PyTorch 2.15.0 CPU, 4 threads)
after every model was made a PyTorch checkpoint.

## Result

- **1,731 of 1,760 files read correctly (98.4 %)**, with 0 errors, across 16 models.
- On the real app screen, one file per model: **16 of 16** read-outs matched.
- All 110 voice recordings were converted to spectrograms by the app and all were read correctly.
- A live estimate on the real sensors ran end to end with the default model and showed "Battery Level: 50%".

The models were trained on the same Oct 21 recordings this dataset is made from,
so these figures show that the app runs every model correctly on them. They do
not show how the models would do on a new recording day.

## Models

All 16 models are PyTorch checkpoints with the `.pth` extension (`.pt` is accepted
as well). Before this test the 1 - 1 - 1 folder also held an ONNX copy of its
checkpoint, and the app used that copy for live estimates by default. It was
compared with the `.pth` beside it on that model's 110 dataset files (the same
answer on all 110; largest probability difference 5.75e-08) and then removed,
together with the ONNX loading and export code. The default model for live
estimates is now the `.pth` checkpoint.

## What was tested

The dataset is `~/Datasets/CoBas/CoBas Latest Dataset`: for each of the 16 models,
10 files for each of the 11 battery levels (0 % to 100 %), 110 per model. The
standalone models' files are unchanged copies of saved thermal frames, mmWave
frames and voice recordings; the fusion models' files are fused images made with
each checkpoint's own weights. The expected answer is the battery level in the
folder name.

1. **Every file.** All 1,760 files went through the app's Local File Inference
   code, the part behind the screen that loads the chosen model, converts a
   voice recording to its spectrogram and classifies the file. Each file was
   read by its own model.
2. **The screen.** The real app was run on the Pi's display. For each model the
   model was chosen with the picker's buttons and one file from its folder was
   opened with the file browser's buttons, each at a different battery level.
3. **A live estimate.** "Start Estimate" was pressed once with the default
   model: the radar, thermal camera, speaker and microphone captured one chirp
   and the app classified it.

## Results by model

| Models sub-folder | Model | Checkpoint | Correct of 110 | Time per file |
|---|---|---|---|---|
| Standalone | CoBas mmWave Model | `CoBas_mmWave_model.pth` | 110 | 433 ms |
| Standalone | CoBas Thermal Model | `CoBas_thermal_model.pth` | 107 | 161 ms |
| Standalone | CoBas Voice Reflection Recognition Model | `CoBas_voice_reflection_recognition_model.pth` | 110 | 241 ms |
| 1 - 1 Fusions | CoBas mmWave Thermal Fusion Model | `CoBas_mmwave_thermal_fusion_model.pth` | 110 | 145 ms |
| 1 - 1 Fusions | CoBas mmWave and Voice Fusion Model | `CoBas_mmwave_and_voice_fusion_model.pth` | 109 | 149 ms |
| 1 - 1 Fusions | CoBas Thermal and Voice Model | `CoBas_thermal_and_voice_model.pth` | 106 | 153 ms |
| 1 - 2 Fusions | CoBas mmWave Thermal Fusion Model | `CoBas_mmwave_thermal_fusion_model.pth` | 110 | 150 ms |
| 1 - 2 Fusions | CoBas mmWave and Voice Fusion Model | `CoBas_mmwave_and_voice_fusion_model.pth` | 110 | 149 ms |
| 1 - 2 Fusions | CoBas Thermal and Voice Model | `CoBas_thermal_and_voice_model.pth` | 102 | 152 ms |
| 2 - 1 Fusions | CoBas mmWave Thermal Fusion Model | `CoBas_mmwave_thermal_fusion_model.pth` | 110 | 155 ms |
| 2 - 1 Fusions | CoBas mmWave and Voice Fusion Model | `CoBas_mmwave_and_voice_fusion_model.pth` | 107 | 154 ms |
| 2 - 1 Fusions | CoBas Thermal and Voice Model | `CoBas_thermal_and_voice_model.pth` | 100 | 154 ms |
| 1 - 1 - 1 Fusions | CoBas mmWave Thermal Voice Fusion Model | `CoBas_mmwave_thermal_voice_fusion_model.pth` | 110 | 154 ms |
| 1 - 1 - 2 Fusions | CoBas mmWave Thermal Voice Fusion Model | `CoBas_mmwave_thermal_voice_fusion_model.pth` | 110 | 154 ms |
| 1 - 2 - 1 Fusions | CoBas mmWave Thermal Voice Fusion Model | `CoBas_mmwave_thermal_voice_fusion_model.pth` | 110 | 164 ms |
| 2 - 1 - 1 Fusions | CoBas mmWave Thermal Voice Fusion Model | `CoBas_mmwave_thermal_voice_fusion_model.pth` | 110 | 155 ms |
| | **All 16 models** | | **1,731 of 1,760** | |

Time per file is the whole step on the Pi: reading the file, preparing it
(for a voice recording, computing the spectrogram) and running the network.

## Results by battery level

| Battery level | Correct of 160 |
|---|---|
| 0 % | 160 |
| 10 % | 160 |
| 20 % | 149 |
| 30 % | 159 |
| 40 % | 160 |
| 50 % | 160 |
| 60 % | 149 |
| 70 % | 159 |
| 80 % | 157 |
| 90 % | 158 |
| 100 % | 160 |

## The 29 misses

| Model | File | Recorded | Read as |
|---|---|---|---|
| Standalone / CoBas Thermal Model | `Thermal_Image_80_Percent_Position_2_001.png` | 80 % | 100 % |
| Standalone / CoBas Thermal Model | `Thermal_Image_80_Percent_Position_2_021.png` | 80 % | 100 % |
| Standalone / CoBas Thermal Model | `Thermal_Image_80_Percent_Position_2_041.png` | 80 % | 100 % |
| 1 - 1 Fusions / CoBas mmWave and Voice Fusion Model | `Fused_Image_20_Percent_Position_4_001.png` | 20 % | 30 % |
| 1 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_1_011.png` | 20 % | 10 % |
| 1 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_3_011.png` | 20 % | 10 % |
| 1 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_3_031.png` | 20 % | 30 % |
| 1 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_4_041.png` | 20 % | 30 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_2_001.png` | 60 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_2_041.png` | 60 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_3_011.png` | 60 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_4_001.png` | 60 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_4_021.png` | 60 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_70_Percent_Position_4_041.png` | 70 % | 80 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_90_Percent_Position_2_001.png` | 90 % | 100 % |
| 1 - 2 Fusions / CoBas Thermal and Voice Model | `Fused_Image_90_Percent_Position_4_001.png` | 90 % | 100 % |
| 2 - 1 Fusions / CoBas mmWave and Voice Fusion Model | `Fused_Image_20_Percent_Position_4_001.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas mmWave and Voice Fusion Model | `Fused_Image_20_Percent_Position_4_041.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas mmWave and Voice Fusion Model | `Fused_Image_30_Percent_Position_4_001.png` | 30 % | 20 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_1_011.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_3_011.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_4_001.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_20_Percent_Position_4_041.png` | 20 % | 30 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_2_001.png` | 60 % | 100 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_2_041.png` | 60 % | 100 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_3_011.png` | 60 % | 90 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_4_001.png` | 60 % | 100 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_4_021.png` | 60 % | 100 % |
| 2 - 1 Fusions / CoBas Thermal and Voice Model | `Fused_Image_60_Percent_Position_4_041.png` | 60 % | 100 % |

The Thermal and Voice models account for 22 of them, the mmWave and Voice models
for 4 and the Thermal model for 3. These are the models' own errors: the two
weakest models give the same counts (102 and 100) when the raw chirps are
classified directly, and a fused image opened from disk reaches the network
exactly as the chirp would.

## The screen check

| Model chosen in the picker | File opened | Folder says | Screen showed | Seconds |
|---|---|---|---|---|
| Standalone · mmWave only (pth) | `mmWave_Frame_30_Percent_Position_1_011.jpg` | 30 % | 30 % | 0.8 |
| Standalone · Thermal only (pth) | `Thermal_Image_70_Percent_Position_2_021.png` | 70 % | 70 % | 0.2 |
| Standalone · Voice only (pth) | `Voice_Recording_0_Percent_Position_3_031.wav` | 0 % | 0 % | 1.4 |
| 1 - 1 Fusions · mmWave + Thermal (pth) | `Fused_Image_40_Percent_Position_4_041.png` | 40 % | 40 % | 0.6 |
| 1 - 1 Fusions · mmWave + Voice (pth) | `Fused_Image_80_Percent_Position_2_001.png` | 80 % | 80 % | 0.2 |
| 1 - 1 Fusions · Thermal + Voice (pth) | `Fused_Image_10_Percent_Position_3_011.png` | 10 % | 10 % | 0.2 |
| 1 - 2 Fusions · mmWave + Thermal (pth) | `Fused_Image_50_Percent_Position_4_021.png` | 50 % | 50 % | 0.2 |
| 1 - 2 Fusions · mmWave + Voice (pth) | `Fused_Image_90_Percent_Position_1_031.png` | 90 % | 90 % | 0.2 |
| 1 - 2 Fusions · Thermal + Voice (pth) | `Fused_Image_20_Percent_Position_2_041.png` | 20 % | 20 % | 0.2 |
| 2 - 1 Fusions · mmWave + Thermal (pth) | `Fused_Image_60_Percent_Position_4_001.png` | 60 % | 60 % | 0.2 |
| 2 - 1 Fusions · mmWave + Voice (pth) | `Fused_Image_100_Percent_Position_1_011.png` | 100 % | 100 % | 0.2 |
| 2 - 1 Fusions · Thermal + Voice (pth) | `Fused_Image_30_Percent_Position_2_021.png` | 30 % | 30 % | 0.3 |
| 1 - 1 - 1 Fusions · mmWave + Thermal + Voice (pth) | `Fused_Image_70_Percent_Position_3_031.png` | 70 % | 70 % | 0.2 |
| 1 - 1 - 2 Fusions · mmWave + Thermal + Voice (pth) | `Fused_Image_0_Percent_Position_4_041.png` | 0 % | 0 % | 0.2 |
| 1 - 2 - 1 Fusions · mmWave + Thermal + Voice (pth) | `Fused_Image_40_Percent_Position_2_001.png` | 40 % | 40 % | 0.2 |
| 2 - 1 - 1 Fusions · mmWave + Thermal + Voice (pth) | `Fused_Image_80_Percent_Position_3_011.png` | 80 % | 80 % | 0.2 |

Seconds run from tapping the file to the read-out and include loading the model.
No pointer or keyboard input reached the app during the run.

## The live estimate

- Model: 1 - 1 - 1 Fusions · mmWave + Thermal + Voice (pth) (`CoBas_mmwave_thermal_voice_fusion_model.pth`), the app's default.
- Read-out: "Battery Level: 50%", 7.3 seconds after pressing Start Estimate.
- Nothing was left on disk: the temporary capture folder was removed.

The charge of the battery on the rig was not known to the test, so this shows
that the live path works with the `.pth` model, not that the reading is right.

## Repeating the test

```bash
.venv/bin/python -m cobas_inference check-latest-dataset "/home/tailab/Datasets/CoBas/CoBas Latest Dataset"
```

run from `CoBas/Inference` classifies every file with its own model and prints
the count and the misses for each model. It takes about five and a half minutes
on the Pi.
