# CoBas Inference

On-device inference for the CoBas battery-level fusion models (ResNet-18 on a
fused mmWave + thermal + voice-reflection image). This package turns the files
the Battery Reader app writes for one chirp into the exact tensor each
checkpoint was trained on, runs the network, and reports the estimated battery
level (0 %–100 % in 10 % steps).

## Install (Raspberry Pi 5, app virtual environment)

```bash
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r "CoBas/Inference/requirements.txt"
```

Place checkpoints under `CoBas/Inference/models/<ratio folder>/<name>.pth`,
mirroring the Drive layout (for example
`models/1 - 1 - 1 Fusions/CoBas_mmwave_thermal_voice_fusion_model.pth`). The
ratio folder is what tells the Thermal-Voice family its fusion weights; the
other families carry their weights inside the checkpoint.

## Command line

Run from `CoBas/Inference`:

```bash
.venv/bin/python -m cobas_inference inspect --model "models/1 - 1 - 1 Fusions/CoBas_mmwave_thermal_voice_fusion_model.pth"
```

```bash
.venv/bin/python -m cobas_inference predict --model MODEL.pth --mmwave frame_000001.jpg --thermal Thermal_Image_100_Percent_Position_1_001.png --voice CoBas_V1_PulseVoice_..._Position_1_001.wav
```

```bash
.venv/bin/python -m cobas_inference replay --model MODEL.pth "/home/tailab/Datasets/CoBas/10 Battery 4 Positions Outside Dataset" --csv replay.csv
```

```bash
.venv/bin/python -m cobas_inference export-onnx --model MODEL.pth --out MODEL.onnx
```

`replay` walks `<N>_Percent_Battery/Session_*/` folders, pairs the k-th thermal
frame and voice recording of each position with the matching global mmWave
frame (exactly as the training notebooks paired them), prints a per-session
vote and per-chirp accuracy when the folder name carries the label, and can
write every prediction to CSV.

```bash
.venv/bin/python -m cobas_inference replay --as-images --model "models/Standalone/CoBas_thermal_model.pth" "/home/tailab/Datasets/CoBas/10 Battery 4 Positions Outside Dataset"
```

`--as-images` runs the same walk through the Local Image Inference recipe: one
picture per chirp, opened as that screen opens it. The thermal and mmWave
standalone models open the saved frame itself; a session holds no ready-made
picture for the other models, so their fused image (or voice spectrogram) is
written to a PNG first and that file is opened.

## Python API

```python
from cobas_inference import Predictor, VoteAggregator
from cobas_inference.session import discover_session

predictor = Predictor.from_model("models/1 - 1 - 1 Fusions/CoBas_mmwave_thermal_voice_fusion_model.pth")
votes = VoteAggregator(predictor.class_names)
for chirp in discover_session("Captures/100_Percent_Battery/Session_..."):
    prediction = predictor.predict_triplet(chirp)   # needs the modalities the model uses
    votes.add(prediction)
print(votes.result().label, votes.consistency())
```

## Inference Mode in the Battery Reader app

The app (`run-cobas.sh`) now asks at startup whether to run in **Data
Collection** (unchanged: labelled sessions for training) or **Inference**.
Set `COBAS_APP_MODE=collect` or `COBAS_APP_MODE=infer` to skip the question,
for example on a kiosk. In Inference Mode:

- no battery percentage is asked; the chirp is captured into a temporary
  `Captures/Inference/` session that is deleted as soon as the estimate is done
  (nothing from an inference run stays on disk);
- the "Inference setup" tab has a model picker (checkpoints under
  `CoBas/Inference/models`, or `COBAS_MODELS_DIR`) and "Chirps per estimate"
  (default 1). The equal-weight three-modal model is pre-selected and loads in
  the background;
- "Start Estimate" runs the normal chirp capture; as each chirp's thermal,
  mmWave and voice files land they are classified and the footer shows only
  "Battery Level: N%" (no confidence or breakdown on screen). The setup tab
  holds the model, the chirps per estimate and the thermal display controls.
  Several chirps are combined by averaging log-probabilities;
- results are shown on screen only; no prediction file is written.

The app side lives in `CoBas Battery Reader V1.0 App/InferenceMode.py`; the
capture, pulse and sensor code is untouched.

### Local Image Inference

Inference Mode also has a **Local Image Inference** tab that estimates the
battery level of one picture already on disk. No chirp is played and no sensor
is read.

- "Open Image" opens a touch file browser: folders first, then PNG, JPG and BMP
  files; tap a folder to enter it and an image to open it, and tap
  "Page N / M" for a keypad to jump in a long folder. It starts in the app's
  `Captures` folder and reopens where the last image came from.
- The picture is classified in the background and the result appears under it
  as "Battery Level: N%". Nothing is written to disk.
- The tab has its own model button (the thermal standalone model by default),
  independent of the model used for live estimates. Changing the model
  re-runs the open image.
- The live-capture footer is hidden on this tab so the picture has the room;
  it stays while a live estimate is running, and an image cannot be opened
  until that estimate has finished.

The opened picture is taken as the selected model's own input image:

| Model | Open |
|---|---|
| Thermal standalone | a thermal camera image (`Thermal Frames/*.png`) |
| mmWave standalone | a mmWave range-angle frame (`mmWave Frames/*.jpg`) |
| Voice standalone | a magma spectrogram image, as its notebook saved it |
| Fusion models | the already fused image (256×256; another size is resized to it) |

The picture is not checked against the model: a thermal image given to the
mmWave model still produces a number, so the tab names what the selected model
expects. The one refusal is an image far larger than a radar frame for the
mmWave standalone, which runs at the picture's own size.

The screen lives in `CoBas Battery Reader V1.0 App/LocalImageInference.py`; the
recipe is `preprocess.image_model_input`, also available as
`Predictor.predict_image(path)`.

## Screenshots

Captured on the Pi's 800×480 touchscreen (`CoBas/Screenshots/`):

| | |
|---|---|
| `1.png` startup mode chooser | `2.png` Data Collection: capture setup |
| `3.png` Data Collection: live sensors (idle) | `4.png` Data Collection: system tab |
| `5.png` Inference setup with the model loaded | `6.png` model picker (fusion ratios and standalone models) |
| `7.png` live sensors during the chirp | `8.png` result after a real chirp: "Battery Level: N%" |
| `9.png` Local Image Inference: result for an opened thermal image | `10.png` Local Image Inference: image browser |

## Model families and their recipes

| Checkpoint | Inputs | Fusion | Tensor |
|---|---|---|---|
| `CoBas_mmwave_thermal_voice_fusion_model.pth` | mmWave, thermal, voice | per-image min-max, weighted mean | ImageNet normalization |
| `CoBas_mmwave_thermal_fusion_model.pth` | mmWave, thermal | /255, weighted mean | 0–1 only |
| `CoBas_mmwave_and_voice_fusion_model.pth` | mmWave, voice | raw 0–255 mean | 0–1 only |
| `CoBas_thermal_and_voice_model.pth` | thermal, voice | raw 0–255 mean (thermal via a 224 crop) | 0–1 only |
| `Standalone/CoBas_thermal_model.pth` | thermal | none (resize 256, crop 224) | 0–1 only |
| `Standalone/CoBas_mmWave_model.pth` | mmWave | none (the raw 724×256 frame, no resize) | 0–1 only |
| `Standalone/CoBas_voice_reflection_recognition_model.pth` | voice | none (magma-coloured spectrogram, resized to 224×224) | 0–1 only |

The three standalone checkpoints on Drive sit under "CoBas Portable 11
Batteries Outside with Random Positions Model/Standalone", but their training
run metadata shows they were trained on the same "4 Positions" dataset as the
fusion models (the notebooks' output path was updated after that run).

`cobas_inference/specs.py` is the single source of truth for these recipes;
`tests/test_parity.py` proves them against images the training notebooks saved
on Drive: every resize, the floor-quantized spectrogram and all fusion
arithmetic match byte for byte, and the round-to-nearest spectrograms match to
within one grey level on 8 of 65,536 pixels (a last-bit floating-point
difference between the Colab and Raspberry Pi FFT stacks; see
`tests/fixtures/README.md`).

## Tests

```bash
.venv/bin/python -m pytest "CoBas/Inference/tests" -q
```

## Phase 1 results (2026-11-07, Raspberry Pi 5)

Since 2026-11-09 the device holds only the latest dataset, Oct 21 "10 Battery
4 Positions Outside Dataset", and all testing uses it. The Oct 16 "ELB
Outside" dataset was removed from the device that day, so the Oct 16 figures
below are a record of the 2026-11-07 runs and cannot be re-run here.

**Parity.** Regenerating the training images from the raw chirp files matches
the notebook output byte for byte for every resize, the floor-quantized
spectrogram and all fusion arithmetic; the round-to-nearest spectrograms
differ by one grey level on 8 of 65,536 pixels (see `tests/fixtures/README.md`).
The device therefore feeds the network what it was trained on.

**Speed.** ResNet-18 forward pass, 4 threads: PyTorch ≈ 175 ms median, ONNX
Runtime ≈ 190 ms. Whole chirp (spectrogram + resizes + fusion + network) ≈
250–450 ms, well inside the 2 s chirp period. ONNX export works
(`export-onnx`) and agrees with PyTorch to 1e-11, but gives no speedup here.

**Accuracy by dataset** (three-modal 1-1-1 model, 25 chirps per session):

| Dataset | Relationship to training | Per-chirp accuracy | Session votes |
|---|---|---|---|
| Oct 21 "4 Positions" | the sessions the model was trained on | 100 % (275/275, every 8th chirp so all four positions are covered) | all correct, confidence ≈ 1.0 |
| Oct 16 "ELB Outside" | different day, never seen | **8.4 % (23/275), chance is 9.1 %** | every session votes "100 %" |

The two-modal 1-1 models behave the same way on the unseen Oct 16 sessions
(25 chirps per session, 275 chirps):

| Model | Per-chirp accuracy | Session votes |
|---|---|---|
| mmWave-Thermal | 9.8 % (27/275) | every session votes "60 %" |
| mmWave-Voice | 9.1 % (25/275) | every session votes "10 %" |
| Thermal-Voice | 9.1 % (25/275) | every session votes "60 %" at confidence ≈ 0.99 |

On the unseen day the models are confidently wrong (three-modal mean
confidence 0.69 when wrong; mmWave-Voice and Thermal-Voice above 0.9), so
confidence is not a usable reliability signal either.

**Full check (2026-11-07, every 8th chirp, 12 per session, both datasets).**
All 16 checkpoints load, run on the fixture chirp and predict its true label
(100 %); per-chirp latency including preprocessing is 106–322 ms on the Pi 5
with four threads, and the ONNX export agrees with PyTorch to 1e-11. A real
chirp on the device through the three-modal model returned "Battery Level:
50%" (`Screenshots/8.png`). Accuracy per model:

| Model | Oct 21 sessions (trained on) | Oct 16 sessions (unseen) | Unseen-day votes |
|---|---|---|---|
| Three-modal 1-1-1 | 100 % (132/132) | 9.1 % | all "100 %" |
| mmWave-Thermal 1-1 | 100 % (132/132) | 9.8 % | all "60 %" |
| mmWave-Voice 1-1 | 99.2 % (131/132) | 9.1 % | all "10 %" |
| Thermal-Voice 1-1 | 96.2 % (127/132) | 9.1 % | all "60 %" |
| Standalone thermal | 99.2 % (131/132) | 9.8 % | all "50 %" |
| Standalone mmWave | 98.5 % (130/132) | 5.3 % | "60 %" or "10 %" |
| Standalone voice | 97.7 % (129/132) | 8.3 % | mostly "40 %" |

Chance is 9.1 %. The integration is correct (every model reproduces its
training-day behaviour through the device pipeline); the models themselves do
not transfer to a different recording day, and the standalone models are no
exception. The clearest
measurable shift between the two days is the thermal channel: mean thermal
frame colour is (0, 134, 233) on Oct 16 versus (2, 213, 208) on Oct 21 while
the logged temperatures differ by only 1–2 °C, which is what the absolute
temperature-to-colour mapping of the thermal renderer predicts. mmWave
brightness and audio level are similar between the days.

Conclusion for deployment: the current checkpoints recognise the recording
session, not the battery level, and must be retrained with several sessions
per class (and validated leave-one-session-out) before Inference Mode can show
a trustworthy number. The pipeline itself is ready; `replay` is the harness for
that validation.

## Local Image Inference check (2026-11-09, latest dataset only)

Eleven pictures per model: the first chirp of each battery level's session in
the Oct 21 "4 Positions" dataset, opened through the Local Image Inference
recipe with `replay --as-images --stride 200 --limit 1`.

| Model | Picture opened | Correct |
|---|---|---|
| Standalone thermal | thermal frame | 11/11 |
| Standalone mmWave | mmWave frame | 10/11 (40 % read as 30 %) |
| Standalone voice | spectrogram PNG | 11/11 |
| Three-modal 1-1-1 | fused PNG | 11/11 |
| mmWave-Thermal 1-1 | fused PNG | 11/11 |
| mmWave-Voice 1-1 | fused PNG | 10/11 (40 % read as 30 %) |
| Thermal-Voice 1-1 | fused PNG | 10/11 (90 % read as 80 %) |

On all 77 chirps the picture gives the same answer as the raw chirp through
`replay`, so the three misses are the models' own, not the picture path.

The screen itself was also driven on the Pi: for each battery level one
thermal frame and one mmWave frame (a different chirp per level) were opened
by tapping through the image browser, and all 22 read-outs matched the
recorded level.

These are sessions the models were trained on, so this checks that the screen
and its recipe work; it says nothing about a new recording day.
