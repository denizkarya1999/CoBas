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

## Model families and their recipes

| Checkpoint | Inputs | Fusion | Tensor |
|---|---|---|---|
| `CoBas_mmwave_thermal_voice_fusion_model.pth` | mmWave, thermal, voice | per-image min-max, weighted mean | ImageNet normalization |
| `CoBas_mmwave_thermal_fusion_model.pth` | mmWave, thermal | /255, weighted mean | 0–1 only |
| `CoBas_mmwave_and_voice_fusion_model.pth` | mmWave, voice | raw 0–255 mean | 0–1 only |
| `CoBas_thermal_and_voice_model.pth` | thermal, voice | raw 0–255 mean (thermal via a 224 crop) | 0–1 only |

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
confidence is not a usable reliability signal either. The clearest
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
