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
other families carry their weights inside the checkpoint. Every model is a
PyTorch checkpoint: `.pth`, or `.pt` under the same name. No other format is
loaded.

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

`replay` walks `<N>_Percent_Battery/Session_*/` folders, pairs the k-th thermal
frame and voice recording of each position with the matching global mmWave
frame (exactly as the training notebooks paired them), prints a per-session
vote and per-chirp accuracy when the folder name carries the label, and can
write every prediction to CSV.

```bash
.venv/bin/python -m cobas_inference replay --as-files --model "models/Standalone/CoBas_thermal_model.pth" "/home/tailab/Datasets/CoBas/10 Battery 4 Positions Outside Dataset"
```

`--as-files` runs the same walk through the Local File Inference recipe: one
file per chirp, opened as that screen opens it. A standalone model opens the
saved file itself (the thermal frame, the mmWave frame or the voice
recording); a session holds no ready-made picture for a fusion model, so its
fused image is written to a PNG first and that file is opened.

```bash
.venv/bin/python -m cobas_inference build-latest-dataset "/home/tailab/Datasets/CoBas/10 Battery 4 Positions Outside Dataset" "/home/tailab/Datasets/CoBas/CoBas Latest Dataset"
```

```bash
.venv/bin/python -m cobas_inference check-latest-dataset "/home/tailab/Datasets/CoBas/CoBas Latest Dataset"
```

`build-latest-dataset` writes a small dataset of ready-to-open files: for
every checkpoint under `models/`, ten files per battery level (110 per
model), laid out as `<models sub-folder>/<model name>/<N>_Percent_Battery/`,
for example `Standalone/CoBas Thermal Model/60_Percent_Battery/`. Each file
is what that model takes in Local File Inference: an unchanged copy of the
saved thermal frame, mmWave frame or voice recording for the standalone
models, and the image fused with that checkpoint's own weights for a fusion
model. Every model gets the same chirps, spread evenly over each level's
recordings, and `manifest.csv` lists the source files. `check-latest-dataset`
classifies every file with its own model and reports the misses.

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

### Local File Inference

Inference Mode also has a **Local File Inference** tab that estimates the
battery level of one file already on disk: a picture or a voice recording. No
chirp is played and no sensor is read.

- "Open File" opens a touch file browser: folders first, then pictures (PNG,
  JPG, BMP) and voice recordings (WAV, FLAC); tap a folder to enter it and a
  file to open it, and tap "Page N / M" for a keypad to jump in a long folder.
  It starts in `~/Datasets` (or `COBAS_DATASETS_DIR`) and reopens where the
  last file came from.
- The file is classified in the background and the result appears under the
  picture as "Battery Level: N%". Nothing is written to disk.
- A voice recording is converted to its spectrogram first, by the voice
  standalone's own recipe, and the spectrogram is shown where a picture would
  be. Only the voice standalone can read a recording; with any other model the
  tab says so and gives no estimate.
- The tab has its own model button (the thermal standalone model by default),
  independent of the model used for live estimates. Changing the model
  re-runs the open file.
- The live-capture footer is hidden on this tab so the picture has the room;
  it stays while a live estimate is running, and a file cannot be opened
  until that estimate has finished.

What to open for each model:

| Model | Open |
|---|---|
| Thermal standalone | a thermal camera image (`Thermal Frames/*.png`) |
| mmWave standalone | a mmWave range-angle frame (`mmWave Frames/*.jpg`) |
| Voice standalone | a voice recording (`Voices/*.wav`), or a magma spectrogram image as its notebook saved it |
| Fusion models | the already fused image (256×256; another size is resized to it) |

A picture is not checked against the model: a thermal image given to the
mmWave model still produces a number, so the tab names what the selected model
expects. Two things are refused: a voice recording with a model other than the
voice standalone, and a picture far larger than a radar frame for the mmWave
standalone, which runs at the picture's own size. Only lossless recordings are
listed; a lossy copy (OGG, MP3) loses the 17.5–20 kHz band the model reads.

The screen lives in `CoBas Battery Reader V1.0 App/LocalFileInference.py`; the
recipe is `preprocess.file_model_input`, also available as
`Predictor.predict_file(path)`.

### Reliability fixes (2026-11-09)

A review of the whole app found these defects. The first four, the frozen
loop and the swallowed thermal error were reproduced against the old code and
checked against the new code, using the app's real capture, abort and export
routines with stand-in sensors. The thermal start retry was checked with a
stand-in sensor and the real one, the settings carry-over and the folder
clean-up on the real app, and the model-load message by reading only. Both
modes were then run on the real sensors: live estimates, and a Data Collection
capture of 4 chirps over 2 positions that saved 4 recordings, 4 thermal frames
and 4 radar frames.

- **Thermal files of a finished position were deleted.** Stopping or aborting
  before the next position started recording removed the previous position's
  thermal video and logs, and the export then failed. The same happened to
  files kept after a failed export. Cleanup now touches only the position that
  made the recording.
- **A stopped chirp left a partial voice recording in the saved dataset.**
  Recordings (and radar frames) that belong to no saved chirp are now removed
  when the capture is saved.
- **A sensor failure in a later position blocked saving the finished ones.**
  The failed position's chirps were still counted, so the export stopped with
  "Expected N mmWave frames". They are now dropped and the finished positions
  are saved.
- **A failed estimate looked like a successful one.** The reason was replaced
  by "Ready" and the read-out stayed at "…". The reason now stays on screen
  and the read-out returns to "—".
- **A model that failed to load said "still loading" for ever.** It now says
  the model could not be loaded.
- **One error could freeze the app.** An exception in any handler stopped the
  loop that delivers every sensor and worker result, leaving "Saving…" stuck
  with Close refused. Both polling loops now keep running.
- **The thermal sensor sometimes failed to start.** The driver allows the
  first frame only 200 ms beyond the frame period; when it took longer the
  capture stopped with "Thermal recording could not start" (seen on 3 of 7
  starts in one test series). The start is now retried up to three times, and
  if it still fails the message gives the camera's own reason. A thermal
  error raised while a recording starts also stops the capture now instead of
  leaving a frozen frame.
- **Relaunching for the next battery reset every setting.** Chirp count,
  positions, thermal range and colour mode, microphone and speaker now carry
  over to the relaunch.
- **Empty folders were left behind.** An attempt that saved nothing no longer
  leaves an empty `Session_…` folder, and Inference Mode no longer leaves an
  empty `Captures` folder. The launcher also stops Python writing
  `__pycache__` folders.

Known and left as it is: a radar whose serial ports are present but whose
firmware does not answer takes about 70 seconds to report the failure, and
thermal Minimum/Maximum take effect only after "Apply".

## Screenshots

Captured on the Pi's 800×480 touchscreen (`CoBas/Screenshots/`):

| | |
|---|---|
| `1.png` startup mode chooser | `2.png` Data Collection: capture setup |
| `3.png` Data Collection: live sensors (idle) | `4.png` Data Collection: system tab |
| `5.png` Inference setup with the model loaded | `6.png` model picker for live estimates (all models are `.pth` checkpoints) |
| `7.png` live sensors during a real chirp | `8.png` result after that chirp: "Battery Level: N%" |
| `9.png` Local File Inference: a 60 % thermal image read by the thermal model | `10.png` file browser: thermal images of one battery level |
| `11.png` Local File Inference: nothing opened yet | `12.png` Local File Inference: model picker, standalone models first |
| `13.png` file browser: the CoBas Latest Dataset's sub-folders | `14.png` file browser: the standalone model folders |
| `15.png` file browser: voice recordings of one battery level | `16.png` a 40 % mmWave frame read by the mmWave model |
| `17.png` a 30 % voice recording converted to its spectrogram and read by the voice model | `18.png` a 60 % fused image read by the three-modal 1-1-1 model |
| `19.png` a photo the mmWave model refuses as too large | `20.png` a voice recording with a model that cannot read it |
| `21.png` a voice recording while it is being converted | |

Screenshots 5 to 21 were retaken on 2026-11-09 with the final code; 7 and 8
are a real chirp through the radar, thermal camera, speaker and microphone,
and 9 to 18 and 20 to 21 use files from the CoBas Latest Dataset. Screenshots
5 to 8 were taken again after the models became `.pth` only.

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

`cobas_inference/specs.py` is the single source of truth for these recipes.
They were proved against images the training notebooks saved on Drive: every
resize, the floor-quantized spectrogram and all fusion arithmetic match byte
for byte, and the round-to-nearest spectrograms match to within one grey level
on 8 of 65,536 pixels (a last-bit floating-point difference between the Colab
and Raspberry Pi FFT stacks).

The automated tests and their sample chirp were removed from the repository on
2026-11-09 (they last passed, 53 of 53 for the recipes and the engine, against
this code that day). They remain in the git history up to commit `eafcb25`
under `CoBas/Inference/tests`.

## Inference report

[INFERENCE_REPORT.md](INFERENCE_REPORT.md) is the latest full test of the app's
inference (2026-11-09): 1,731 of the 1,760 files of the CoBas Latest Dataset
read correctly (98.4 %) across the 16 `.pth` models, with accuracy by model,
model type and battery level, one file per model opened on the
real screen (16 of 16), and a live estimate on the real sensors.

## Phase 1 results (2026-11-07, Raspberry Pi 5)

The ONNX figures below are a record of the 2026-11-07 runs. ONNX loading and
export were removed on 2026-11-09, when every model became a PyTorch
checkpoint; the one ONNX copy in use gave the same answers as its `.pth`
(largest probability difference 5.75e-08 over 110 files).

Since 2026-11-09 the device holds only the latest dataset, Oct 21 "10 Battery
4 Positions Outside Dataset", and all testing uses it. The Oct 16 "ELB
Outside" dataset was removed from the device that day, so the Oct 16 figures
below are a record of the 2026-11-07 runs and cannot be re-run here.

**Parity.** Regenerating the training images from the raw chirp files matches
the notebook output byte for byte for every resize, the floor-quantized
spectrogram and all fusion arithmetic; the round-to-nearest spectrograms
differ by one grey level on 8 of 65,536 pixels.
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

## Local File Inference check (2026-11-09, latest dataset only)

Eleven files per model: the first chirp of each battery level's session in
the Oct 21 "4 Positions" dataset, opened through the Local File Inference
recipe with `replay --as-files --stride 200 --limit 1`.

| Model | File opened | Correct |
|---|---|---|
| Standalone thermal | thermal frame | 11/11 |
| Standalone mmWave | mmWave frame | 10/11 (40 % read as 30 %) |
| Standalone voice | voice recording, converted to its spectrogram | 11/11 |
| Three-modal 1-1-1 | fused PNG | 11/11 |
| mmWave-Thermal 1-1 | fused PNG | 11/11 |
| mmWave-Voice 1-1 | fused PNG | 10/11 (40 % read as 30 %) |
| Thermal-Voice 1-1 | fused PNG | 10/11 (90 % read as 80 %) |

On all 77 chirps the file gives the same answer as the raw chirp through
`replay`, so the three misses are the models' own, not the file path.

The screen itself was also driven on the Pi: for each battery level one
thermal frame and one mmWave frame (a different chirp per level) were opened
by tapping through the file browser, and all 22 read-outs matched the
recorded level.

These are sessions the models were trained on, so this checks that the screen
and its recipe work; it says nothing about a new recording day.

### CoBas Latest Dataset (2026-11-09)

`/home/tailab/Datasets/CoBas/CoBas Latest Dataset` was built from the same Oct
21 dataset with `build-latest-dataset`: 16 checkpoints × 11 battery levels × 10
files = 110 files per model, 1,760 in total (140 MB). Each level's ten chirps
are numbers 11, 31, … 191 of its session, so all four positions are covered.
The standalone models' folders hold unchanged copies of the saved thermal
frames, mmWave frames and voice recordings (the voice model's 110 files are
the WAV recordings, which the app converts to spectrograms); the fusion
models' folders hold fused images. `check-latest-dataset` reads 1,731 of
1,760 correctly:

| Models sub-folder | Model folder | Correct of 110 |
|---|---|---|
| Standalone | CoBas Thermal Model | 107 |
| Standalone | CoBas mmWave Model | 110 |
| Standalone | CoBas Voice Reflection Recognition Model | 110 |
| 1 - 1 - 1, 1 - 1 - 2, 1 - 2 - 1 and 2 - 1 - 1 Fusions | CoBas mmWave Thermal Voice Fusion Model | 110 each |
| 1 - 1, 1 - 2 and 2 - 1 Fusions | CoBas mmWave Thermal Fusion Model | 110 each |
| 1 - 1 Fusions | CoBas mmWave and Voice Fusion Model | 109 |
| 1 - 2 Fusions | CoBas mmWave and Voice Fusion Model | 110 |
| 2 - 1 Fusions | CoBas mmWave and Voice Fusion Model | 107 |
| 1 - 1 Fusions | CoBas Thermal and Voice Model | 106 |
| 1 - 2 Fusions | CoBas Thermal and Voice Model | 102 |
| 2 - 1 Fusions | CoBas Thermal and Voice Model | 100 |

The misses are the models' own: the Thermal and Voice ratio folders give the
same 102 and 100 from the raw chirps, and reading their folder ratio the other
way round drops them to chance (10 and 12 of 110). The misses and the date are
kept beside the files in `check_results_2026-11-09.txt`. The three standalone
models were checked on the final build; the fusion models' images are
byte-identical to the build that was checked earlier the same day.

On the Pi, a thermal image, a mmWave frame, a voice recording and a fused
image from this dataset were each opened through the Local File Inference
screen with their own model, and all four were read correctly.

The 1 - 2 and 2 - 1 mmWave Thermal checkpoints make the loader warn that their
stored description (mmWave min-max, ImageNet normalization) differs from the
registry recipe. The registry recipe is the one that fits them: it reads
110/110 on these chirps, while the stored description reads 10 and 13 of 110.
