# Parity fixtures

`pair0_100pct/` is chirp 1 of position 1 from the 100 % battery session recorded
with the CoBas Portable on 2026-10-21 (`Session_20261021_183911_h0fwcc0u`).

- `raw/` — the files the Battery Reader app wrote for that chirp: the mmWave
  range-angle frame (`frame_000001.jpg`), the thermal frame and the 2 s voice
  reflection recording.
- `expected/<family>/` — the images the Colab training notebooks saved for the
  same pair (`pair_index` 0 of class `100_ Percent` on Drive, dataset
  "CoBas 11 Batteries with 4 Positions Recorded with CoBas Portable"):
  - `three_modal_1-1-1/` — `mmwave.png`, `thermal.png` (256×256 resizes),
    `voice.png` (256×256 grayscale spectrogram, low frequency at the bottom)
    and `fused.png` from "1 - 1 - 1 Fusions".
  - `thermal_voice_1-1/` — `thermal224.png` (the notebook's Resize(256)+
    CenterCrop(224) intermediate), `voice.png` (spectrogram before the flip)
    and `fused.png` from "1 - 1 Fusions / CoBas Thermal-Voice Fusion Dataset".
  - `mmwave_voice_1-1/` — `mmwave.png`, `voice.png` and `fused.png` from
    "1 - 1 Fusions / mmWave - Voice Fusion Dataset".
  - `voice_standalone/spectrogram.png` — the Voice standalone notebook's
    magma-coloured 224×188 spectrogram of the same recording, from
    "CoBas Spectrogram Dataset / 100_ Percent".

`tests/test_parity.py` regenerates each of these from `raw/` and requires a
byte-for-byte match, with one documented exception: the round-to-nearest
spectrograms (three-modal and mmWave-Voice) differ from the Colab output in
8 of 65,536 pixels by exactly one grey level. No arithmetic variant removes
the difference and none of the values is near a rounding tie, so it is a
last-bit floating-point difference between the Colab (x86) and Raspberry Pi
(aarch64) FFT/mel stacks; those tests allow one grey level on at most 0.05 %
of pixels. The Thermal-Voice spectrogram (floor quantization), every resize and
all fusion arithmetic match exactly. The mmWave-Thermal family kept its fused
images in a NumPy cache rather than PNGs, so it is checked against the recipe
string stored in its checkpoint instead.
