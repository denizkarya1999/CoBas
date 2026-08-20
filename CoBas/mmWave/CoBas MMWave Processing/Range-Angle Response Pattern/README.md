# Live Range-Angle Response Pattern

This app builds a live range-azimuth heatmap on top of the sibling
`Raw IQ Signals` USB1 reader. The radar is configured automatically through
USB0, while the zero-Doppler complex range-FFT matrix is decoded from USB1.

## Structure

```text
Range-Angle Response Pattern/
├── Logic/
│   ├── config.py
│   ├── frame_range_angle_transformer.py
│   ├── raw_iq_source.py
│   ├── range_angle_processor.py
│   ├── recording_clock.py
│   ├── reference_frame_generator.py
│   ├── session_logger.py
│   ├── video_frame_recorder.py
│   └── live_service.py
├── Interface/
│   └── range_angle_interface.py
├── Frames/
├── References/
├── Logs/
│   ├── Raw IQ Signals/
│   └── Range-Angle Responses/
├── Theory Behind MMWave.txt
├── generate_references.py
├── start_range_angle_app.py
├── transform_existing_frames.py
└── requirements.txt
```

- `Logic/raw_iq_source.py` reuses the existing Raw IQ UART and TLV parser.
- `Logic/frame_range_angle_transformer.py` converts legacy full-window images
  to the selected range-angle window exactly once.
- `Logic/range_angle_processor.py` constructs the complex matrix and performs
  calibrated azimuth beamforming.
- `Logic/recording_clock.py` excludes battery-repositioning pauses from the
  requested recording duration.
- `Logic/reference_frame_generator.py` selects one random frame and creates one
  labeled reference image for each battery level.
- `Logic/session_logger.py` writes paired input and display-response CSV files.
- `Logic/video_frame_recorder.py` records temporary spectrogram video and
  extracts it into clean image frames.
- `Logic/live_service.py` keeps serial acquisition off the GUI thread.
- `Interface/range_angle_interface.py` contains the Tkinter/Matplotlib UI.
- `start_range_angle_app.py` is the application launcher.
- `generate_references.py` creates any missing references from saved frames.
- `transform_existing_frames.py` crops legacy frames and rebuilds references.
- `Theory Behind MMWave.txt` explains the complete signal path and equations.

## Install and run

From `CoBas MMWave Processing/Range-Angle Response Pattern`:

```bash
python3 -m pip install -r requirements.txt
python3 start_range_angle_app.py
```

Tkinter is supplied by many Raspberry Pi Python installations. If it is not
available:

```bash
sudo apt install python3-tk
```

The app asks for a battery level (for example `20%`), a spectrogram video
duration in seconds, and the number of battery positions to record. That
battery level becomes the shared filename for both logs and the name of the
extracted-frame folder. The duration is divided equally between the requested
positions. Between intervals, recording and the countdown pause while a dialog
asks the user to move the battery and click OK. For example, a 60-second session
with four positions pauses after 15, 30, and 45 recorded seconds. Repositioning
time is not included in the requested duration.

The Raw IQ log, Range-Angle log, and clean spectrogram frames remain one
continuous session with the same filenames, folder, frame format, frame rate,
and sequential frame naming. All three stop automatically when the countdown
reaches zero; the Stop button can end them early. Close other programs using
`/dev/ttyUSB0` or `/dev/ttyUSB1` before starting it because serial ports allow
only one owner at a time.

## Session logs

Each battery level creates two matching CSV files without overwriting an
earlier capture. For a `20%` entry, the files are:

```text
Logs/Raw IQ Signals/20_Percent_Battery.csv
Logs/Range-Angle Responses/20_Percent_Battery.csv
```

The Raw IQ CSV stores only `frame_number`, `range_bin`, `virtual_antenna`, `i`,
and `q` for range bins 1 through 7. These are the bins whose centers fall in
the requested 0.05-0.50 m window.

The Range-Angle Responses CSV stores `frame_number`, `range_meters`, and one
relative-power dB column for each displayed angle from -60° to +60°. The rows
for a frame are the exact 7-by-121 data matrix given to the live figure before
Matplotlib performs visual interpolation.

## Spectrogram video frames

The live range-angle response is recorded for the requested number of seconds.
The intermediate AVI is temporary: after recording, every video image is
captured at 2 FPS, extracted as a numbered JPEG, and the AVI is deleted. For a
`20%` session:

```text
Frames/20_Percent/frame_000001.jpg
Frames/20_Percent/frame_000002.jpg
...
```

Each JPEG contains only the Viridis-colored spectrogram pixels. It has no
title, X/Y axes, labels, ticks, grid, colorbar, controls, or status text. The
image uses the same -50 dB to 0 dB response range as the live figure. Both CSV
logs and the spectrogram recording use the entered duration and stop together.

After extraction, one saved frame is selected randomly and converted into the
battery's single labeled reference image:

```text
References/20_Percent_Battery_Reference.jpg
```

Unlike the clean training frames, the reference image includes Angle
(degrees), Range (meters), and Energy (dB) references. A fixed output filename
prevents more than one reference from being created for the same battery.

To generate missing references for existing frame folders manually:

```bash
python3 generate_references.py
```

Legacy frames collected with the former 0-4.22 m and -90° to +90° view can be
converted in place, with their references rebuilt, by running:

```bash
python3 transform_existing_frames.py
```

A marker in each battery folder prevents the crop from being applied twice.
New recordings already use the selected window and write the same marker.

## Processing

For every USB1 frame, the app:

1. builds `X[range_bin, virtual_antenna] = I + jQ`;
2. selects the matching 12-, 8-, or 4-channel IWR6843AOP azimuth layout;
3. applies the AOP phase-rotation correction and a Hamming window;
4. evaluates conventional beamforming from -60° to +60°;
5. smooths linear power between frames; and
6. converts power to a relative decibel scale from -50 dB to 0 dB.

The range spacing is derived from the embedded 2 MSPS, 70 MHz/us, 64-bin
profile and is approximately 0.0669 meters per bin. Only bins 1 through 7 are
processed and logged; their centers are approximately 0.067-0.468 m. The plot
is displayed with requested limits of 0.05-0.50 m and -60° to +60°, and both
limits are shown in the status bar and on the axes.

The plotted data is the zero-Doppler range-angle response, so it emphasizes
static returns. The decibel values are relative to the strongest displayed
cell; they are not calibrated dBm measurements.

The current Raw IQ capture contains eight virtual antennas. For that format,
the processor uses all eight channels and their AOP horizontal coordinates,
combining the duplicated azimuth positions across two elevation rows. The GUI
shows the selected/input channel counts beside the peak measurement.

## CoBas V1 integration

The CoBas V1 battery reader imports this processing pipeline through its
`MMWave/capture.py` bridge. CoBas remains the sole owner of chirp duration and
battery-position prompts. The bridge only enables acquisition during an active
position, pauses while the battery is moved, and stops after the final chirp.
It reports the radar as ready only after a valid complex I/Q frame arrives and
aborts the coordinated capture if the USB1 stream stops. The integrated live
preview includes calibrated angle, range, relative-power, peak, frame, and
antenna references; saved training frames remain clean heatmap pixels.

Integrated outputs are written beneath the selected battery capture directory:

```text
mmWave Data/
├── Logs/
│   ├── Raw IQ Signals/<battery>.csv
│   └── Range-Angle Responses/<battery>.csv
├── Frames/frame_000001.jpg
└── References/<battery>_Reference.jpg
```

Frame numbering and both logs remain continuous across positions. No mmWave
data is written during a position-change pause, and the integration does not
create a spectrogram video.
