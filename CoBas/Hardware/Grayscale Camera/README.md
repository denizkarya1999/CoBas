# Grayscale Thermal Camera

This variant displays the MLX90642's coldest current reading as black and its
hottest as white, using the full greyscale range between them. Native 32x24
sensor pixels are enlarged as crisp square blocks in both the live view and
recorded video. Video is recorded at 1280x960 by default, making each sensor
pixel a 40x40 block.

The live GUI includes an estimated Celsius spectrum beside the image. White at
the top represents the hottest value in the current frame, black at the bottom
represents the coldest value, and the intermediate labels show the predicted
temperature bands. The range updates automatically for every camera frame.

Run without camera hardware:

```bash
python3 "Grayscale Camera/grayscale_camera_app.py" --mock
```

Run with the MLX90642 connected:

```bash
python3 "Grayscale Camera/grayscale_camera_app.py"
```

Use **Record** / **Stop Recording** in the bottom toolbar. MP4 files are saved
to `Grayscale Camera/recordings`. Recording requires `ffmpeg`, which is
installed by the parent project's Raspberry Pi setup script.

The output size can be changed with `MLX90642_RECORD_WIDTH` and
`MLX90642_RECORD_HEIGHT`.
