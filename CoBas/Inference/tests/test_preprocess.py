import numpy as np
import pytest
from PIL import Image

from cobas_inference import preprocess, specs


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _save_rgb(path, array):
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def test_modality_image_bilinear_matches_pil(tmp_path, rng):
    source = rng.integers(0, 256, size=(256, 724, 3), dtype=np.uint8)  # mmWave frame shape
    path = _save_rgb(tmp_path / "frame.png", source)
    expected = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.BILINEAR))
    got = preprocess.modality_image_256(path, specs.ImageSpec("bilinear"))
    assert got.shape == (256, 256, 3)
    np.testing.assert_array_equal(got, expected)


def test_thermal_voice_pre_crop_round_trip(tmp_path, rng):
    source = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    path = _save_rgb(tmp_path / "thermal.png", source)
    image = Image.fromarray(source).resize((256, 256), Image.Resampling.BILINEAR)
    image = image.crop((16, 16, 240, 240))
    expected = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS))
    got = preprocess.modality_image_256(path, specs.ImageSpec("bilinear", pre_crop_224=True))
    np.testing.assert_array_equal(got, expected)


def test_center_crop_matches_torchvision_rounding():
    image = Image.new("RGB", (256, 256))
    cropped = preprocess.center_crop(image, 224)
    assert cropped.size == (224, 224)


def test_spectrogram_quantization_rules():
    db = np.linspace(-80.0, 0.0, 224 * 10, dtype=np.float32).reshape(224, 10)
    rint = preprocess.spectrogram_to_uint8(db, specs.VoiceSpec(quantize="rint"))
    floor = preprocess.spectrogram_to_uint8(db, specs.VoiceSpec(quantize="floor"))
    assert rint.dtype == np.uint8 and floor.dtype == np.uint8
    assert rint.max() == 255 and rint.min() == 0
    assert floor.max() == 255 and floor.min() == 0
    # floor never exceeds round-to-nearest
    assert np.all(floor.astype(int) <= rint.astype(int))


def test_spectrogram_constant_input_is_zero():
    db = np.zeros((224, 50), dtype=np.float32)
    assert preprocess.spectrogram_to_uint8(db, specs.VoiceSpec()).sum() == 0


def test_fuse_equal_weights_three_modal_minmax():
    a = np.full((256, 256, 3), 10, dtype=np.uint8)
    a[0, 0] = 255  # give it a range
    b = np.full((256, 256, 3), 200, dtype=np.uint8)
    b[0, 0] = 0
    c = np.zeros((256, 256, 3), dtype=np.uint8)
    c[0, 0] = 255
    fused = preprocess.fuse([a, b, c], (1 / 3, 1 / 3, 1 / 3), specs.PER_IMAGE_MINMAX)
    # pixel (1,1): a -> (10-10)/(255-10) = 0, b -> (200-0)/(200-0) = 1, c -> 0; mean then x255
    assert fused[1, 1, 0] == np.rint((0.0 + 1.0 + 0.0) / 3.0 * 255.0)
    # pixel (0,0): a -> 1, b -> 0, c -> 1
    assert fused[0, 0, 0] == np.rint((1.0 + 0.0 + 1.0) / 3.0 * 255.0)
    assert fused.dtype == np.uint8


def test_fuse_div255_and_raw255_agree_for_half_weights(rng):
    a = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    div = preprocess.fuse([a, b], (0.5, 0.5), specs.DIV255)
    raw = preprocess.fuse([a, b], (0.5, 0.5), specs.RAW255)
    # Both are the integer-rounded mean; allow the odd 1-LSB float difference.
    assert np.abs(div.astype(int) - raw.astype(int)).max() <= 1
    assert np.array_equal(raw, np.rint((a.astype(np.float32) + b) / 2.0).astype(np.uint8))


def test_fuse_weighted_order_matches_notebook_expression(rng):
    a, b, c = (rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(3))
    fused = preprocess.fuse([a, b, c], (0.25, 0.25, 0.5), specs.PER_IMAGE_MINMAX)
    na, nb, nc = (preprocess.minmax_unit(x.astype(np.float32)) for x in (a, b, c))
    expected = np.rint(np.clip(na * np.float32(0.25) + nb * np.float32(0.25) + nc * np.float32(0.5), 0, 1) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(fused, expected)


def test_to_model_input_shapes_and_normalization(rng):
    fused = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    unit = preprocess.to_model_input(fused, specs.UNIT)
    imagenet = preprocess.to_model_input(fused, specs.IMAGENET)
    assert unit.shape == imagenet.shape == (1, 3, 224, 224)
    assert unit.dtype == np.float32
    assert 0.0 <= unit.min() and unit.max() <= 1.0
    # center crop offset 16, channel-first
    assert unit[0, 2, 0, 0] == pytest.approx(fused[16, 16, 2] / 255.0)
    expected = (fused[16, 16, 0] / 255.0 - 0.485) / 0.229
    assert imagenet[0, 0, 0, 0] == pytest.approx(expected, abs=1e-5)


def test_native_pipeline_keeps_frame_size(tmp_path, rng):
    source = rng.integers(0, 256, size=(256, 724, 3), dtype=np.uint8)
    path = _save_rgb(tmp_path / "frame.jpg", source)
    native = preprocess.modality_image_256(path, specs.ImageSpec("none"))
    assert native.shape == (256, 724, 3)
    tensor = preprocess.to_model_input(native, specs.UNIT, specs.NATIVE)
    assert tensor.shape == (1, 3, 256, 724)
    assert tensor[0, 1, 5, 7] == pytest.approx(native[5, 7, 1] / 255.0)


def test_resize224_pipeline_matches_torchvision_resize(rng):
    image = rng.integers(0, 256, size=(224, 188, 3), dtype=np.uint8)  # native spectrogram shape
    tensor = preprocess.to_model_input(image, specs.UNIT, specs.RESIZE224)
    expected = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    assert tensor.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(tensor[0].transpose(1, 2, 0), expected, atol=1e-7)


def test_spectrogram_to_rgb_uses_magma_lut():
    matplotlib = pytest.importorskip("matplotlib")
    db = np.linspace(-80.0, 0.0, 224 * 10, dtype=np.float32).reshape(224, 10)
    rgb = preprocess.spectrogram_to_rgb(db, "magma")
    assert rgb.shape == (224, 10, 3) and rgb.dtype == np.uint8
    lut = matplotlib.colormaps["magma"]
    np.testing.assert_array_equal(rgb[0, 0], (np.asarray(lut(0.0)[:3]) * 255).astype(np.uint8))
    np.testing.assert_array_equal(rgb[-1, -1], (np.asarray(lut(1.0)[:3]) * 255).astype(np.uint8))


def test_gray_to_rgb():
    gray = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
    rgb = preprocess.gray_to_rgb(gray)
    assert rgb.shape == (256, 256, 3)
    assert np.array_equal(rgb[..., 0], gray) and np.array_equal(rgb[..., 2], gray)
