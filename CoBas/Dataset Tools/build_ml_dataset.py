from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = REPO_ROOT / "Samples"
ML_ROOT = REPO_ROOT / "Machine Learning"
VIDEO_ROOT = REPO_ROOT / "Video Processing"
VOICE_ROOT = REPO_ROOT / "Voice Processing"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VIDEO_SEGMENTATION = load_module(
    "dataset_video_segmentation",
    VIDEO_ROOT / "HelperTools" / "2S_Segmentation.py",
)
VIDEO_AUDIO = load_module(
    "dataset_video_audio",
    VIDEO_ROOT / "HelperTools" / "VideoSoundSeperator.py",
)
VIDEO_FRAMES = load_module(
    "dataset_video_frames",
    VIDEO_ROOT / "HelperTools" / "VideoFrameSlicer.py",
)
VOICE_PIPELINE = load_module(
    "dataset_voice_pipeline",
    VOICE_ROOT / "Voice_Processing_Pipeline.py",
)
VOICE_OLD_CROP = load_module(
    "dataset_voice_old_crop",
    VOICE_ROOT / "Cropping" / "Old_Code_Cropping.py",
)
VOICE_STFT = load_module(
    "dataset_voice_stft",
    VOICE_ROOT / "Spectogram" / "STFT_Spectogram.py",
)
VOICE_PREPARE = load_module(
    "dataset_voice_prepare",
    VOICE_ROOT / "Spectogram" / "Prepare_STFT_Spectograms.py",
)


def sample_number(sample_dir: Path) -> int | None:
    if not sample_dir.name.startswith("Sample_"):
        return None

    number_text = sample_dir.name.removeprefix("Sample_")
    if not number_text.isdigit():
        return None

    return int(number_text)


def next_sample_dir(ml_root: Path) -> Path:
    existing_numbers = [
        number
        for number in (sample_number(path) for path in ml_root.glob("Sample_*"))
        if number is not None
    ]
    next_number = max(existing_numbers, default=0) + 1
    return ml_root / f"Sample_{next_number}"


def find_existing_sample_for_video(ml_root: Path, video_stem: str) -> Path | None:
    expected_output = f"{video_stem}_Image_and_Video"

    for sample_dir in sorted(ml_root.glob("Sample_*")):
        if (sample_dir / expected_output).is_dir():
            return sample_dir
        if any(path.stem == video_stem for path in sample_dir.glob("*")):
            return sample_dir

    return None


def sample_has_complete_output(sample_dir: Path, video_stem: str) -> bool:
    output_dir = sample_dir / f"{video_stem}_Image_and_Video"
    frames_dir = output_dir / "Frames"
    spectrogram_dir = output_dir / f"{video_stem}_Spectogram"
    voices_dir = output_dir / "Voices"

    return (
        frames_dir.is_dir()
        and spectrogram_dir.is_dir()
        and voices_dir.is_dir()
        and any(frames_dir.glob("*.jpg"))
        and any(spectrogram_dir.glob("*.npy"))
        and any(voices_dir.glob("*.wav"))
    )


def list_sample_videos(samples_root: Path) -> list[Path]:
    return sorted(
        path
        for path in samples_root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def clean_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_frames(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Frame folder was not created: {source_dir}")

    clean_folder(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def copy_voice_files(source_dir: Path, destination_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Voice folder was not created: {source_dir}")

    wav_paths = sorted(source_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files were created in: {source_dir}")

    clean_folder(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_paths:
        shutil.copy2(wav_path, destination_dir / wav_path.name)

    return destination_dir / wav_paths[0].name


def copy_prepared_spectrograms(source_dir: Path, destination_dir: Path) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prepared spectrogram folder was not created: {source_dir}")

    npy_paths = sorted(source_dir.glob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(f"No prepared spectrogram arrays were created in: {source_dir}")

    clean_folder(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for npy_path in npy_paths:
        shutil.copy2(npy_path, destination_dir / npy_path.name)

    return len(npy_paths)


def run_old_code_voice_fallback(input_path: Path, final_output_folder: Path) -> Path:
    print("Beacon crop failed; falling back to fixed 2-second voice crops.")

    VOICE_PIPELINE.clean_intermediates()
    clean_folder(final_output_folder)

    old_crop_dir = VOICE_ROOT / "Seperated_Audios" / "Old_Code_2s_Cropped"
    stft_output_root = VOICE_ROOT / "Spectogram_Output"

    clean_folder(old_crop_dir)
    clean_folder(stft_output_root)

    VOICE_OLD_CROP.old_code_crop(input_path, old_crop_dir)
    VOICE_STFT.process_all(
        VOICE_ROOT / "Seperated_Audios" / "Beacon_Cropped" / "Segments_2s",
        old_crop_dir,
        stft_output_root,
    )
    VOICE_PREPARE.prepare_all(
        stft_output_root / "Beacon_STFT_15k_19p2k",
        stft_output_root / "Old_Code_STFT_15k_19p2k",
        final_output_folder,
    )

    return final_output_folder / "Old_Code_Prepared"


def run_voice_processing(input_path: Path, final_output_folder: Path) -> Path:
    try:
        VOICE_PIPELINE.run_voice_pipeline(input_path, final_output_folder)
        return final_output_folder / "Beacon_Prepared"
    except RuntimeError as error:
        if "beacon cropping failed" not in str(error):
            raise
        return run_old_code_voice_fallback(input_path, final_output_folder)


def process_video(video_path: Path, sample_dir: Path) -> tuple[int, int]:
    output_dir = sample_dir / f"{video_path.stem}_Image_and_Video"
    frames_dir = output_dir / "Frames"
    voices_dir = output_dir / "Voices"
    spectrogram_dir = output_dir / f"{video_path.stem}_Spectogram"

    clean_folder(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video_path, sample_dir / video_path.name)

    with tempfile.TemporaryDirectory(prefix=f"{video_path.stem}_", dir=Path(__file__).resolve().parent) as temp_name:
        temp_root = Path(temp_name)
        segment_dir = temp_root / "2S_Segmentation"
        separated_dir = temp_root / "Seperated_Output"
        sliced_frame_dir = temp_root / "Sliced_Frame"
        voice_spectrogram_work_dir = temp_root / "Voice_Spectogram_Output"

        try:
            if not VIDEO_SEGMENTATION.segment_video(video_path, segment_dir):
                raise RuntimeError(f"Video segmentation failed for {video_path}")

            if not VIDEO_AUDIO.separate_audio_video(video_path, separated_dir):
                raise RuntimeError(f"Audio extraction failed for {video_path}")

            if not VIDEO_FRAMES.slice_video_frames(video_path, sliced_frame_dir):
                raise RuntimeError(f"Frame extraction failed for {video_path}")

            copy_frames(sliced_frame_dir, frames_dir)
            voice_path = copy_voice_files(separated_dir / "Audios", voices_dir)

            prepared_spectrogram_dir = run_voice_processing(
                voice_path,
                voice_spectrogram_work_dir,
            )
            spectrogram_count = copy_prepared_spectrograms(
                prepared_spectrogram_dir,
                spectrogram_dir,
            )
        finally:
            VOICE_PIPELINE.clean_intermediates()

    frame_count = len(list(frames_dir.glob("*.jpg")))
    return frame_count, spectrogram_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Machine Learning/Sample_N inputs from videos in Samples/, "
            "matching the existing Sample_1 multimodal training layout."
        )
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=SAMPLES_ROOT,
        help="Folder containing source videos.",
    )
    parser.add_argument(
        "--ml-root",
        type=Path,
        default=ML_ROOT,
        help="Machine Learning folder where Sample_N folders are created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild a matching existing Sample_N folder for videos already imported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_root = args.samples_root.resolve()
    ml_root = args.ml_root.resolve()

    if not samples_root.is_dir():
        raise FileNotFoundError(f"Samples folder not found: {samples_root}")

    ml_root.mkdir(parents=True, exist_ok=True)
    videos = list_sample_videos(samples_root)

    if not videos:
        print(f"No videos found in {samples_root}")
        return

    for video_path in videos:
        existing_sample = find_existing_sample_for_video(ml_root, video_path.stem)

        if (
            existing_sample is not None
            and not args.overwrite
            and sample_has_complete_output(existing_sample, video_path.stem)
        ):
            print(f"[SKIP] {video_path.name} already appears in {existing_sample.name}")
            continue

        sample_dir = existing_sample if existing_sample is not None else next_sample_dir(ml_root)
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[BUILD] {video_path.name} -> {sample_dir.relative_to(REPO_ROOT)}")
        frame_count, spectrogram_count = process_video(video_path, sample_dir)
        print(
            f"[DONE] {sample_dir.name}: "
            f"{frame_count} frames, {spectrogram_count} spectrogram arrays"
        )


if __name__ == "__main__":
    main()
