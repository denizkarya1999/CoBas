import shutil
import subprocess
import sys
from pathlib import Path


INPUT_VIDEO = Path("0p_2m2.mov")
FINAL_OUTPUT_FOLDER = f"{INPUT_VIDEO.stem}_Image_and_Video"

GENERATED_FOLDERS = [
    "Beacon_Segmentation",
    "Seperated_Output",
    "Separated_Output",
    "Sliced_Frame",
    "Sliced_Frames",
    "Output",
    FINAL_OUTPUT_FOLDER,
]


def run_step(label, command):
    print(f"\nStarting {label}...")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Pipeline stopped: {label} failed.")
        return False

    print(f"Finished {label}.")
    return True


def clean_generated_folders():
    for folder in GENERATED_FOLDERS:
        folder_path = Path(folder)

        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)


def main():
    if not INPUT_VIDEO.exists():
        print(f"Error: Input video not found: {INPUT_VIDEO}")
        raise SystemExit(1)

    clean_generated_folders()

    helper_folder = Path("HelperTools")
    beacon_folder = Path("Beacon_Segmentation")
    separated_folder = Path("Seperated_Output")
    silent_video_folder = separated_folder / "Silent_Videos"
    frames_folder = Path("Sliced_Frame")
    final_output_folder = Path(FINAL_OUTPUT_FOLDER)
    voice_pipeline_script = Path("..") / "Voice Processing" / "Voice_Processing_Pipeline.py"
    voice_spectrogram_folder = final_output_folder / f"{INPUT_VIDEO.stem}_Spectogram"

    if not run_step(
        "beacon video segmentation",
        [
            sys.executable,
            str(helper_folder / "Beacon_Video_Segmentation.py"),
            str(INPUT_VIDEO),
            "--output-folder",
            str(beacon_folder),
        ],
    ):
        raise SystemExit(1)

    segment_paths = sorted(beacon_folder.glob("*.mp4"))
    if not segment_paths:
        print(f"Error: no beacon video segments found in {beacon_folder}")
        raise SystemExit(1)

    for segment_path in segment_paths:
        if not run_step(
            f"voice/silent-video split for {segment_path.name}",
            [
                sys.executable,
                str(helper_folder / "VideoSoundSeperator.py"),
                str(segment_path),
                "--output-folder",
                str(separated_folder),
            ],
        ):
            raise SystemExit(1)

    silent_video_paths = sorted(silent_video_folder.glob("*.mp4"))
    if not silent_video_paths:
        print(f"Error: no silent videos found in {silent_video_folder}")
        raise SystemExit(1)

    for silent_video_path in silent_video_paths:
        if not run_step(
            f"frame slicing for {silent_video_path.name}",
            [
                sys.executable,
                str(helper_folder / "VideoFrameSlicer.py"),
                str(silent_video_path),
                "--output-folder",
                str(frames_folder),
            ],
        ):
            raise SystemExit(1)

    if not run_step(
        "frames and voices finalization",
        [
            sys.executable,
            str(helper_folder / "FramesAndVideosOnly.py"),
            "--output-folder",
            str(final_output_folder),
        ],
    ):
        raise SystemExit(1)

    if not run_step(
        "voice STFT preprocessing",
        [
            sys.executable,
            str(voice_pipeline_script),
            "--input",
            str(final_output_folder / "Voices"),
            "--output-folder",
            str(voice_spectrogram_folder),
            "--already-cropped",
        ],
    ):
        raise SystemExit(1)

    print("\nPipeline finished successfully.")
    print(f"Frames saved in: {FINAL_OUTPUT_FOLDER}/Frames")
    print(f"Voice saved in: {FINAL_OUTPUT_FOLDER}/Voices")
    print(f"Spectrograms saved in: {voice_spectrogram_folder}")


if __name__ == "__main__":
    main()
