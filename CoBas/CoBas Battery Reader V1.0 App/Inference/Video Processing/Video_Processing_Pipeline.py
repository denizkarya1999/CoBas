import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def get_app_root():
    return Path(__file__).resolve().parents[2]


def run_step(label, command):
    print(f"\nStarting {label}...")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Pipeline stopped: {label} failed.")
        return False

    print(f"Finished {label}.")
    return True


def delete_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def build_pipeline_paths(input_video_path):
    app_root = get_app_root()
    captures_folder = app_root / "Captures"
    video_name = input_video_path.stem

    output_folder = captures_folder / f"{video_name}_Image_and_Video"
    work_root_folder = captures_folder / "_Video_Processing_Work"
    work_folder = work_root_folder / video_name
    segment_folder = work_folder / "Beacon_Segmentation"
    silent_video_folder = work_folder / "Silent_Videos"
    frames_folder = output_folder / "Frames"
    voices_folder = output_folder / "Voices"

    return (
        output_folder,
        work_root_folder,
        work_folder,
        segment_folder,
        silent_video_folder,
        frames_folder,
        voices_folder
    )


def process_video(input_video):
    input_video_path = Path(input_video).resolve()

    if not input_video_path.exists():
        print(f"Error: Input video not found: {input_video_path}")
        return False

    pipeline_folder = Path(__file__).resolve().parent
    helper_folder = pipeline_folder / "HelperTools"

    (
        output_folder,
        work_root_folder,
        work_folder,
        segment_folder,
        silent_video_folder,
        frames_folder,
        voices_folder
    ) = build_pipeline_paths(input_video_path)

    delete_folder(output_folder)
    delete_folder(work_folder)
    silent_video_folder.mkdir(parents=True, exist_ok=True)
    frames_folder.mkdir(parents=True, exist_ok=True)
    voices_folder.mkdir(parents=True, exist_ok=True)

    voice_pipeline_script = get_app_root() / "Inference" / "Voice Processing" / "Voice_Processing_Pipeline.py"
    voice_stft_folder = output_folder / f"{input_video_path.stem}_Spectogram"
    voice_stft_folder.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "beacon video segmentation",
            [
                sys.executable,
                str(helper_folder / "Beacon_Video_Segmentation.py"),
                str(input_video_path),
                "--output-folder",
                str(segment_folder)
            ]
        ),
        (
            "frames and voices finalization",
            [
                sys.executable,
                str(helper_folder / "FramesAndVideosOnly.py"),
                "--output-folder",
                str(output_folder),
                "--work-folder",
                str(work_folder)
            ]
        ),
        (
            "voice STFT preprocessing",
            [
                sys.executable,
                str(voice_pipeline_script),
                str(voices_folder),
                "--output-folder",
                str(voice_stft_folder),
                "--already-cropped"
            ]
        ),
    ]

    if not run_step(steps[0][0], steps[0][1]):
        return False

    segment_paths = sorted(segment_folder.glob("*.mp4"))
    if not segment_paths:
        print(f"Pipeline stopped: no beacon video segments found in {segment_folder}.")
        return False

    for segment_path in segment_paths:
        split_command = [
            sys.executable,
            str(helper_folder / "VideoSoundSeperator.py"),
            str(segment_path),
            "--output-folder",
            str(voices_folder),
            "--silent-video-folder",
            str(silent_video_folder),
        ]
        if not run_step(f"voice/silent-video split for {segment_path.name}", split_command):
            return False

    silent_video_paths = sorted(silent_video_folder.glob("*.mp4"))
    if not silent_video_paths:
        print(f"Pipeline stopped: no silent videos found in {silent_video_folder}.")
        return False

    for silent_video_path in silent_video_paths:
        frame_command = [
            sys.executable,
            str(helper_folder / "VideoFrameSlicer.py"),
            str(silent_video_path),
            "--output-folder",
            str(frames_folder),
        ]
        if not run_step(f"frame extraction for {silent_video_path.name}", frame_command):
            return False

    for label, command in steps[1:]:
        if not run_step(label, command):
            return False

    delete_folder(work_root_folder)

    print("\nPipeline finished successfully.")
    print(f"Frames saved in: {output_folder / 'Frames'}")
    print(f"Voice saved in: {voices_folder}")
    print(f"Spectrograms saved in: {voice_stft_folder}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    args = parser.parse_args()

    if not process_video(args.input_video):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
