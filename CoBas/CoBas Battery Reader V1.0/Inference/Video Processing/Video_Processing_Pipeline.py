import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CUT_SECONDS = 2


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
    segment_folder = work_folder / "2S_Segmentation"
    frames_folder = output_folder / "Frames"
    voices_folder = output_folder / "Voices"

    return (
        output_folder,
        work_root_folder,
        work_folder,
        segment_folder,
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
        frames_folder,
        voices_folder
    ) = build_pipeline_paths(input_video_path)

    delete_folder(output_folder)
    delete_folder(work_folder)
    frames_folder.mkdir(parents=True, exist_ok=True)
    voices_folder.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "2-second segmentation",
            [
                sys.executable,
                str(helper_folder / "2S_Segmentation.py"),
                str(input_video_path),
                "--output-folder",
                str(segment_folder),
                "--cut-seconds",
                str(CUT_SECONDS)
            ]
        ),
        (
            "audio extraction",
            [
                sys.executable,
                str(helper_folder / "VideoSoundSeperator.py"),
                str(segment_folder),
                "--output-folder",
                str(voices_folder)
            ]
        ),
        (
            "frame extraction",
            [
                sys.executable,
                str(helper_folder / "VideoFrameSlicer.py"),
                str(input_video_path),
                "--output-folder",
                str(frames_folder)
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
    ]

    for label, command in steps:
        if not run_step(label, command):
            return False

    delete_folder(work_root_folder)

    print("\nPipeline finished successfully.")
    print(f"Frames saved in: {output_folder / 'Frames'}")
    print(f"Voices saved in: {output_folder / 'Voices'}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    args = parser.parse_args()

    if not process_video(args.input_video):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
