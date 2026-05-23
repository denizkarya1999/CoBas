import shutil
import subprocess
import sys
from pathlib import Path


INPUT_VIDEO = Path("Jeniffer_Lopez_Play_Live.mp4")
FINAL_OUTPUT_FOLDER = f"{INPUT_VIDEO.stem}_Image_and_Video"

PIPELINE_STEPS = [
    ("2-second segmentation", Path("HelperTools") / "2S_Segmentation.py"),
    ("video/audio separation", Path("HelperTools") / "VideoSoundSeperator.py"),
    ("frame slicing", Path("HelperTools") / "VideoFrameSlicer.py"),
    ("frames and voices only", Path("HelperTools") / "FramesAndVideosOnly.py"),
]

GENERATED_FOLDERS = [
    "2S_Segmentation",
    "Seperated_Output",
    "Separated_Output",
    "Sliced_Frame",
    "Sliced_Frames",
    "Output",
    FINAL_OUTPUT_FOLDER,
]


def run_step(label, script_name):
    print(f"\nStarting {label}...")

    result = subprocess.run([sys.executable, str(script_name)])

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

    for label, script_name in PIPELINE_STEPS:
        if not run_step(label, script_name):
            raise SystemExit(1)

    print("\nPipeline finished successfully.")
    print(f"Frames saved in: {FINAL_OUTPUT_FOLDER}/Frames")
    print(f"Voices saved in: {FINAL_OUTPUT_FOLDER}/Voices")


if __name__ == "__main__":
    main()
