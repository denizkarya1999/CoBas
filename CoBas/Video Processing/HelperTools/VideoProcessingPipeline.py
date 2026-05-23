import subprocess
import sys


PIPELINE_STEPS = [
    ("2-second segmentation", "2S_Segmentation.py"),
    ("full audio extraction", "VideoSoundSeperator.py"),
    ("frame slicing", "VideoFrameSlicer.py"),
]


def run_step(label, script_name):
    print(f"\nStarting {label}...")

    result = subprocess.run([sys.executable, script_name])

    if result.returncode != 0:
        print(f"Pipeline stopped: {label} failed.")
        return False

    print(f"Finished {label}.")
    return True


def main():
    for label, script_name in PIPELINE_STEPS:
        if not run_step(label, script_name):
            sys.exit(1)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
