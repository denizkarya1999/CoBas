import subprocess
from pathlib import Path


INPUT_VIDEO = "Jeniffer_Lopez_Play_Live.mp4"
SLICED_FRAME_FOLDER = "Sliced_Frame"


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def slice_video_frames(
    input_video_path=INPUT_VIDEO,
    sliced_frame_folder=SLICED_FRAME_FOLDER
):
    """
    Extracts one frame every 2 seconds, matching fps=0.5 in the notebook.

    Frames are saved as:
        Sliced_Frame/<video_name>_frame000.jpg
    """

    input_video_path = Path(input_video_path)
    frames_folder = Path(sliced_frame_folder)

    if not input_video_path.exists():
        print(f"Error: Input video file does not exist: {input_video_path}")
        return False

    frames_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = frames_folder / f"{input_video_path.stem}_frame%03d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-vf", "fps=0.5",
        "-start_number", "0",
        "-qscale:v", "2",
        str(output_pattern)
    ]

    if run_command(command):
        print(f"Frames saved to: {frames_folder}")
        return True

    print(f"Frame extraction failed: {input_video_path}")
    return False


def main():
    if not slice_video_frames():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
