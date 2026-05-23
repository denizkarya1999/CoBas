import argparse
import subprocess
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def slice_video_frames(input_video_path, output_folder):
    input_video_path = Path(input_video_path)
    output_folder = Path(output_folder)

    if not input_video_path.exists():
        print(f"Error: Input video file does not exist: {input_video_path}")
        return False

    output_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = output_folder / f"{input_video_path.stem}_frame%03d.jpg"

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
        print(f"Frames saved to: {output_folder}")
        return True

    print(f"Frame extraction failed: {input_video_path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output-folder", required=True)
    args = parser.parse_args()

    if not slice_video_frames(args.input_video, args.output_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
