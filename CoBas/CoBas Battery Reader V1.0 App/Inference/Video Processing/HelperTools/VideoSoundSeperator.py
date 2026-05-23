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


def extract_single_audio(input_video_path, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = output_folder / f"{input_video_path.stem}.wav"

    if not input_video_path.exists():
        print(f"Error: Input video not found: {input_video_path}")
        return False

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-vn",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_audio_path)
    ]

    if run_command(command):
        print(f"Audio saved: {output_audio_path}")
        return True

    print(f"Audio extraction failed: {input_video_path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output-folder", required=True)
    args = parser.parse_args()

    if not extract_single_audio(args.input_video, args.output_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
