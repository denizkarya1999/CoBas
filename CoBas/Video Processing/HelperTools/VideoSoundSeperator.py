import subprocess
import argparse
from pathlib import Path


INPUT_VIDEO = "Jeniffer_Lopez_Play_Live.mp4"
CUT_SECONDS = 2
SEGMENTATION_FOLDER = "2S_Segmentation"
SEPARATED_OUTPUT_FOLDER = "Seperated_Output"


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def format_seconds(seconds):
    if float(seconds).is_integer():
        return f"{int(seconds):03d}s"

    return f"{seconds:06.2f}s".replace(".", "p")


def separate_single_video(input_video_path=INPUT_VIDEO, output_folder=SEPARATED_OUTPUT_FOLDER):
    audio_output_folder = Path(output_folder) / "Audios"
    silent_video_output_folder = Path(output_folder) / "Silent_Videos"

    audio_output_folder.mkdir(parents=True, exist_ok=True)
    silent_video_output_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = audio_output_folder / f"{input_video_path.stem}.wav"
    output_video_path = silent_video_output_folder / f"{input_video_path.stem}.mp4"

    if not input_video_path.exists():
        print(f"Error: Input video not found: {input_video_path}")
        return False

    print(f"Extracting audio: {input_video_path.name}")

    audio_command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-vn",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_audio_path)
    ]

    video_command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-an",
        "-c:v", "copy",
        str(output_video_path)
    ]

    audio_success = run_command(audio_command)
    video_success = run_command(video_command)

    if audio_success:
        print(f"Audio saved: {output_audio_path}")
    else:
        print(f"Audio separation failed: {input_video_path}")

    if video_success:
        print(f"Silent video saved: {output_video_path}")
    else:
        print(f"Silent video separation failed: {input_video_path}")

    return audio_success and video_success


def separate_audio_video(
    input_video_path=INPUT_VIDEO,
    output_folder=SEPARATED_OUTPUT_FOLDER
):
    success = separate_single_video(input_video_path, output_folder)
    if success:
        print("\nFinished extracting audio and silent video.")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", nargs="?", default=INPUT_VIDEO)
    parser.add_argument("--output-folder", default=SEPARATED_OUTPUT_FOLDER)
    args = parser.parse_args()

    if not separate_audio_video(args.input_video, args.output_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
