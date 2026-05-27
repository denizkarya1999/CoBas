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


def extract_audio_and_silent_video(input_video_path, voice_folder, silent_video_folder):
    voice_folder = Path(voice_folder)
    silent_video_folder = Path(silent_video_folder)
    voice_folder.mkdir(parents=True, exist_ok=True)
    silent_video_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = voice_folder / f"{input_video_path.stem}.wav"
    output_video_path = silent_video_folder / f"{input_video_path.stem}.mp4"

    if not input_video_path.exists():
        print(f"Error: Input video not found: {input_video_path}")
        return False

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
        print(f"Audio extraction failed: {input_video_path}")

    if video_success:
        print(f"Silent video saved: {output_video_path}")
    else:
        print(f"Silent video extraction failed: {input_video_path}")

    return audio_success and video_success


def extract_single_audio(input_video_path, output_folder):
    return extract_audio_and_silent_video(input_video_path, output_folder, Path(output_folder).parent / "Silent_Videos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--silent-video-folder", required=True)
    args = parser.parse_args()

    if not extract_audio_and_silent_video(args.input_video, args.output_folder, args.silent_video_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
