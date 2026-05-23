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


def build_audio_file_name(video_path):
    segment_marker = "_seg"
    marker_index = video_path.stem.rfind(segment_marker)

    if marker_index == -1:
        return f"{video_path.stem}.wav"

    base_name = video_path.stem[:marker_index]
    segment_number = video_path.stem[marker_index + len(segment_marker):][:3]

    if not segment_number.isdigit():
        return f"{video_path.stem}.wav"

    interval_start_index = marker_index + len(segment_marker) + len(segment_number)
    interval_part = video_path.stem[interval_start_index:]

    return f"{base_name}_seg{segment_number}{interval_part}.wav"


def extract_single_audio(input_video_path, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = output_folder / build_audio_file_name(input_video_path)

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


def extract_audio_segments(input_folder, output_folder):
    input_folder = Path(input_folder)
    input_videos = sorted(input_folder.glob("*.mp4"))

    if not input_videos:
        print(f"Error: No segmented videos found in: {input_folder}")
        return False

    success_count = 0

    for input_video_path in input_videos:
        if extract_single_audio(input_video_path, output_folder):
            success_count += 1

    print(f"\nFinished extracting audio for {success_count}/{len(input_videos)} segments.")
    return success_count == len(input_videos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--output-folder", required=True)
    args = parser.parse_args()

    if not extract_audio_segments(args.input_folder, args.output_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
