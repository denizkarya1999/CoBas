import subprocess
from pathlib import Path


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


def separate_single_video(input_video_path, output_folder, cut_seconds=CUT_SECONDS):
    audio_output_folder = Path(output_folder) / "Audios"

    audio_output_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = audio_output_folder / build_audio_file_name(input_video_path)

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

    audio_success = run_command(audio_command)

    if audio_success:
        print(f"Audio saved: {output_audio_path}")
    else:
        print(f"Audio separation failed: {input_video_path}")

    return audio_success


def separate_audio_video(
    input_folder=SEGMENTATION_FOLDER,
    output_folder=SEPARATED_OUTPUT_FOLDER,
    cut_seconds=CUT_SECONDS
):
    input_folder = Path(input_folder)
    input_videos = sorted(input_folder.glob("*.mp4"))

    if not input_videos:
        print(f"Error: No segmented videos found in: {input_folder}")
        return False

    success_count = 0

    for input_video_path in input_videos:
        if separate_single_video(input_video_path, output_folder, cut_seconds):
            success_count += 1

    print(f"\nFinished separating {success_count}/{len(input_videos)} video files.")
    return success_count == len(input_videos)


def main():
    if not separate_audio_video():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
