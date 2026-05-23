import os
import subprocess
from pathlib import Path


INPUT_VIDEO = "Jeniffer_Lopez_Play_Live.mp4"
CUT_SECONDS = 2
SEGMENTATION_FOLDER = "2S_Segmentation"


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


def add_intervals_to_segment_names(output_folder, video_name, cut_seconds):
    for segment_path in sorted(Path(output_folder).glob(f"{video_name}_seg*.mp4")):
        segment_marker = "_seg"
        marker_index = segment_path.stem.rfind(segment_marker)

        if marker_index == -1:
            continue

        segment_number_text = segment_path.stem[marker_index + len(segment_marker):]

        if not segment_number_text.isdigit():
            continue

        segment_number = int(segment_number_text)
        start_second = segment_number * cut_seconds
        end_second = start_second + cut_seconds
        interval = f"{format_seconds(start_second)}-{format_seconds(end_second)}"
        new_path = segment_path.with_name(
            f"{video_name}_seg{segment_number:03d}_{interval}.mp4"
        )

        os.replace(segment_path, new_path)


def segment_video(
    input_video_path=INPUT_VIDEO,
    output_folder=SEGMENTATION_FOLDER,
    cut_seconds=CUT_SECONDS
):
    input_video_path = Path(input_video_path)
    output_folder = Path(output_folder)

    if not input_video_path.exists():
        print(f"Error: Input video file does not exist: {input_video_path}")
        return False

    if cut_seconds <= 0:
        print("Error: cut_seconds must be greater than 0.")
        return False

    output_folder.mkdir(parents=True, exist_ok=True)

    video_name = input_video_path.stem
    output_pattern = output_folder / f"{video_name}_seg%03d.mp4"

    print(f"Segmenting video every {cut_seconds} seconds: {input_video_path.name}")

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video_path),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(cut_seconds),
        "-reset_timestamps", "1",
        str(output_pattern)
    ]

    if not run_command(command):
        print(f"Video segmentation failed: {input_video_path}")
        return False

    add_intervals_to_segment_names(output_folder, video_name, cut_seconds)
    print(f"Segments saved: {output_folder}")
    return True


def main():
    if not segment_video():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
