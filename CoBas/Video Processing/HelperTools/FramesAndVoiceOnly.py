import shutil
import subprocess
from pathlib import Path


INPUT_VIDEO = "Jeniffer_Lopez_Play_Live.mp4"
CUT_SECONDS = 2

FINAL_OUTPUT_FOLDER = "Frames_And_Voice"
TEMP_SEGMENT_FOLDER = "_Temp_2S_Segments"

GENERATED_FOLDERS_TO_DELETE = [
    "2S_Segmentation",
    "Seperated_Output",
    "Separated_Output",
    "Sliced_Frame",
    "Sliced_Frames",
    TEMP_SEGMENT_FOLDER,
]


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


def delete_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def clean_generated_folders():
    for folder in GENERATED_FOLDERS_TO_DELETE:
        delete_folder(folder)


def prepare_output_folders():
    final_output_folder = Path(FINAL_OUTPUT_FOLDER)
    frames_folder = final_output_folder / "Frames"
    voice_folder = final_output_folder / "Voice"

    delete_folder(final_output_folder)

    frames_folder.mkdir(parents=True, exist_ok=True)
    voice_folder.mkdir(parents=True, exist_ok=True)

    return frames_folder, voice_folder


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

        segment_path.replace(new_path)


def segment_video(input_video_path, segment_folder, cut_seconds):
    input_video_path = Path(input_video_path)
    segment_folder = Path(segment_folder)

    if not input_video_path.exists():
        print(f"Error: Input video does not exist: {input_video_path}")
        return False

    segment_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = segment_folder / f"{input_video_path.stem}_seg%03d.mp4"

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
        print("Error: Video segmentation failed.")
        return False

    add_intervals_to_segment_names(segment_folder, input_video_path.stem, cut_seconds)
    return True


def save_voice(segment_path, voice_folder):
    output_voice_path = voice_folder / f"{segment_path.stem}_voice.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(segment_path),
        "-map", "0:a:0",
        "-vn",
        "-acodec", "pcm_s24le",
        str(output_voice_path)
    ]

    if run_command(command):
        print(f"Voice saved: {output_voice_path}")
        return True

    print(f"Voice extraction failed: {segment_path}")
    return False


def save_frames(segment_path, frames_folder):
    segment_frames_folder = frames_folder / segment_path.stem
    segment_frames_folder.mkdir(parents=True, exist_ok=True)

    output_pattern = segment_frames_folder / "frame_%06d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(segment_path),
        str(output_pattern)
    ]

    if run_command(command):
        print(f"Frames saved: {segment_frames_folder}")
        return True

    print(f"Frame extraction failed: {segment_path}")
    return False


def save_frames_and_voice_only():
    clean_generated_folders()
    frames_folder, voice_folder = prepare_output_folders()

    if not segment_video(INPUT_VIDEO, TEMP_SEGMENT_FOLDER, CUT_SECONDS):
        return False

    segment_paths = sorted(Path(TEMP_SEGMENT_FOLDER).glob("*.mp4"))

    if not segment_paths:
        print(f"Error: No segments were created in {TEMP_SEGMENT_FOLDER}.")
        return False

    success_count = 0

    for segment_path in segment_paths:
        voice_success = save_voice(segment_path, voice_folder)
        frames_success = save_frames(segment_path, frames_folder)

        if voice_success and frames_success:
            success_count += 1

    delete_folder(TEMP_SEGMENT_FOLDER)

    print(
        f"\nFinished saving frames and voice for "
        f"{success_count}/{len(segment_paths)} segments."
    )
    print(f"Final output folder: {FINAL_OUTPUT_FOLDER}")
    print(f"Original video kept: {INPUT_VIDEO}")

    return success_count == len(segment_paths)


def main():
    save_frames_and_voice_only()


if __name__ == "__main__":
    main()
