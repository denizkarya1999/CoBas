import subprocess
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

    audio_output_folder.mkdir(parents=True, exist_ok=True)

    input_video_path = Path(input_video_path)
    output_audio_path = audio_output_folder / f"{input_video_path.stem}.wav"

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

    audio_success = run_command(audio_command)

    if audio_success:
        print(f"Audio saved: {output_audio_path}")
    else:
        print(f"Audio separation failed: {input_video_path}")

    return audio_success


def separate_audio_video(
    input_video_path=INPUT_VIDEO,
    output_folder=SEPARATED_OUTPUT_FOLDER
):
    success = separate_single_video(input_video_path, output_folder)
    if success:
        print("\nFinished extracting one unsegmented audio file.")
    return success


def main():
    if not separate_audio_video():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
