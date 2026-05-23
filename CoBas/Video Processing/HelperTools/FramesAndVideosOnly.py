import shutil
from pathlib import Path


SLICED_FRAME_FOLDER = "Sliced_Frame"
SEPARATED_OUTPUT_FOLDER = "Seperated_Output"
INPUT_VIDEO = "Jeniffer_Lopez_Play_Live.mp4"
OUTPUT_FOLDER = f"{Path(INPUT_VIDEO).stem}_Image_and_Video"

INTERMEDIATE_FOLDERS = [
    "2S_Segmentation",
    "Seperated_Output",
    "Separated_Output",
    "Sliced_Frame",
    "Sliced_Frames",
    "_Temp_2S_Segments",
    "Frames_And_Voice",
    "Output",
]


def delete_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def copy_frames(output_frames_folder):
    source_frames_folder = Path(SLICED_FRAME_FOLDER)

    if not source_frames_folder.exists():
        print(f"Error: Frames folder does not exist: {source_frames_folder}")
        return False

    shutil.copytree(source_frames_folder, output_frames_folder)
    print(f"Frames saved to: {output_frames_folder}")
    return True


def copy_voices(output_voices_folder):
    source_voices_folder = Path(SEPARATED_OUTPUT_FOLDER) / "Audios"

    if not source_voices_folder.exists():
        print(f"Error: Voices folder does not exist: {source_voices_folder}")
        return False

    output_voices_folder.mkdir(parents=True, exist_ok=True)

    voice_paths = sorted(source_voices_folder.glob("*.wav"))

    if not voice_paths:
        print(f"Error: No voice files found in: {source_voices_folder}")
        return False

    for voice_path in voice_paths:
        output_voice_path = output_voices_folder / voice_path.name
        shutil.copy2(voice_path, output_voice_path)

    print(f"Voices saved to: {output_voices_folder}")
    return True


def clean_intermediate_folders():
    for folder in INTERMEDIATE_FOLDERS:
        delete_folder(folder)


def keep_frames_and_voices_only():
    output_folder = Path(OUTPUT_FOLDER)
    output_frames_folder = output_folder / "Frames"
    output_voices_folder = output_folder / "Voices"

    delete_folder(output_folder)

    frames_success = copy_frames(output_frames_folder)
    voices_success = copy_voices(output_voices_folder)

    if not frames_success or not voices_success:
        return False

    clean_intermediate_folders()

    print(f"\nFinal output saved in {OUTPUT_FOLDER} folder.")
    print("Original video remains beside Video_Processing_Pipeline.py.")
    return True


def main():
    if not keep_frames_and_voices_only():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
