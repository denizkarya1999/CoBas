import argparse
import shutil
from pathlib import Path


def delete_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def finalize_output(output_folder, work_folder):
    output_folder = Path(output_folder)
    frames_folder = output_folder / "Frames"
    voices_folder = output_folder / "Voices"

    if not frames_folder.exists():
        print(f"Error: Frames folder does not exist: {frames_folder}")
        return False

    if not voices_folder.exists():
        print(f"Error: Voices folder does not exist: {voices_folder}")
        return False

    if not any(frames_folder.glob("*.jpg")):
        print(f"Error: No frame files found in: {frames_folder}")
        return False

    if not any(voices_folder.glob("*.wav")):
        print(f"Error: No voice files found in: {voices_folder}")
        return False

    delete_folder(work_folder)

    print(f"Final output saved in: {output_folder}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--work-folder", required=True)
    args = parser.parse_args()

    if not finalize_output(args.output_folder, args.work_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
