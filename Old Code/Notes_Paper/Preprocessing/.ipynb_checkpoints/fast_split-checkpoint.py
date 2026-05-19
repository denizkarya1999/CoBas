import subprocess, os
import glob

def fast_split(video_path, out_audio_dir, out_frame_dir, skip_if_exists=True):
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_pattern = os.path.join(out_audio_dir, f"{base}_seg%03d.wav")
    frame_pattern = os.path.join(out_frame_dir, f"{base}_frame%03d.jpg")

    # Simple existence check: if first segment + first frame exist, assume this video is done
    first_seg  = os.path.join(out_audio_dir, f"{base}_seg000.wav")
    first_frame = os.path.join(out_frame_dir, f"{base}_frame000.jpg")

    if skip_if_exists and os.path.exists(first_seg) and os.path.exists(first_frame):
        print(f"[SKIP] {video_path} already split.")
        return

    # Audio segments
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-f", "segment", "-segment_time", "2",
            "-ar", "48000", "-ac", "1",
            "-c:a", "pcm_s16le",
            audio_pattern
        ],
        check=True
    )

    # Frame extraction
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "fps=0.5",
            "-start_number", "0",
            "-qscale:v", "2",
            frame_pattern
        ],
        check=True
    )

if __name__ == "__main__":

    RAW_VIDEOS_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset1/lists/merges"
    AUDIO_SEGMENTS_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset1/audio_segments"
    VIDEO_FRAMES_DIR = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset1/video_frames"

    raw_video_files = glob.glob(RAW_VIDEOS_DIR + "/*")

    for video_file in raw_video_files:

        fast_split(video_path=video_file,
                   out_audio_dir=AUDIO_SEGMENTS_DIR,
                   out_frame_dir=VIDEO_FRAMES_DIR)
