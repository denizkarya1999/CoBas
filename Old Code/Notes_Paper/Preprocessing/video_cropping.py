import os
import subprocess
import librosa
import numpy as np
from scipy.signal import find_peaks


# ============================================================
# 1. Extract audio from the (already cropped) video
# ============================================================
def extract_audio(video_path, sample_rate=48000):
    base = os.path.splitext(video_path)[0]
    wav_path = base + "_audio.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path
    ], check=True)

    return wav_path


# ============================================================
# 2. Detect chirp timestamps using Spectral Flux
# ============================================================
def detect_chirps(audio_path, sr=48000, n_fft=2048, hop=240,
                  fmin=15000, fmax=19200):

    y, _ = librosa.load(audio_path, sr=sr)

    # Full spec
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))

    # Frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Crop to ultrasonic band
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    S_ultra = S[band, :]   # <-- ONLY chirp energy

    # Spectral flux ON THE ULTRASONIC BAND ONLY
    flux = np.sum((S_ultra[:, 1:] - S_ultra[:, :-1])**2, axis=0)

    # Threshold relative to ULTRASONIC energy, not whole spectrum
    threshold = np.mean(flux) + 2 * np.std(flux)

    # Find peaks (one per sweep)
    peaks, _ = find_peaks(flux, height=threshold, distance=20)

    # Convert frame index → time (seconds)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

    return times


# ============================================================
# 3. Choose crop window using internal chirp indices
# ============================================================
def compute_alignment_window(chirp_times, start_idx=2, end_idx=-3):
    if len(chirp_times) < abs(end_idx):
        raise ValueError("Video does not contain enough chirps to apply chosen indices.")

    crop_start = chirp_times[start_idx]
    crop_end = chirp_times[end_idx]

    if crop_end <= crop_start:
        raise ValueError("Computed crop window is invalid — check chirp indices.")

    return crop_start, crop_end


# ============================================================
# 4. Crop video (copy streams, no re-encoding)
# ============================================================
def crop_video(input_video, output_video, crop_start, crop_end):
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{crop_start}",
        "-to", f"{crop_end}",
        "-i", input_video,
        "-c:v", "copy",
        "-c:a", "copy",
        output_video
    ], check=True)


# ============================================================
# 5. Full pipeline for one video
# ============================================================
def align_video(video_path, start_idx=2, end_idx=-3):
    print(f"\n=== Processing: {video_path} ===")

    # Step A — Extract audio
    audio_path = extract_audio(video_path)
    print(f"Extracted audio: {audio_path}")

    # Step B — Detect chirps
    chirp_times = detect_chirps(audio_path)
    print(f"Detected {len(chirp_times)} chirps.")
    print("First few chirps (sec):", chirp_times[:5])
    print("Last few chirps (sec):", chirp_times[-5:])

    # Step C — Determine crop window
    crop_start, crop_end = compute_alignment_window(
        chirp_times,
        start_idx=start_idx,
        end_idx=end_idx
    )
    print(f"Crop window → start: {crop_start:.3f} sec, end: {crop_end:.3f} sec")
    print(f"Output duration: {crop_end - crop_start:.3f} sec")

    # Step D — Produce aligned video
    output_video = video_path.replace(".mov", "_aligned.mov")
    crop_video(video_path, output_video, crop_start, crop_end)

    print(f"Aligned video saved to: {output_video}")

    return output_video, crop_end - crop_start


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    video_path = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset/raw_videos_10m/100p_10m.mov"

    # start_idx = which chirp to begin at (third chirp is index 2)
    # end_idx = which chirp to end at (third from last is index -3)
    align_video(video_path, start_idx=2, end_idx=-3)
