import subprocess
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import wave
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

# ==========================================================
# USER CONFIGURATION
# ==========================================================

VIDEO_PATH = "/Users/pedropaiva/Dev/Research/CoBasE-Energy/cobasFork/Full_Paper/AcousticTests/NewMetrics/croppingAndAlignmentTest_phone_000/raw_videos/100soc.MOV"
OUTPUT_VIDEO = "/Users/pedropaiva/Dev/Research/CoBasE-Energy/cobasFork/Full_Paper/AcousticTests/NewMetrics/croppingAndAlignmentTest_phone_000/cropped_videos/cropped_100soc.MOV"
TEMP_AUDIO = "temp_audio.wav"

SAMPLE_RATE = 48000

BEACON_FREQ = 10000        # Hz
BEACON_BW = 300            # ± Hz around beacon frequency
MIN_BEACON_DURATION = 2.5  # seconds DO NOT TOUCH
THRESHOLD_RATIO = 0.3      # relative energy threshold

# ==========================================================
# AUDIO EXTRACTION (VIDEO → WAV)
# ==========================================================

def extract_audio(video_path, audio_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",                 # overwrite output if it exists
            "-i", video_path,     # input video
            "-vn",                # remove video stream
            "-ac", "1",           # force mono
            "-ar", str(SAMPLE_RATE),  # resample for predictability
            audio_path
        ],
        check=True
    )

# ==========================================================
# LOAD WAV AUDIO (BIT-EXACT)
# ==========================================================

def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio.astype(np.float32), sr

# ==========================================================
# DSP UTILITIES
# ==========================================================

def bandpass(signal, sr, center, freq_bandwidth):
    nyq = sr * 0.5 # Nyquist freq. -> max freq. represented given a sr
    low = (center - freq_bandwidth) / nyq
    high = (center + freq_bandwidth) / nyq
    b, a = butter(4, [low, high], btype="band") # 4th order butterworth filter
    return filtfilt(b, a, signal) # applies butterworth filter given params a (numerator) and b (denominator) coefficients

def envelope(signal):
    analytic = hilbert(signal)
    return np.abs(analytic)

def plotting_env_threshold(env, threshold, sr, start, end):
    time = np.arange(len(env)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(time, env, label="Envelope (10 kHz band)")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")

    plt.axvline(start, color="yellow", linestyle="--", label="start")
    plt.axvline(end, color="orange", linestyle="--", label="end")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Beacon Envelope and Detection Threshold")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    OUTPUT_IMG = OUTPUT_VIDEO[:-3] + "png"
    plt.savefig(OUTPUT_IMG)

# ==========================================================
# BEACON DETECTION (CORE LOGIC)
# ==========================================================

def detect_beacons(audio, sr):
    # Isolate beacon frequency band
    filtered = bandpass(audio, sr, BEACON_FREQ, BEACON_BW)

    # Convert to energy envelope
    env = envelope(filtered)

    # Adaptive threshold
    threshold = THRESHOLD_RATIO * np.max(env)
    # threshold = THRESHOLD_RATIO * np.percentile(env, 95)

    # Binary mask
    mask = env > threshold
    indices = np.where(mask)[0]

    # Finds the indices where mask is True
    if len(indices) == 0:
        raise RuntimeError("No beacon energy detected")

    t_start = (indices[0]/sr) + MIN_BEACON_DURATION # BEGIN beacon timestamp
    t_end = (indices[-1]/sr) -  MIN_BEACON_DURATION # END beacon timestamp

    # t_start = t_BEGIN_beacon_end[0]
    # t_end = t_END_beacon_start[-1]

    # print(t_BEGIN_beacon_end)
    # print(t_END_beacon_start)
    # print(t_BEGIN_beacon_end == t_END_beacon_start)
    print(f"START: {t_start} | END: {t_end} | DIFF: {t_end-t_start}")

    plotting_env_threshold(env, threshold, sr, t_start, t_end)

    return t_start, t_end

# ==========================================================
# VIDEO CROPPING (TIMESTAMP-BASED)
# ==========================================================

def crop_video(video_path, output_path, t_start, t_end):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", f"{t_start:.6f}",
            "-to", f"{t_end:.6f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path
        ],
        check=True
    )

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():
    print("Extracting audio...")
    extract_audio(VIDEO_PATH, TEMP_AUDIO)

    print("Loading audio...")
    audio, sr = load_wav(TEMP_AUDIO)

    print("Detecting beacons...")
    t_start, t_end = detect_beacons(audio, sr)
    # detect_beacons(audio, sr)

    print(f"Cropping video from {t_start:.3f}s to {t_end:.3f}s")
    crop_video(VIDEO_PATH, OUTPUT_VIDEO, t_start, t_end)

    print("Done. Cropped video written to:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
