import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa


# ==========================================================
# FUNCTION TO LOAD AUDIO FILE
# Supports WAV, MP3, FLAC, OGG, etc.
# ==========================================================

def load_audio(path):

    print("Loading audio file...")

    with tqdm(total=3, desc="Reading Audio") as pbar:

        # Load audio
        x, sr = librosa.load(
            path,
            sr=None,
            mono=True
        )
        pbar.update(1)

        # Convert to float32
        x = x.astype(np.float32)
        pbar.update(1)

        # Normalize
        x /= np.max(np.abs(x))
        pbar.update(1)

    return x, sr


# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

output_dir = "Outputs"
os.makedirs(output_dir, exist_ok=True)


# ==========================================================
# INPUT AUDIO FILE
# ==========================================================

input_audio = (
    "/home/denizkaryaacikbas/Projects/CoBas/"
    "Organized Code/Data Acquisition/Inputs/"
    "5_15sPause_BeaconProtocol.wav"
)


# ==========================================================
# OUTPUT IMAGE FILE
# ==========================================================

output_path = os.path.join(
    output_dir,
    "waveform_overview.png"
)


# ==========================================================
# LOAD AUDIO
# ==========================================================

x, sr = load_audio(input_audio)

print("Generating time axis...")

t = np.arange(len(x)) / sr

print("Creating waveform plot...")


# ==========================================================
# CREATE WAVEFORM PLOT
# ==========================================================

with tqdm(total=4, desc="Rendering Plot") as pbar:

    plt.figure(figsize=(14, 4))
    pbar.update(1)

    plt.plot(t, x, linewidth=0.3)
    pbar.update(1)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Full waveform overview")

    plt.tight_layout()
    pbar.update(1)

    plt.savefig(output_path, dpi=300)
    pbar.update(1)


# ==========================================================
# REPORT
# ==========================================================

print(f"Saved waveform image as: {output_path}")

# import numpy as np
# import wave
#
# def load_wav(path):
#     with wave.open(path, 'rb') as wf:
#         sr = wf.getframerate()
#         x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
#     return x.astype(np.float32), sr
#
# x, sr = load_wav("your_file.wav")
#
# # extract BEGIN and END regions manually (adjust times)
# begin = x[int(60*sr):int(62*sr)]
# end   = x[int((len(x)/sr - 7)*sr):int((len(x)/sr - 5)*sr)]
#
# print("Max abs difference:", np.max(np.abs(begin - end)))
# print("RMS BEGIN:", np.sqrt(np.mean(begin**2)))
# print("RMS END:  ", np.sqrt(np.mean(end**2)))
#
