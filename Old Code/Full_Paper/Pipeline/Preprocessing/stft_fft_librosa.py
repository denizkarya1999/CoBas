import librosa
import numpy as np

def compute_frequency_domain_spectrogram(
        audio_path: str,
        sampling_rate: int,
        n_fft: int = 2048,
        hop_length: int = 240,
        win_length: int = 480,
        fmin: int = 15000, # DEFAULT: 15000
        fmax: int = 19200): # DEFAULT: 19200

    """
    :param audio_path:
    :param sampling_rate: AMPLITUDE SAMPLES PER SECOND
    :param n_fft:
    :param hop_length: OVERLAP
    :param win_length:
    :param fmin: LOWER FREQUENCY BOUND
    :param fmax: UPPER FREQUENCY BOUND
    """

    if ".wav" not in audio_path:
        print("[ERROR]: 'audio_path' ONLY TAKES AUDIO FILES (.wav)!")
        return

    # Loads audio clip (.wav)
    y, sr = librosa.load(audio_path, sr=sampling_rate)

    # Computes Short-Time Fourier Transform (STFT) --> complex matrix
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Converts to magnitude
    S_mag = np.abs(S)

    # Converts to dB scale if desired
    S_db = librosa.amplitude_to_db(S_mag)

    # Frequency axis for STFT
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)

    # Finds index range for [fmin, fmax]
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    # Crop
    S_ultra = S_db[idx, :]

    return S_ultra.astype(np.float32), freqs, idx
    # return S_ultra.astype(np.float32)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    AUDIO_PATH = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/official00/train/audio_segments/crpd_control_train_seg001.wav"
    SAMPLING_RATE = 48000
    N_FFT = 2048
    HOP_LENGTH = 240
    WIN_LENGTH = 480
    FMIN = 15000
    FMAX = 19200
    # FMIN = 0
    # FMAX = 20000

    s_ultra, freqs, idx = compute_frequency_domain_spectrogram(
        audio_path=AUDIO_PATH,
        sampling_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )

    # plt.figure(figsize=(10, 6))
    # plt.imshow(s_ultra, aspect='auto', origin='lower', cmap='inferno')
    # plt.colorbar()
    # plt.title("Ultrasonic Linear-Frequency Spectrogram (15–19.2 kHz)")
    # plt.savefig("stft_test", dpi=300, bbox_inches="tight")
    # plt.show()



    # plt.imshow(s_ultra,
    #            aspect='auto',
    #            origin='lower',
    #            extent=[0, s_ultra.shape[1], freqs[idx][0], freqs[idx][-1]],
    #            cmap='inferno')
    # plt.colorbar()
    # plt.show()

    plt.figure(figsize=(10, 6), constrained_layout=False)
    times = np.arange(s_ultra.shape[1]) * HOP_LENGTH / SAMPLING_RATE
    print(times)
    plt.imshow(
        s_ultra,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], FMIN, FMAX],
        cmap='inferno'
    )

    plt.yticks(
        np.arange(FMIN, FMAX + 1, 500)
    )

    plt.xticks(
        np.arange(0, times[-1] + 0.001, 0.25)
    )

    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Magnitude (dB re. max)", fraction=0.046, pad=0.02)
    plt.savefig("paper/notes/15000_19200_labeled", dpi=300, bbox_inches='tight', pad_inches=0.2)
    # plt.show()







    # fig, ax = plt.subplots()
    #
    # im = ax.imshow(s_ultra, aspect='auto', origin='lower', cmap='inferno')
    #
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    # ax.set_title("")
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # plt.savefig("ftest1.png", dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    # plt.show()
