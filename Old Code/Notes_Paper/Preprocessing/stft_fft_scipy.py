import numpy as np
from scipy.signal import stft
import soundfile as sf

def compute_frequency_domain_spectrogram_scipy(
        audio_path: str,
        sampling_rate: int,
        n_fft: int = 2048,
        hop_length: int = 240,
        win_length: int = 480,
        fmin: float = 15000.0,
        fmax: float = 19200.0
):

    # reads audio file
    y, sr = sf.read(audio_path)

    # ensure audio sampling rate equals determined sampling rate
    if sr != sampling_rate:
        raise ValueError(f"Expected {sampling_rate} Hz, got {sr} Hz")

    # guarantees mono (1D)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # computes stft ndarray
    # returns frequency axis, time axis, and complex frequency/time matrix
    f, t, Zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False
    )

    # breaks complex matrix -> computes energy/amplitude proxy
    S_mag = np.abs(Zxx)

    # frequency masking and filtering
    freq_mask = (f >= fmin) & (f <= fmax)
    S_ultra = S_mag[freq_mask, :]
    freqs_ultra = f[freq_mask]

    return S_ultra.astype(np.float32), freqs_ultra, t
    # return S_ultra.astype(np.float32)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    AUDIO_PATH = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/Test1/audio_segments/crpd_control_seg060.wav"
    SAMPLING_RATE = 48000
    N_FFT = 2048
    HOP_LENGTH = 240
    WIN_LENGTH = 480
    FMIN = 15000
    FMAX = 19200
    # FMIN = 5000
    # FMAX = 20000

    s_ultra, freqs, idx = compute_frequency_domain_spectrogram_scipy(
        audio_path=AUDIO_PATH,
        sampling_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )

    fig, ax = plt.subplots()

    im = ax.imshow(s_ultra, aspect='auto', origin='lower', cmap='inferno')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig("crpd_control_060_TEST.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
