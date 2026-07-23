"""Generate a single chirp pulse that lasts exactly two seconds."""

from pathlib import Path
import wave

import numpy as np


# ==========================================================
# USER SETTINGS
# ==========================================================

sample_rate = 48_000
pulse_duration_sec = 2.0
start_freq = 15_000.0
end_freq = 19_200.0
amplitude = 0.85
fade_ms = 5.0

output_dir = Path(__file__).resolve().parent / "Output"
output_path = output_dir / "2_second_pulse.wav"


def apply_fade(signal, duration_ms):
    """Fade both ends of the pulse to prevent playback clicks."""
    fade_samples = int(round(duration_ms * 1e-3 * sample_rate))

    if fade_samples == 0 or 2 * fade_samples >= signal.size:
        return signal

    window = np.ones_like(signal)
    window[:fade_samples] = np.linspace(
        0.0, 1.0, fade_samples, dtype=np.float32
    )
    window[-fade_samples:] = np.linspace(
        1.0, 0.0, fade_samples, dtype=np.float32
    )
    return signal * window


def build_pulse():
    """Build a two-second linear chirp using the existing protocol frequencies."""
    sample_count = int(round(sample_rate * pulse_duration_sec))
    time = np.arange(sample_count, dtype=np.float32) / sample_rate

    sweep_rate = (end_freq - start_freq) / pulse_duration_sec
    phase = 2.0 * np.pi * (
        start_freq * time + 0.5 * sweep_rate * time * time
    )

    pulse = (amplitude * np.sin(phase)).astype(np.float32)
    return apply_fade(pulse, fade_ms)


def write_wav(signal):
    """Save the pulse as a mono, 16-bit WAV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main():
    pulse = build_pulse()
    write_wav(pulse)

    duration = pulse.size / sample_rate
    print(f"Wrote: {output_path}")
    print(f"Pulse duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
