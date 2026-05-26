import argparse
import os
import wave

import numpy as np
import sounddevice as sd
from tqdm import tqdm


# ==========================================================
# USER SETTINGS
# ==========================================================

sample_rate = 48000

# --- Alignment structure ---
initial_silence_sec = 30.0
beacon_freq = 10000
beacon_duration_sec = 2.0
guard_silence_sec = 1.5
tail_silence_sec = 5.0

# --- Chirp pulse train ---
pulse_duration = 0.10
gap_duration = 0.05
start_freq = 15000.0
end_freq = 19200.0
amplitude = 0.85
fade_ms = 5.0

cycles_total = 5
active_secs = 60.0

# ==========================================================
# SAVE INTO INPUTS FOLDER
# ==========================================================

input_dir = "Inputs"
os.makedirs(input_dir, exist_ok=True)

output_file = f"{cycles_total}_15sPause_BeaconProtocol.wav"
output_path = os.path.join(input_dir, output_file)


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def silence(sec):
    return np.zeros(int(round(sec * sample_rate)), dtype=np.float32)


def tone(freq, sec):
    t = np.arange(int(round(sec * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def apply_fade(x, fade_ms):
    fade_samp = int(round(fade_ms * 1e-3 * sample_rate))

    if fade_samp == 0 or 2 * fade_samp >= x.size:
        return x

    w = np.ones_like(x)
    w[:fade_samp] = np.linspace(0.0, 1.0, fade_samp, dtype=np.float32)
    w[-fade_samp:] = np.linspace(1.0, 0.0, fade_samp, dtype=np.float32)

    return x * w


def build_signal():
    with tqdm(total=7, desc="Generating Protocol") as pbar:
        beacon_tone = apply_fade(tone(beacon_freq, beacon_duration_sec), fade_ms)
        pbar.update(1)

        ns_pulse = int(round(sample_rate * pulse_duration))
        t = np.arange(ns_pulse, dtype=np.float32) / sample_rate

        k = (end_freq - start_freq) / pulse_duration
        phase = 2.0 * np.pi * (start_freq * t + 0.5 * k * t * t)

        pulse = (amplitude * np.sin(phase)).astype(np.float32)
        pulse = apply_fade(pulse, fade_ms)
        pbar.update(1)

        gap = silence(gap_duration)
        small_cycle = np.concatenate([pulse, gap])
        pbar.update(1)

        ns_active = int(round(active_secs * sample_rate))
        cycles_in_block = ns_active // small_cycle.size
        residual = ns_active - cycles_in_block * small_cycle.size

        active_block = np.concatenate(
            [small_cycle] * cycles_in_block +
            ([small_cycle[:residual]] if residual else [])
        )
        pbar.update(1)

        chirps = np.concatenate([active_block] * cycles_total)
        pbar.update(1)

        full_signal = np.concatenate([
            silence(initial_silence_sec),
            beacon_tone,
            silence(guard_silence_sec),
            chirps,
            silence(guard_silence_sec),
            beacon_tone,
            silence(tail_silence_sec)
        ])
        pbar.update(1)

        pcm = np.rint(
            np.clip(full_signal, -1.0, 1.0) * 32767.0
        ).astype(np.int16)
        pbar.update(1)

    return full_signal, pcm


def write_wav(pcm):
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        signal = signal.reshape(-1, channels)

    return signal, rate


def generate_protocol():
    full_signal, pcm = build_signal()
    write_wav(pcm)

    total_duration = full_signal.size / sample_rate

    print(f"Wrote: {output_path}")
    print(f"Total duration: {total_duration:.2f} s ({total_duration / 60:.2f} min)")

    return full_signal


def play_protocol(signal=None, rate=sample_rate):
    if signal is None:
        signal, rate = read_wav(output_path)

    print("Playing pulse protocol...")
    print("PLAYBACK_STARTED", flush=True)

    try:
        sd.play(signal, samplerate=rate, blocking=True)
        sd.stop()
        print("Pulse protocol playback finished.")

    except KeyboardInterrupt:
        sd.stop()
        print("Pulse protocol playback stopped.")

    except Exception as e:
        sd.stop()
        print(f"Pulse protocol playback failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate and play pulse protocol audio.")
    parser.add_argument(
        "--mode",
        choices=["generate-and-play", "generate-only", "play-existing"],
        default="generate-and-play"
    )
    args = parser.parse_args()

    if args.mode == "generate-only":
        generate_protocol()
        return

    if args.mode == "play-existing":
        play_protocol()
        return

    signal = generate_protocol()
    play_protocol(signal)


if __name__ == "__main__":
    main()
