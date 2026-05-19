import numpy as np
import wave

# ==========================================================
# USER SETTINGS
# ==========================================================
sample_rate = 48000

# --- Alignment structure ---
initial_silence_sec = 30.0
beacon_freq           = 10000     # Hz (outside chirp band)
beacon_duration_sec   = 2.0
guard_silence_sec    = 1.5
tail_silence_sec     = 5.0

# --- Chirp pulse train ---
pulse_duration = 0.10
gap_duration   = 0.05
start_freq     = 15000.0
end_freq       = 19200.0
amplitude      = 0.85
fade_ms        = 5.0

cycles_total   = 5
active_secs    = 60.0

output_file = f"{cycles_total}_15sPause_BeaconProtocol.wav"

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
    w[:fade_samp]  = np.linspace(0.0, 1.0, fade_samp, dtype=np.float32)
    w[-fade_samp:] = np.linspace(1.0, 0.0, fade_samp, dtype=np.float32)
    return x * w

# ==========================================================
# BUILD BEGIN / END MARKERS
# ==========================================================
beacon_tone = apply_fade(tone(beacon_freq, beacon_duration_sec), fade_ms)

# ==========================================================
# BUILD ONE CHIRP PULSE
# ==========================================================
Ns_pulse = int(round(sample_rate * pulse_duration))
t = np.arange(Ns_pulse, dtype=np.float32) / sample_rate

k = (end_freq - start_freq) / pulse_duration
phase = 2.0 * np.pi * (start_freq * t + 0.5 * k * t * t)
pulse = (amplitude * np.sin(phase)).astype(np.float32)
pulse = apply_fade(pulse, fade_ms)

gap = silence(gap_duration)
small_cycle = np.concatenate([pulse, gap])

# ==========================================================
# BUILD ACTIVE CHIRP BLOCK
# ==========================================================
Ns_active = int(round(active_secs * sample_rate))
cycles_in_block = Ns_active // small_cycle.size
residual = Ns_active - cycles_in_block * small_cycle.size

active_block = np.concatenate(
    [small_cycle] * cycles_in_block +
    ([small_cycle[:residual]] if residual else [])
)

# Repeat active block cycles_total times
chirps = np.concatenate([active_block] * cycles_total)

# ==========================================================
# ASSEMBLE FULL PROTOCOL
# ==========================================================
full_signal = np.concatenate([
    silence(initial_silence_sec),
    beacon_tone,
    silence(guard_silence_sec),
    chirps,
    silence(guard_silence_sec),
    beacon_tone,
    silence(tail_silence_sec)
])

# ==========================================================
# WRITE WAV (INT16 PCM)
# ==========================================================
pcm = np.rint(np.clip(full_signal, -1.0, 1.0) * 32767.0).astype(np.int16)

with wave.open(output_file, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm.tobytes())

# ==========================================================
# REPORT
# ==========================================================
total_duration = full_signal.size / sample_rate
print(f"Wrote: {output_file}")
print(f"Total duration: {total_duration:.2f} s ({total_duration/60:.2f} min)")
