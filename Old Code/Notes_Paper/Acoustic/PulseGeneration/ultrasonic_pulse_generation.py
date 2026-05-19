import numpy as np
import wave
import time

# ------------------------------
# SETTINGS
# ------------------------------
sample_rate    = 48000        # Prefer 48 kHz near ~19 kHz
pulse_duration = 0.10         # 100 ms pulse duration
gap_duration   = 0.05         # 50 ms gap
start_freq     = 15000        # Hz
end_freq       = 19200        # Hz
amplitude      = 0.85         #
fade_ms        = 5.0          # 5 ms fade in/out
cycles_total   = 31           # <-- Total number of repetitions
active_secs    = 60.0         # <-- 60s of pulses per cycle
pause_secs     = 1.0          # <-- 1s of silence AFTER EACH block

output_file =  str(cycles_total) + "CyclesUltrasonicWave.wav"

t0 = time.perf_counter()

# ------------------------------
# BUILD ONE PULSE (linear chirp) AS float32
# ------------------------------
Ns_pulse = int(round(sample_rate * pulse_duration))
t_pulse  = (np.arange(Ns_pulse, dtype=np.float32) / np.float32(sample_rate))

k = np.float32((end_freq - start_freq) / pulse_duration)  # Hz/s
phase = 2.0 * np.pi * (start_freq * t_pulse + 0.5 * k * t_pulse * t_pulse)
pulse = (amplitude * np.sin(phase)).astype(np.float32)

# Fade-in/out
fade_samples = int(round(fade_ms * 1e-3 * sample_rate))
if fade_samples > 0 and 2 * fade_samples < Ns_pulse:
    window = np.ones_like(pulse)
    window[:fade_samples]  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    window[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    pulse *= window

# Gap (silence)
Ns_gap = int(round(sample_rate * gap_duration))
gap = np.zeros(Ns_gap, dtype=np.float32)

# One small cycle = pulse + gap
cycle = np.concatenate([pulse, gap])               # float32
Ns_cycle = cycle.size                              # at 48 kHz, 0.05s+0.10s => 2400+4800=7200 samples

# Convert that small cycle ONCE to int16
cycle_i16 = np.rint(np.clip(cycle, -1.0, 1.0) * 32767.0).astype(np.int16)
cycle_bytes = cycle_i16.tobytes()

# For each 60 s active block, how many small cycles fit?
Ns_active     = int(round(active_secs * sample_rate))
cycles_in_60  = Ns_active // Ns_cycle                # with 0.15 s cycles, this is exactly 400
residual_samp = Ns_active - cycles_in_60 * Ns_cycle  # should be 0 for 0.15 s cycles

# 3 s pause bytes
Ns_pause = int(round(pause_secs * sample_rate))
pause_bytes = (b'\x00\x00') * Ns_pause

t1 = time.perf_counter()

# ------------------------------
# STREAM TO WAV: 50 × (60 s active + 3 s silence)
# ------------------------------
with wave.open(output_file, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)        # 16-bit PCM
    wf.setframerate(sample_rate)

    # cycles_in_60 is small (likely 400), so one write is fine.
    active_block_bytes = cycle_bytes * cycles_in_60

    for _ in range(cycles_total):
        # 60 s active section
        wf.writeframes(active_block_bytes)

        # If active_secs isn't an exact multiple of pulse+gap, write the remainder
        if residual_samp:
            wf.writeframes(cycle_i16[:residual_samp].tobytes())

        # 3 s pause AFTER each active block
        wf.writeframes(pause_bytes)

t2 = time.perf_counter()

total_seconds = (cycles_total * (active_secs + pause_secs))
print(f"Wrote: {output_file}")
print(f"Intended total duration: {total_seconds:.2f} s ({total_seconds/60:.2f} min)")
print(f"build_small_cycle: {t1 - t0:.3f}s, stream_write: {t2 - t1:.3f}s, total: {t2 - t0:.3f}s")
