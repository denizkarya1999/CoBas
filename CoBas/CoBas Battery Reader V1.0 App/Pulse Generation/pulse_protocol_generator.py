"""Generate, play, and optionally record two-second CoBas chirp pulses."""

import argparse
import os
import queue
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48_000
PULSE_DURATION_SECONDS = 2.0
START_FREQUENCY = 15_000.0
END_FREQUENCY = 19_200.0
AMPLITUDE = 0.90  # Keep 0.9 dB of digital headroom to prevent fuzzy clipping.
FADE_MILLISECONDS = 5.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "Inputs")
OUTPUT_PATH = os.path.join(INPUT_DIR, "2_second_pulse.wav")


def apply_fade(signal, fade_milliseconds=FADE_MILLISECONDS):
    """Apply smooth raised-cosine edges without adding broadband clicks."""
    fade_samples = int(
        round(fade_milliseconds * 1e-3 * SAMPLE_RATE)
    )

    if fade_samples == 0 or 2 * fade_samples >= signal.size:
        return signal

    fade_phase = np.linspace(
        0.0,
        np.pi / 2.0,
        fade_samples,
        dtype=np.float64,
    )
    fade_curve = np.square(np.sin(fade_phase)).astype(signal.dtype)
    window = np.ones_like(signal)
    window[:fade_samples] = fade_curve
    window[-fade_samples:] = fade_curve[::-1]
    return signal * window


def build_pulse():
    """Build a clean, linear chirp lasting exactly two seconds."""
    sample_count = int(round(SAMPLE_RATE * PULSE_DURATION_SECONDS))
    # Calculate phase in float64. At ultrasonic frequencies, float32 phase
    # quantization adds avoidable jitter and weak spurious frequency content.
    time_axis = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    sweep_rate = (
        END_FREQUENCY - START_FREQUENCY
    ) / PULSE_DURATION_SECONDS
    phase = 2.0 * np.pi * (
        START_FREQUENCY * time_axis
        + 0.5 * sweep_rate * time_axis * time_axis
    )
    signal = (AMPLITUDE * np.sin(phase)).astype(np.float32)
    return apply_fade(signal)


def write_wav(path, signal):
    """Write a floating-point mono signal as a 16-bit WAV."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    pcm = np.rint(
        np.clip(signal, -1.0, 1.0) * 32767.0
    ).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def read_wav(path):
    """Read a mono 16-bit WAV into a floating-point signal."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2:
        raise ValueError("The pulse WAV must be mono and 16-bit.")

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return signal / 32768.0, rate


def generate_pulse():
    """Generate and save a fresh copy of the two-second pulse."""
    signal = build_pulse()
    write_wav(OUTPUT_PATH, signal)
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Pulse duration: {signal.size / SAMPLE_RATE:.2f} seconds")
    return signal


def play_pulse(signal=None, rate=SAMPLE_RATE):
    """Play one pulse without opening an input stream."""
    if signal is None:
        signal, rate = read_wav(OUTPUT_PATH)

    try:
        playback_started_at = time.time()
        sd.play(signal, samplerate=rate, blocking=False)
        print(
            f"PLAYBACK_STARTED {playback_started_at:.9f}",
            flush=True,
        )
        sd.wait()
        print("PULSE_FINISHED", flush=True)
    finally:
        sd.stop()


def play_and_record_pulse(
    signal,
    recording_path,
    input_device=None,
):
    """
    Play and record one pulse through a single duplex stream.

    sounddevice creates the input and output streams together, records exactly
    the number of samples in the pulse, and closes both when playback finishes.
    """
    playback = np.asarray(signal, dtype=np.float32).reshape(-1, 1)

    try:
        playback_started_at = time.time()
        recording = sd.playrec(
            playback,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=(input_device, None),
            blocking=False,
        )
        print(
            f"PLAYBACK_STARTED {playback_started_at:.9f}",
            flush=True,
        )
        sd.wait()
        write_wav(recording_path, recording.reshape(-1))
        print(f"PULSE_FINISHED {recording_path}", flush=True)
    finally:
        sd.stop()


def get_sequence_recording_paths(recording_template, pulse_count):
    """Expand a ``{pulse}`` template into one unique path per pulse."""
    if pulse_count < 1:
        raise ValueError("Pulse count must be greater than zero.")

    try:
        paths = [
            os.path.abspath(recording_template.format(pulse=pulse_number))
            for pulse_number in range(1, pulse_count + 1)
        ]
    except (AttributeError, KeyError, ValueError) as exc:
        raise ValueError(
            "Recording template must contain a valid {pulse} field."
        ) from exc

    if len(set(paths)) != pulse_count:
        raise ValueError(
            "Recording template must produce a unique path for every pulse."
        )

    return paths


def _write_float_frames(wav_file, frames):
    """Append floating-point microphone frames to a 16-bit WAV."""
    pcm = np.rint(
        np.clip(frames, -1.0, 1.0) * 32767.0
    ).astype(np.int16)
    wav_file.writeframesraw(pcm.tobytes())


def play_and_record_pulse_sequence(
    signal,
    recording_template,
    pulse_count,
    input_device=None,
    wait_for_start=False,
):
    """
    Play back-to-back pulses through one stream and save one WAV per pulse.

    Keeping a single full-duplex stream open removes the process and device
    setup gaps that would otherwise extend an N-pulse sensor session beyond
    exactly N times the two-second pulse duration.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("The pulse signal cannot be empty.")

    recording_paths = get_sequence_recording_paths(
        recording_template,
        pulse_count,
    )
    pulse_samples = signal.size
    total_samples = pulse_samples * pulse_count
    audio_queue = queue.Queue(maxsize=512)
    stream_finished = threading.Event()
    state = {
        "position": 0,
        "started_at": None,
        "error": None,
        "status": None,
    }

    def callback(indata, outdata, frames, _time_info, status):
        if status:
            state["status"] = str(status)

        start_sample = state["position"]
        valid_frames = min(frames, total_samples - start_sample)
        outdata.fill(0)

        if valid_frames <= 0:
            raise sd.CallbackStop

        if state["started_at"] is None:
            state["started_at"] = time.time()

        output_offset = 0
        sequence_position = start_sample
        while output_offset < valid_frames:
            pulse_offset = sequence_position % pulse_samples
            copy_count = min(
                valid_frames - output_offset,
                pulse_samples - pulse_offset,
            )
            outdata[
                output_offset:output_offset + copy_count,
                0,
            ] = signal[pulse_offset:pulse_offset + copy_count]
            output_offset += copy_count
            sequence_position += copy_count

        try:
            audio_queue.put_nowait(
                (
                    start_sample,
                    indata[:valid_frames, 0].copy(),
                )
            )
        except queue.Full:
            state["error"] = RuntimeError(
                "Microphone recording queue could not keep up."
            )
            raise sd.CallbackAbort

        state["position"] = start_sample + valid_frames
        if state["position"] >= total_samples:
            raise sd.CallbackStop

    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=1024,
        dtype="float32",
        channels=(1, 1),
        device=(input_device, None),
        latency="high",
        callback=callback,
        finished_callback=stream_finished.set,
    )
    current_wav = None
    completed_normally = False

    try:
        print(
            f"SEQUENCE_READY {pulse_count} "
            f"{pulse_count * PULSE_DURATION_SECONDS:.3f}",
            flush=True,
        )
        if wait_for_start:
            start_message = sys.stdin.readline()
            if not start_message:
                raise RuntimeError(
                    "Pulse sequence start signal was not received."
                )

        stream.start()
        recorded_position = 0

        while recorded_position < total_samples:
            if state["error"] is not None:
                raise state["error"]

            try:
                chunk_start, recorded_frames = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if stream_finished.is_set():
                    break
                continue

            if chunk_start != recorded_position:
                raise RuntimeError(
                    "Microphone frames were received out of sequence."
                )

            frame_offset = 0
            while frame_offset < recorded_frames.size:
                pulse_index = recorded_position // pulse_samples
                pulse_offset = recorded_position % pulse_samples

                if current_wav is None:
                    recording_path = recording_paths[pulse_index]
                    os.makedirs(
                        os.path.dirname(recording_path),
                        exist_ok=True,
                    )
                    current_wav = wave.open(recording_path, "wb")
                    current_wav.setnchannels(1)
                    current_wav.setsampwidth(2)
                    current_wav.setframerate(SAMPLE_RATE)
                    playback_started_at = (
                        state["started_at"]
                        + pulse_index * PULSE_DURATION_SECONDS
                    )
                    print(
                        f"PLAYBACK_STARTED {pulse_index + 1} "
                        f"{playback_started_at:.9f}",
                        flush=True,
                    )

                write_count = min(
                    recorded_frames.size - frame_offset,
                    pulse_samples - pulse_offset,
                )
                _write_float_frames(
                    current_wav,
                    recorded_frames[
                        frame_offset:frame_offset + write_count
                    ],
                )
                frame_offset += write_count
                recorded_position += write_count

                if recorded_position % pulse_samples == 0:
                    finished_pulse = recorded_position // pulse_samples
                    current_wav.close()
                    current_wav = None
                    print(
                        f"PULSE_FINISHED {finished_pulse} "
                        f"{recording_paths[finished_pulse - 1]}",
                        flush=True,
                    )

        if recorded_position != total_samples:
            raise RuntimeError(
                "Pulse sequence ended before all microphone frames were saved."
            )

        if not stream_finished.wait(timeout=5):
            raise RuntimeError("Pulse audio stream did not finish cleanly.")

        if state["status"]:
            print(f"AUDIO_WARNING {state['status']}", flush=True)

        completed_normally = True

    finally:
        if current_wav is not None:
            current_wav.close()

        try:
            if stream.active:
                if completed_normally:
                    stream.stop()
                else:
                    stream.abort()
        finally:
            stream.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and use two-second CoBas chirp pulses."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "generate-only",
            "play-existing",
            "generate-and-play",
            "generate-play-record",
        ],
        default="generate-and-play",
    )
    parser.add_argument(
        "--record-output",
        help="WAV path for microphone audio captured during the pulse.",
    )
    parser.add_argument(
        "--record-output-template",
        help=(
            "WAV path template containing {pulse}, used to save one "
            "recording per pulse."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of consecutive two-second pulses to play.",
    )
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="Prepare the audio stream, then wait for a line on stdin.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice input-device index; defaults to the system input.",
    )
    args = parser.parse_args()

    if args.mode == "generate-only":
        generate_pulse()
        return

    if args.mode == "play-existing":
        play_pulse()
        return

    signal = generate_pulse()

    if args.mode == "generate-play-record":
        if args.count < 1:
            parser.error("--count must be greater than zero")

        if args.record_output_template:
            try:
                play_and_record_pulse_sequence(
                    signal,
                    args.record_output_template,
                    args.count,
                    input_device=args.input_device,
                    wait_for_start=args.wait_for_start,
                )
            except ValueError as exc:
                parser.error(str(exc))
            return

        if args.count != 1:
            parser.error(
                "--record-output-template is required when --count is "
                "greater than one"
            )
        if not args.record_output:
            parser.error(
                "--record-output or --record-output-template is required "
                "for generate-play-record"
            )
        play_and_record_pulse(
            signal,
            args.record_output,
            input_device=args.input_device,
        )
        return

    play_pulse(signal)


if __name__ == "__main__":
    main()
