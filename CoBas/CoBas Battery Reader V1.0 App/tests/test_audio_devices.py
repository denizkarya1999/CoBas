"""Hardware-free USB audio selection and exact-duration playback checks."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np


APP = Path(__file__).resolve().parents[1]


class PortAudioError(Exception):
    pass


class CallbackStop(Exception):
    pass


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, APP / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


# Import without initializing a real audio library or querying hardware.
with patch.dict(sys.modules, {"sounddevice": SimpleNamespace(PortAudioError=PortAudioError)}):
    audio = load_module("audio_devices_under_test", "AudioDevices.py")
    with patch.dict(sys.modules, {"AudioDevices": audio}):
        pulse = load_module("pulse_under_test", "Pulse Generation/pulse_protocol_generator.py")


def device(name, inputs=0, outputs=0):
    return {"name": name, "max_input_channels": inputs, "max_output_channels": outputs}


class AudioDeviceTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            device("vc4-hdmi-1: MAI PCM (hw:1,0)", outputs=2),
            device("USB Microphone: Audio (hw:2,0)", inputs=2),
            device("UACDemoV1.0: USB Audio (hw:3,0)", outputs=2),
        ]
        self.cards = {
            1: {"id": "vc4hdmi1", "is_usb": False},
            2: {"id": "Microphone", "is_usb": True},
            3: {"id": "UACDemoV10", "is_usb": True},
        }
        self.sd = SimpleNamespace(
            query_devices=MagicMock(side_effect=lambda: self.devices),
            check_input_settings=MagicMock(),
            check_output_settings=MagicMock(),
            default=SimpleNamespace(device=[1, 0]),
            PortAudioError=PortAudioError,
            CallbackStop=CallbackStop,
            CallbackAbort=RuntimeError,
            play=MagicMock(), playrec=MagicMock(), wait=MagicMock(), stop=MagicMock(),
        )
        self.enterContext(patch.object(audio, "sd", self.sd))
        self.enterContext(patch.object(audio, "_alsa_cards", side_effect=lambda: self.cards))
        self.enterContext(patch.object(pulse, "sd", self.sd))
        self.sleep = self.enterContext(patch.object(pulse.time, "sleep"))
        self.calls = MagicMock()
        for name in ("play", "wait", "stop", "playrec"):
            self.calls.attach_mock(getattr(self.sd, name), name)
        self.calls.attach_mock(self.sleep, "sleep")

    def test_auto_uses_separate_usb_mic_and_speaker_not_default_hdmi(self):
        output = audio.resolve_audio_device("output")
        microphone = audio.resolve_audio_device("input")
        self.assertEqual((microphone.index, output.index), (1, 2))
        self.assertEqual((microphone.channels, output.channels), (1, 2))
        self.assertEqual(output.selector, "alsa:UACDemoV10:0")
        self.sd.check_output_settings.assert_called_once_with(
            device=2, channels=2, samplerate=48000, dtype="float32"
        )

    def test_saved_selector_survives_usb_card_and_device_reordering(self):
        selector = audio.resolve_audio_device("output").selector
        self.devices[:] = [
            device("UACDemoV1.0: USB Audio (hw:4,0)", outputs=2),
            device("vc4-hdmi-1: MAI PCM (hw:1,0)", outputs=2),
            device("USB Microphone: Audio (hw:0,0)", inputs=2),
        ]
        self.cards = {
            0: {"id": "Microphone", "is_usb": True},
            1: {"id": "vc4hdmi1", "is_usb": False},
            4: {"id": "UACDemoV10", "is_usb": True},
        }
        self.assertEqual(audio.resolve_audio_device("output", selector).index, 0)
        self.assertEqual(audio.resolve_audio_device("input", "alsa:Microphone:0").index, 2)

    def test_name_fallback_ignores_volatile_hardware_card_number(self):
        self.cards = {}
        selector = audio.resolve_audio_device("output").selector
        self.assertEqual(selector, "name:UACDemoV1.0: USB Audio")
        self.devices[2]["name"] = "UACDemoV1.0: USB Audio (hw:8,0)"
        self.assertEqual(audio.resolve_audio_device("output", selector).index, 2)

    def test_missing_explicit_device_does_not_switch_to_other_speaker(self):
        with self.assertRaisesRegex(audio.AudioDeviceError, "not connected"):
            audio.resolve_audio_device("output", "alsa:Missing:0")
        self.sd.check_output_settings.assert_not_called()

    def test_legacy_numeric_selector_still_works(self):
        self.assertEqual(audio.resolve_audio_device("input", "1").index, 1)

    def test_multiple_usb_speakers_require_selection(self):
        self.devices.append(device("Other USB Speaker", outputs=2))
        with self.assertRaisesRegex(audio.AudioDeviceError, "Multiple compatible USB"):
            audio.resolve_audio_device("output")

    def test_multiple_usb_speakers_can_use_suitable_system_default(self):
        self.devices.append(device("Other USB Speaker", outputs=2))
        self.sd.default.device = [1, 3]
        self.assertEqual(audio.resolve_audio_device("output").index, 3)

    def test_no_usb_uses_compatible_default(self):
        self.devices[:] = [device("Built-in Audio", inputs=1, outputs=2)]
        self.sd.default.device = [0, 0]
        self.assertEqual(audio.resolve_audio_device("output").index, 0)
        self.assertEqual(audio.resolve_audio_device("input").index, 0)

    def test_no_usb_or_valid_default_is_clear_error(self):
        self.devices[:] = [device("Built-in Audio", outputs=2)]
        self.sd.default.device = [-1, -1]
        with self.assertRaisesRegex(audio.AudioDeviceError, "No USB output"):
            audio.resolve_audio_device("output")

    def test_capture_only_usb_mic_is_never_selected_as_speaker(self):
        self.devices[:] = [device("USB Microphone", inputs=2)]
        self.sd.default.device = [0, 0]
        with self.assertRaisesRegex(audio.AudioDeviceError, "No USB output"):
            audio.resolve_audio_device("output")

    def test_unsupported_usb_rate_does_not_fall_back_to_hdmi(self):
        self.sd.check_output_settings.side_effect = PortAudioError("48 kHz unsupported")
        with self.assertRaisesRegex(audio.AudioDeviceError, "No compatible USB output"):
            audio.resolve_audio_device("output")
        self.assertTrue(all(c.kwargs["device"] == 2 for c in self.sd.check_output_settings.call_args_list))

    def test_stereo_capture_is_supported_when_mono_not_available(self):
        self.sd.check_input_settings.side_effect = [PortAudioError("stereo required"), None]
        self.assertEqual(audio.resolve_audio_device("input").channels, 2)

    def test_mono_speaker_is_supported(self):
        self.devices[2]["max_output_channels"] = 1
        self.assertEqual(audio.resolve_audio_device("output").channels, 1)

    def test_dropdown_selectors_are_stable_and_direction_specific(self):
        self.assertEqual(audio.list_audio_choices("input", fresh=False), [
            {"label": "USB Microphone: Audio (hw:2,0)", "selector": "alsa:Microphone:0"}
        ])
        self.assertEqual(len(audio.list_audio_choices("output", fresh=False)), 2)

    def test_dropdown_refresh_uses_fresh_process_without_resetting_audio(self):
        choices = [{"label": "Speaker", "selector": "alsa:Speaker:0"}]
        result = SimpleNamespace(stdout=json.dumps(choices))
        with patch.object(audio.subprocess, "run", return_value=result) as run:
            self.assertEqual(audio.list_audio_choices("output"), choices)
        self.assertEqual(run.call_args.args[0][-2:], ["--list", "output"])
        self.sd.query_devices.assert_not_called()

    def test_refresh_timeout_reports_actionable_error(self):
        with patch.object(audio.subprocess, "run", side_effect=subprocess.TimeoutExpired("query", 10)):
            with self.assertRaisesRegex(audio.AudioDeviceError, "Could not refresh"):
                audio.list_audio_choices("output")

    def test_playback_has_full_duration_rising_audible_chirp_without_mutating_reference(self):
        signal = pulse.build_pulse()
        reference = signal.copy()
        playback = pulse.build_playback_pulse(signal)
        self.assertEqual(playback.shape, (96000,))
        self.assertEqual(playback.dtype, np.float32)
        self.assertTrue(np.isfinite(playback).all())
        self.assertLessEqual(float(np.max(np.abs(playback))), 0.95)
        self.assertEqual((playback[0], playback[-1]), (0, 0))
        np.testing.assert_array_equal(signal, reference)
        self.assertEqual((pulse.START_FREQUENCY, pulse.END_FREQUENCY, pulse.AMPLITUDE), (15000, 19200, .95))
        audible = playback - reference * (0.65 / 0.95)
        # Check the start, middle, and end: a short cue, beep, or reversed sweep fails.
        for center_seconds, expected_hz in ((.1, 1150), (1.0, 2500), (1.9, 3850)):
            with self.subTest(center_seconds=center_seconds):
                center = round(center_seconds * 48000)
                segment = audible[center - 480:center + 480]
                spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
                peak = np.fft.rfftfreq(segment.size, 1 / 48000)[np.argmax(spectrum)]
                self.assertAlmostEqual(peak, expected_hz, delta=60)
                self.assertAlmostEqual(float(np.sqrt(np.mean(segment ** 2))), .30 / np.sqrt(2), delta=.005)

    def test_playback_uses_requested_rate_for_pitch_and_ten_millisecond_fades(self):
        rate = 44100
        audible = pulse.build_playback_pulse(np.zeros(2 * rate, dtype=np.float32), rate=rate)
        self.assertEqual(audible.shape, (88200,))
        self.assertEqual((audible[0], audible[-1]), (0, 0))
        center = rate
        segment = audible[center - 441:center + 441]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
        peak = np.fft.rfftfreq(segment.size, 1 / rate)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 2500, delta=50)
        # Halfway through the 10 ms fade, a raised-cosine window is half level.
        halfway = 220
        for index in (halfway, len(audible) - halfway - 1):
            time_seconds = index / rate
            raw_sample = .30 * np.sin(2 * np.pi * (1000 * time_seconds + 750 * time_seconds ** 2))
            self.assertAlmostEqual(float(audible[index]), raw_sample * .5, delta=1e-6)

    def test_playback_leaves_headroom_for_full_scale_input(self):
        for level in (-1.0, 1.0):
            with self.subTest(level=level):
                signal = np.full(96000, level, dtype=np.float32)
                playback = pulse.build_playback_pulse(signal)
                self.assertLess(float(np.max(np.abs(playback))), 1.0)
                np.testing.assert_array_equal(signal, np.full(96000, level, dtype=np.float32))

    def test_failed_capture_stops_audio_without_writing_a_recording(self):
        self.sd.wait.side_effect = PortAudioError("speaker disconnected")
        with patch.object(pulse, "write_wav") as write_wav:
            with self.assertRaisesRegex(PortAudioError, "speaker disconnected"):
                pulse.play_and_record_pulse(pulse.build_pulse(), "unused.wav")
        write_wav.assert_not_called()
        self.sd.stop.assert_called_once()
        self.sd.playrec.assert_called_once()
        self.sd.play.assert_not_called()
        self.sleep.assert_not_called()

    def test_generate_only_does_not_play_audio(self):
        with patch.object(sys, "argv", ["pulse", "--mode", "generate-only"]), \
                patch.object(pulse, "generate_pulse"):
            pulse.main()
        self.sd.play.assert_not_called()
        self.sd.playrec.assert_not_called()

    def test_play_duplicates_exact_two_second_chirp_to_stereo(self):
        signal = pulse.build_pulse()
        pulse.play_pulse(signal)
        args, kwargs = self.sd.play.call_args
        self.assertEqual(args[0].shape, (96000, 2))
        expected = pulse.build_playback_pulse(signal)
        np.testing.assert_array_equal(args[0][:, 0], expected)
        np.testing.assert_array_equal(args[0][:, 1], expected)
        self.assertEqual(kwargs["samplerate"], 48000)
        self.assertEqual(kwargs["device"], 2)
        self.sd.stop.assert_called_once()
        self.sd.play.assert_called_once()
        self.assertEqual([c[0] for c in self.calls.mock_calls],
                         ["play", "wait", "stop"])
        self.sleep.assert_not_called()

    def test_play_uses_full_duration_audible_mix_at_44100_hz(self):
        signal = np.zeros(88200, dtype=np.float32)
        pulse.play_pulse(signal, rate=44100)
        args, kwargs = self.sd.play.call_args
        self.assertEqual(kwargs["samplerate"], 44100)
        self.assertEqual(args[0].shape, (88200, 2))
        np.testing.assert_array_equal(args[0][:, 0], pulse.build_playback_pulse(signal, rate=44100))
        self.sd.play.assert_called_once()

    def test_duplex_uses_separate_devices_and_writes_mono_from_stereo_capture(self):
        self.sd.check_input_settings.side_effect = [PortAudioError("stereo required"), None]
        self.sd.playrec.return_value = np.column_stack((np.full(96000, .125), np.full(96000, .875)))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "recorded.wav")
            pulse.play_and_record_pulse(pulse.build_pulse(), path)
            with wave.open(path, "rb") as recorded:
                self.assertEqual((recorded.getnchannels(), recorded.getframerate(), recorded.getnframes()), (1, 48000, 96000))
                samples = np.frombuffer(recorded.readframes(96000), dtype=np.int16)
                np.testing.assert_array_equal(samples, np.full(96000, 4096))
        args, kwargs = self.sd.playrec.call_args
        self.assertEqual(args[0].shape, (96000, 2))
        expected = pulse.build_playback_pulse(pulse.build_pulse())
        np.testing.assert_array_equal(args[0][:, 0], expected)
        np.testing.assert_array_equal(args[0][:, 1], expected)
        self.assertEqual(kwargs["device"], (1, 2))
        self.assertEqual(kwargs["channels"], 2)
        self.assertEqual([c[0] for c in self.calls.mock_calls],
                         ["playrec", "wait", "stop"])
        self.sd.play.assert_not_called()
        self.sleep.assert_not_called()

    def test_sequence_preserves_stereo_pulses_and_exact_mono_recordings(self):
        captured = {}
        calls = self.calls
        output_log = io.StringIO()

        class FakeStream:
            active = False

            def __init__(self, **kwargs):
                calls.stream_created()
                self_before = [c[0] for c in calls.mock_calls]
                assert self_before == ["stream_created"]
                assert "SEQUENCE_READY" not in output_log.getvalue()
                captured.update(kwargs)
                captured["output"] = []
                self.kwargs = kwargs

            def start(self):
                calls.stream_started()
                assert "SEQUENCE_READY 2 4.000" in output_log.getvalue()
                self.active = True
                while True:
                    indata = np.full((1024, self.kwargs["channels"][0]), .125, dtype=np.float32)
                    outdata = np.zeros((1024, self.kwargs["channels"][1]), dtype=np.float32)
                    try:
                        self.kwargs["callback"](indata, outdata, 1024, None, None)
                    except CallbackStop:
                        captured["output"].append(outdata.copy())
                        self.active = False
                        self.kwargs["finished_callback"]()
                        break
                    captured["output"].append(outdata.copy())

            def close(self):
                captured["closed"] = True

        self.sd.Stream = FakeStream
        signal = pulse.build_pulse()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "stdin", io.StringIO("START\n")), patch.object(sys, "stdout", output_log), \
                    patch.object(pulse, "build_playback_pulse", wraps=pulse.build_playback_pulse) as mix:
                pulse.play_and_record_pulse_sequence(signal, str(Path(directory) / "pulse_{pulse}.wav"), 2,
                                                    wait_for_start=True)
            mix.assert_called_once()
            for number in (1, 2):
                with wave.open(str(Path(directory) / f"pulse_{number}.wav"), "rb") as recorded:
                    self.assertEqual((recorded.getnchannels(), recorded.getframerate(), recorded.getnframes()), (1, 48000, 96000))
        self.sd.play.assert_not_called()
        self.sd.playrec.assert_not_called()
        self.sleep.assert_not_called()
        self.assertEqual([c[0] for c in calls.mock_calls],
                         ["stream_created", "stream_started"])
        self.assertEqual(captured["device"], (1, 2))
        self.assertEqual(captured["channels"], (1, 2))
        self.assertTrue(captured["closed"])
        playback = np.concatenate(captured["output"])
        expected = np.tile(pulse.build_playback_pulse(signal), 2)
        np.testing.assert_array_equal(playback[:192000, 0], expected)
        np.testing.assert_array_equal(playback[:192000, 1], expected)
        self.assertTrue(np.all(playback[192000:] == 0))

    def test_cli_accepts_stable_device_selectors(self):
        arguments = ["pulse", "--mode", "generate-play-record", "--record-output", "voice.wav",
                     "--input-device", "alsa:Microphone:0", "--output-device", "alsa:UACDemoV10:0"]
        with patch.object(sys, "argv", arguments), patch.object(pulse, "generate_pulse", return_value=np.zeros(8)), \
                patch.object(pulse, "play_and_record_pulse") as playrec:
            pulse.main()
        self.assertEqual(playrec.call_args.kwargs["input_device"], "alsa:Microphone:0")
        self.assertEqual(playrec.call_args.kwargs["output_device"], "alsa:UACDemoV10:0")

    def test_sequence_records_all_200_chirps_in_one_continuous_stream(self):
        pulse_count = 200
        signal = pulse.build_pulse()
        playback = pulse.build_playback_pulse(signal)
        pulse_samples = 96000
        total_samples = pulse_count * pulse_samples
        output_log = io.StringIO()
        captured = {"samples": 0, "closed": False, "padding": 0}
        test = self

        def microphone_pcm(positions):
            # A continuous non-chirp-aligned pattern exposes dropped, duplicated,
            # or misplaced samples at callback and recording-file boundaries.
            return (positions % 30001 - 15000).astype(np.int16)

        class FakeStream:
            active = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                test.assertIn("SEQUENCE_READY 200 400.000", output_log.getvalue())
                self.active = True
                # Fewer than 512 callbacks fit the recording queue while this
                # synchronous fake runs. Boundaries deliberately cross chirps.
                frames = 65537
                while self.active:
                    start = captured["samples"]
                    positions = np.arange(start, start + frames)
                    indata = (microphone_pcm(positions).astype(np.float32) / 32767)[:, None]
                    outdata = np.empty((frames, self.kwargs["channels"][1]), dtype=np.float32)
                    stopped = False
                    try:
                        self.kwargs["callback"](indata, outdata, frames, None, None)
                    except CallbackStop:
                        stopped = True
                    valid_frames = min(frames, total_samples - start)
                    expected = playback[positions[:valid_frames] % pulse_samples]
                    for channel in range(outdata.shape[1]):
                        np.testing.assert_array_equal(outdata[:valid_frames, channel], expected)
                    np.testing.assert_array_equal(outdata[valid_frames:], 0)
                    captured["samples"] += valid_frames
                    test.assertEqual(stopped, captured["samples"] == total_samples)
                    if stopped:
                        test.assertEqual(captured["samples"], total_samples)
                        captured["padding"] = frames - valid_frames
                        self.active = False
                        self.kwargs["finished_callback"]()

            def close(self):
                captured["closed"] = True

        self.sd.Stream = MagicMock(side_effect=FakeStream)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "stdout", output_log):
                pulse.play_and_record_pulse_sequence(
                    signal, str(Path(directory) / "pulse_{pulse}.wav"), pulse_count
                )
            self.assertEqual(len(list(Path(directory).glob("*.wav"))), pulse_count)
            for number in range(1, pulse_count + 1):
                with self.subTest(pulse=number):
                    with wave.open(str(Path(directory) / f"pulse_{number}.wav"), "rb") as recorded:
                        self.assertEqual(
                            (recorded.getnchannels(), recorded.getsampwidth(),
                             recorded.getframerate(), recorded.getnframes()),
                            (1, 2, 48000, pulse_samples),
                        )
                        samples = np.frombuffer(recorded.readframes(pulse_samples), dtype=np.int16)
                    positions = np.arange((number - 1) * pulse_samples, number * pulse_samples)
                    np.testing.assert_array_equal(samples, microphone_pcm(positions))
            events = [line.split() for line in output_log.getvalue().splitlines()]
            started = [event for event in events if event[0] == "PLAYBACK_STARTED"]
            finished = [event for event in events if event[0] == "PULSE_FINISHED"]
            expected_numbers = list(range(1, pulse_count + 1))
            self.assertEqual([int(event[1]) for event in started], expected_numbers)
            self.assertEqual([int(event[1]) for event in finished], expected_numbers)
            for number, event in enumerate(finished, start=1):
                self.assertEqual(event[2], str(Path(directory) / f"pulse_{number}.wav"))
            start_time = float(started[0][2])
            for index, event in enumerate(started):
                self.assertAlmostEqual(float(event[2]) - start_time, index * 2)
        self.sd.Stream.assert_called_once()
        self.assertEqual(self.sd.Stream.call_args.kwargs["device"], (1, 2))
        self.assertEqual(self.sd.Stream.call_args.kwargs["channels"], (1, 2))
        self.assertEqual(captured["samples"], total_samples)
        self.assertGreater(captured["padding"], 0)
        self.assertTrue(captured["closed"])
        self.sd.play.assert_not_called()
        self.sd.playrec.assert_not_called()
        self.sleep.assert_not_called()


class AlsaIdentityTests(unittest.TestCase):
    def test_alsa_id_and_usb_bus_are_read_without_fixed_card_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            card = root / "proc/card7"
            card.mkdir(parents=True)
            (card / "id").write_text("UACDemoV10\n")
            (card / "usbbus").write_text("001/005\n")
            self.assertEqual(audio._alsa_cards(root / "proc", root / "sys"), {
                7: {"id": "UACDemoV10", "is_usb": True}
            })


if __name__ == "__main__":
    unittest.main()
