"""Hardware-free USB audio selection and exact-duration playback checks."""

import importlib.util
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

    def test_play_duplicates_exact_two_second_chirp_to_stereo(self):
        signal = pulse.build_pulse()
        pulse.play_pulse(signal)
        args, kwargs = self.sd.play.call_args
        self.assertEqual(args[0].shape, (96000, 2))
        np.testing.assert_array_equal(args[0][:, 0], signal)
        np.testing.assert_array_equal(args[0][:, 1], signal)
        self.assertEqual(kwargs["samplerate"], 48000)
        self.assertEqual(kwargs["device"], 2)
        self.sd.stop.assert_called_once()
        self.assertEqual((pulse.START_FREQUENCY, pulse.END_FREQUENCY, pulse.AMPLITUDE), (15000, 19200, .95))

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
        self.assertEqual(kwargs["device"], (1, 2))
        self.assertEqual(kwargs["channels"], 2)

    def test_sequence_preserves_stereo_pulses_and_exact_mono_recordings(self):
        captured = {}

        class FakeStream:
            active = False

            def __init__(self, **kwargs):
                captured.update(kwargs)
                captured["output"] = []
                self.kwargs = kwargs

            def start(self):
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
            pulse.play_and_record_pulse_sequence(signal, str(Path(directory) / "pulse_{pulse}.wav"), 2)
            for number in (1, 2):
                with wave.open(str(Path(directory) / f"pulse_{number}.wav"), "rb") as recorded:
                    self.assertEqual((recorded.getnchannels(), recorded.getframerate(), recorded.getnframes()), (1, 48000, 96000))
        self.assertEqual(captured["device"], (1, 2))
        self.assertEqual(captured["channels"], (1, 2))
        self.assertTrue(captured["closed"])
        playback = np.concatenate(captured["output"])
        expected = np.tile(signal, 2)
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
