"""Hardware-free capture integrity checks: python -m unittest discover -s tests -v."""
import importlib.util
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

APP = Path(os.environ.get('COBAS_APP_SOURCE', Path(__file__).resolve().parents[1]))
UNDER_TEST = Path(os.environ.get('COBAS_APP_UNDER_TEST', APP))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'Camera/thermal_camera'))


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, UNDER_TEST / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load_module('capture_under_test', 'CoBas_V1.py')
settings = load_module('settings_under_test', 'Settings.py')
thermal = load_module('thermal_under_test', 'Camera/thermal_camera/thermal_camera_logic.py')
thermal_gui = load_module('Camera.thermal_camera.gui_under_test', 'Camera/thermal_camera/gui.py')


class CaptureSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.app = object.__new__(main.CoBasV1App)
        self.app.is_closing = False
        self.app.export_in_progress = False
        self.app.pulse_sequence_active = False
        self.app.tracking_start_token = 2
        self.app.ui_callbacks = queue.Queue()

    def write_audio(self, name='voice.wav', frames=96000, rate=48000, channels=1):
        path = self.path / name
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(b'\0\0' * frames * channels)
        self.app.pulse_recordings = [{'path': str(path), 'pulse_number': 1}]
        return path

    def test_repeat_battery_capture_preserves_previous_files(self):
        self.app.base_dir = str(self.path)
        self.app.battery_percentage = 50
        original = self.path / 'Captures/50_Percent_Battery/Voices/existing.wav'
        original.parent.mkdir(parents=True)
        original.write_bytes(b'previous capture')
        self.app.prepare_capture_session()
        first = Path(self.app.captures_dir)
        sample = first / 'Thermal Frames/frame.png'
        sample.write_bytes(b'first session')
        self.app.prepare_capture_session()
        self.assertNotEqual(first, Path(self.app.captures_dir))
        self.assertEqual(original.read_bytes(), b'previous capture')
        self.assertEqual(sample.read_bytes(), b'first session')

    def test_export_blocks_start_before_state_can_be_reset(self):
        self.app.export_in_progress = True
        self.app.start_tracking()
        self.assertEqual(self.app.tracking_start_token, 2)

    def test_export_blocks_close(self):
        self.app.export_in_progress = True
        self.app.root = MagicMock()
        self.app.mmwave_poll_after_id = None
        self.app.cancel_thermal_preview_loop = MagicMock()
        self.app.stop_pulse_process = MagicMock()
        self.app.mmwave_capture = SimpleNamespace(is_running=False)
        self.app.thermal_camera = MagicMock(is_recording=False)
        self.app.on_close()
        self.assertFalse(self.app.is_closing)
        self.app.root.destroy.assert_not_called()

    def test_stale_capture_callback_is_discarded(self):
        seen = []
        self.app.post_ui(seen.append, 'old session', token=1)
        self.app.post_ui(seen.append, 'current session', token=2)
        self.app.drain_ui_callbacks()
        self.assertEqual(seen, ['current session'])

    def test_worker_does_not_call_tk_or_mutate_state(self):
        seen = []
        self.app.root = MagicMock()
        worker = threading.Thread(target=lambda: self.app.post_ui(seen.append, 1, token=2))
        worker.start()
        worker.join()
        self.assertEqual(seen, [])
        self.app.root.after.assert_not_called()
        self.app.drain_ui_callbacks()
        self.assertEqual(seen, [1])

    def test_closed_app_ignores_worker_results(self):
        self.app.is_closing = True
        self.app.post_ui(lambda: self.fail('Closed window was updated'))
        self.assertTrue(self.app.ui_callbacks.empty())

    def test_abort_invalidates_pending_callbacks(self):
        self.app.pulse_sequence_active = True
        self.app.current_position_number = 1
        self.app.stop_pulse_process = MagicMock()
        self.app.finalize_position_segment = MagicMock()
        self.app.finish_capture = MagicMock()
        self.app.abort_capture('Disconnected')
        self.assertGreater(self.app.tracking_start_token, 2)

    def test_stale_recording_cannot_enter_new_session(self):
        self.app.pulse_sequence_active = True
        self.app.pulse_recordings = []
        self.app.handle_pulse_completed = MagicMock()
        self.app.register_pulse_recording(1, 1, 1, 1, 'old.wav')
        self.assertEqual(self.app.pulse_recordings, [])
        self.app.handle_pulse_completed.assert_not_called()

    def test_current_recording_updates_on_ui_thread(self):
        self.app.pulse_sequence_active = True
        self.app.pulse_recordings = []
        self.app.handle_pulse_completed = MagicMock()
        self.app.register_pulse_recording(2, 1, 1, 1, 'current.wav')
        self.assertEqual(self.app.pulse_recordings[0]['path'], 'current.wav')
        self.app.handle_pulse_completed.assert_called_once_with(2, 1, 1, 1)

    def test_stop_remains_available_while_sensors_are_connecting(self):
        self.app.is_preparing_tracking = True
        self.app.pulse_sequence_active = True
        self.app.stop_tracking = MagicMock()
        self.app.toggle_tracking()
        self.app.stop_tracking.assert_called_once()

    def test_pulse_worker_handshake_preserves_recording_order(self):
        script = self.path / 'fake_pulse.py'
        script.write_text("""import sys, wave
from pathlib import Path
folder = Path(sys.argv[1])
print('SEQUENCE_READY 2 4.000', flush=True)
assert sys.stdin.readline().strip() == 'START'
for number in (1, 2):
    path = folder / f'voice_{number}.wav'
    print(f'PLAYBACK_STARTED {number} {1000 + (number-1)*2}', flush=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(b'\\0\\0' * 96000)
    print(f'PULSE_FINISHED {number} {path}', flush=True)
""")
        self.app.current_position_number = 1
        self.app.pulses_per_position = 2
        self.app.pulse_sequence_active = True
        self.app.pulse_recordings = []
        self.app.pulse_process = None
        self.app.pulse_process_lock = threading.Lock()
        self.app.get_pulse_command = lambda: ([sys.executable, str(script), str(self.path)], str(self.path))
        self.app.handle_sequence_ready = MagicMock(return_value=True)
        self.app.handle_pulse_started = MagicMock()
        self.app.handle_pulse_completed = MagicMock()
        self.app.handle_position_finished = MagicMock()
        self.app.abort_capture = MagicMock()
        self.app.start_pulse_sequence(2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.drain_ui_callbacks()
            if self.app.handle_position_finished.called or self.app.abort_capture.called:
                break
            time.sleep(0.01)
        self.app.abort_capture.assert_not_called()
        self.app.handle_position_finished.assert_called_once_with(2, 1)
        self.assertEqual([item['pulse_number'] for item in self.app.pulse_recordings], [1, 2])
        self.assertEqual(self.app.handle_pulse_completed.call_count, 2)
        self.assertEqual(len(self.app.validate_chirp_voice_recordings()), 2)

    def test_complete_audio_accepted(self):
        path = self.write_audio()
        self.assertEqual(self.app.validate_chirp_voice_recordings(), [str(path)])

    def test_short_audio_rejected(self):
        self.write_audio(frames=48000)
        with self.assertRaises(RuntimeError):
            self.app.validate_chirp_voice_recordings()

    def test_wrong_sample_rate_rejected(self):
        self.write_audio(rate=44100, frames=88200)
        with self.assertRaises(RuntimeError):
            self.app.validate_chirp_voice_recordings()

    def test_truncated_payload_rejected_even_with_complete_header(self):
        path = self.write_audio()
        path.write_bytes(path.read_bytes()[:-1000])
        with self.assertRaises(RuntimeError):
            self.app.validate_chirp_voice_recordings()

    def test_existing_thermal_frames_are_not_deleted(self):
        staging = self.path / 'staging'
        staging.mkdir()
        frame = staging / 'new.png'
        frame.write_bytes(b'new')
        output = self.path / 'output'
        output.mkdir()
        prior = output / 'prior.png'
        prior.write_bytes(b'prior')
        self.app.thermal_frames_staging_directory = staging
        self.app.thermal_frames_dir = str(output)
        self.app.thermal_frame_records = [{'path': str(frame), 'pulse_number': 1}]
        with self.assertRaises(RuntimeError):
            self.app.finalize_thermal_frames(1)
        self.assertEqual(prior.read_bytes(), b'prior')
        self.assertEqual(frame.read_bytes(), b'new')

    def test_thermal_frames_can_be_published_to_empty_destination(self):
        staging = self.path / 'staging'
        staging.mkdir()
        frame = staging / 'frame.png'
        frame.write_bytes(b'frame')
        output = self.path / 'output'
        output.mkdir()
        self.app.thermal_frames_staging_directory = staging
        self.app.thermal_frames_dir = str(output)
        self.app.thermal_frame_records = [{'path': str(frame), 'pulse_number': 1}]
        self.app.finalize_thermal_frames(1)
        self.assertEqual((output / 'frame.png').read_bytes(), b'frame')

    def test_missing_video_segment_is_not_silently_omitted(self):
        existing = self.path / 'one.mp4'
        existing.write_bytes(b'original segment')
        segments = [{'position': 1, 'thermal_video_path': str(existing)},
                    {'position': 2, 'thermal_video_path': str(self.path / 'missing.mp4')}]
        with self.assertRaises(RuntimeError):
            self.app.concatenate_thermal_videos(segments, str(self.path / 'combined.mp4'))
        self.assertEqual(existing.read_bytes(), b'original segment')

    def test_export_error_unlocks_retry_without_deleting_data(self):
        self.app.export_in_progress = True
        self.app.track_button = MagicMock()
        self.app.update_status = MagicMock()
        self.app.captures_dir = str(self.path)
        file = self.path / 'partial.wav'
        file.write_bytes(b'partial')
        self.app.handle_export_failed('disk full')
        self.assertFalse(self.app.export_in_progress)
        self.assertEqual(file.read_bytes(), b'partial')
        self.assertEqual(self.app.track_button.configure.call_args.kwargs['state'], 'normal')

    def test_missing_microphone_does_not_crash_apply(self):
        window = object.__new__(settings.SettingsWindow)
        window.app = SimpleNamespace(pulse_sequence_active=False)
        window.selected_id = 999
        window.microphones = [{'id': None, 'name': 'System default'}]
        window.dialog = MagicMock()
        window.render = MagicMock()
        with patch.object(settings.dialogs, 'showwarning') as warning:
            window.apply()
        warning.assert_called_once()
        window.dialog.finish.assert_not_called()

    def test_slow_thermal_worker_remains_tracked_after_stop(self):
        camera = object.__new__(thermal_gui.ThermalCamera)
        camera.is_recording = False
        camera.recorder = camera.scale_recorder = None
        camera.stop_event = threading.Event()
        camera.frame_lock = threading.Lock()
        worker = MagicMock()
        worker.is_alive.return_value = True
        camera.worker = worker
        camera.stop_camera()
        self.assertIs(camera.worker, worker)
        self.assertTrue(camera.stop_event.is_set())
        with self.assertRaises(RuntimeError):
            camera.start_camera()
        self.assertTrue(camera.stop_event.is_set())

    def test_driver_compile_failure_preserves_working_library(self):
        output = self.path / 'driver.so'
        output.write_bytes(b'working library')
        def failed_compile(command, **kwargs):
            Path(command[-1]).write_bytes(b'broken partial library')
            return SimpleNamespace(returncode=1, stderr='compile failed', stdout='')
        with patch.object(thermal.subprocess, 'run', side_effect=failed_compile):
            with self.assertRaises(thermal.DriverError):
                thermal.build_shared_library(output)
        self.assertEqual(output.read_bytes(), b'working library')
        self.assertEqual(list(self.path.iterdir()), [output])

    def test_successful_driver_build_replaces_atomically(self):
        output = self.path / 'driver.so'
        output.write_bytes(b'old')
        inode = output.stat().st_ino
        def compile_ok(command, **kwargs):
            Path(command[-1]).write_bytes(b'new')
            return SimpleNamespace(returncode=0, stderr='', stdout='')
        with patch.object(thermal.subprocess, 'run', side_effect=compile_ok):
            thermal.build_shared_library(output)
        self.assertEqual(output.read_bytes(), b'new')
        self.assertNotEqual(output.stat().st_ino, inode)


if __name__ == '__main__':
    unittest.main()
