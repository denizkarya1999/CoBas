import cv2
import os
import queue
import shutil
import time
import threading
import subprocess
from datetime import datetime

import sounddevice as sd
import soundfile as sf


class Camera:
    """
    Camera logic for CoBas_V1.

    Handles:
    - Camera opening
    - Camera switching
    - Frame reading
    - Digital zoom
    - Photo capture
    - Video recording
    - External microphone selection
    - Audio recording
    - Video/audio merging with FFmpeg
    """

    def __init__(self, camera_index="/dev/video0", output_dir="Captures"):
        self.camera_index = camera_index
        self.output_dir = output_dir

        self.cap = None
        self.is_tracking = False

        # Digital zoom
        self.zoom_factor = 1.0
        self.max_zoom = 3.0
        self.min_zoom = 1.0

        # Video recording
        self.video_writer = None
        self.is_recording = False
        self.record_start_time = None
        self.record_fps = 20.0

        # Audio recording
        self.audio_sample_rate = 44100
        self.audio_channels = 1

        # None means use system default microphone.
        # Otherwise, this stores the selected sounddevice input device index.
        self.microphone_device_id = None
        self.microphone_device_name = "System Default Microphone"

        self.audio_queue = None
        self.audio_thread = None
        self.audio_recording = False
        self.audio_available = True
        self.audio_chunks_written = 0
        self.audio_chunks_dropped = 0

        # Recording paths
        self.temp_video_path = None
        self.temp_audio_path = None
        self.final_video_path = None
        self.voice_audio_path = None
        self.last_saved_voice_path = None

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Camera Source Handling
    # --------------------------------------------------

    def set_camera_source(self, camera_source):
        self.camera_index = camera_source

    def switch_camera_source(self):
        """
        Switch between /dev/video0 and /dev/video1.
        Useful for Android Webcam front/back switching.
        """

        if self.camera_index == "/dev/video0":
            self.camera_index = "/dev/video1"
        else:
            self.camera_index = "/dev/video0"

        return self.camera_index

    def _unique_camera_sources(self):
        sources = [
            self.camera_index,
            "/dev/video0",
            "/dev/video1",
            0,
            1,
        ]

        unique_sources = []

        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)

        return unique_sources

    def _try_open_camera(self, source):
        print(f"Trying camera source: {source}")

        camera = cv2.VideoCapture(source, cv2.CAP_V4L2)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(source)

        if not camera.isOpened():
            camera.release()
            print(f"Failed to open camera source: {source}")
            return None

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, self.record_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)

        ret, frame = camera.read()

        if not ret or frame is None:
            camera.release()
            print(f"Opened but could not read frame from: {source}")
            return None

        print(f"Camera opened successfully: {source}")
        return camera

    # --------------------------------------------------
    # Camera Control
    # --------------------------------------------------

    def start_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.is_tracking = True
            return True

        for source in self._unique_camera_sources():
            camera = self._try_open_camera(source)

            if camera is not None:
                self.cap = camera
                self.camera_index = source
                self.is_tracking = True
                return True

        self.cap = None
        self.is_tracking = False
        print("No working camera found.")
        return False

    def stop_camera(self):
        self.is_tracking = False
        if self.is_recording or self.video_writer is not None:
            self.stop_recording()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def restart_camera(self):
        self.stop_camera()
        return self.start_camera()

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        frame = self.apply_zoom(frame)

        return frame

    # --------------------------------------------------
    # Microphone Handling
    # --------------------------------------------------

    def get_input_microphones(self):
        """
        Return available audio input devices.

        Returns:
            List of dictionaries:
            [
                {"id": None, "name": "System Default Microphone"},
                {"id": 3, "name": "USB Microphone"},
                ...
            ]
        """

        microphones = [
            {
                "id": None,
                "name": "System Default Microphone"
            }
        ]

        try:
            devices = sd.query_devices()

            for index, device in enumerate(devices):
                max_input_channels = device.get("max_input_channels", 0)

                if max_input_channels > 0:
                    microphones.append(
                        {
                            "id": index,
                            "name": device.get("name", f"Input Device {index}")
                        }
                    )

        except Exception as e:
            print(f"Could not list microphones: {e}")

        return microphones

    def set_microphone_device(self, device_id, device_name):
        """
        Set microphone device used for audio recording.

        device_id:
            None means system default microphone.
            Otherwise, this should be a sounddevice input device index.

        device_name:
            User-readable microphone name.
        """

        self.microphone_device_id = device_id
        self.microphone_device_name = device_name

    # --------------------------------------------------
    # Zoom
    # --------------------------------------------------

    def zoom_in(self):
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor += 0.2
            self.zoom_factor = round(self.zoom_factor, 1)

    def zoom_out(self):
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor -= 0.2
            self.zoom_factor = round(self.zoom_factor, 1)

    def reset_zoom(self):
        self.zoom_factor = 1.0

    def apply_zoom(self, frame):
        if self.zoom_factor <= 1.0:
            return frame

        height, width = frame.shape[:2]

        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)

        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2
        x2 = x1 + new_width
        y2 = y1 + new_height

        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height))

        return zoomed

    # --------------------------------------------------
    # Photo Capture
    # --------------------------------------------------

    def take_photo(self):
        frame = self.read_frame()

        if frame is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CoBas_V1_Photo_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        cv2.imwrite(filepath, frame)

        return filepath

    # --------------------------------------------------
    # Audio Recording
    # --------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio warning: {status}")

        if not self.audio_recording or self.audio_queue is None:
            return

        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            self.audio_chunks_dropped += 1

    def _record_audio(self):
        """
        Record microphone audio while video recording is active.
        Uses selected external microphone if one was chosen.
        Audio chunks are streamed to disk through a bounded queue so long
        recordings do not grow memory usage.
        """

        try:
            with sf.SoundFile(
                self.temp_audio_path,
                mode="w",
                samplerate=self.audio_sample_rate,
                channels=self.audio_channels
            ) as audio_file:
                with sd.InputStream(
                    samplerate=self.audio_sample_rate,
                    channels=self.audio_channels,
                    device=self.microphone_device_id,
                    blocksize=4096,
                    latency="high",
                    callback=self._audio_callback
                ):
                    while self.audio_recording or (
                        self.audio_queue is not None
                        and not self.audio_queue.empty()
                    ):
                        try:
                            chunk = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        audio_file.write(chunk)
                        self.audio_chunks_written += len(chunk)

        except Exception as e:
            self.audio_available = False
            self.audio_recording = False
            print(f"Audio recording failed: {e}")

    def _start_audio_recording(self):
        # Keep enough buffered microphone data to survive temporary CPU spikes
        # while both high-resolution thermal views are being rendered. The
        # previous small queue could discard most of a long recording.
        self.audio_queue = queue.Queue(maxsize=512)
        self.audio_available = True
        self.audio_recording = True
        self.audio_chunks_written = 0
        self.audio_chunks_dropped = 0

        self.audio_thread = threading.Thread(
            target=self._record_audio,
            daemon=True
        )
        self.audio_thread.start()

    def _stop_audio_recording(self):
        self.audio_recording = False

        if self.audio_thread is not None:
            self.audio_thread.join(timeout=5)
            if self.audio_thread.is_alive():
                print("Audio writer did not finish cleanly.")
                self.audio_available = False
                return None

            self.audio_thread = None

        if not self.audio_available:
            return None

        if self.audio_chunks_dropped:
            print(f"Audio chunks dropped: {self.audio_chunks_dropped}")

        if self.audio_chunks_written == 0:
            print("No audio frames captured.")
            return None

        return self.temp_audio_path

    # --------------------------------------------------
    # Video Recording with Audio
    # --------------------------------------------------

    def start_recording(self, timestamp=None):
        """
        Start video and microphone audio recording.
        """

        if self.cap is None or not self.cap.isOpened():
            return None

        frame = self.read_frame()

        if frame is None:
            return None

        height, width = frame.shape[:2]

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.temp_video_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_TempVideo_{timestamp}.mp4"
        )

        self.temp_audio_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_TempAudio_{timestamp}.wav"
        )

        self.voice_audio_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Voice_{timestamp}.wav"
        )
        self.last_saved_voice_path = None

        self.final_video_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Video_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.video_writer = cv2.VideoWriter(
            self.temp_video_path,
            fourcc,
            self.record_fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            return None

        self._start_audio_recording()

        self.is_recording = True
        self.record_start_time = time.time()

        return self.final_video_path

    def write_video_frame(self, frame):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_recording(self, keep_audio=False):
        """
        Stop recording and merge audio/video.

        Returns:
            Final merged MP4 path if successful.
            Video-only temp path if audio fails.
            None if nothing was recorded.
        """

        audio_path = self.stop_recording_capture_phase()
        return self.finalize_recording(audio_path=audio_path, keep_audio=keep_audio)

    def stop_recording_capture_phase(self):
        """
        Stop active video/audio capture immediately and return captured WAV path.

        This method only ends capture. It does not run FFmpeg merge, so callers
        can stop multiple recorders first and finalize files afterwards.
        """

        if not self.is_recording and self.video_writer is None:
            return None

        if self.video_writer is not None:
            self.video_writer.release()

        self.video_writer = None

        audio_path = self._stop_audio_recording()

        self.is_recording = False
        self.record_start_time = None

        return audio_path

    def finalize_recording(self, audio_path=None, keep_audio=False):
        """
        Finalize files after capture has already stopped.

        Returns:
            Final merged MP4 path if successful.
            Video-only temp path if audio fails.
            None if nothing was recorded.
        """

        if not self.temp_video_path:
            return None

        if audio_path is None:
            print("Audio failed or unavailable. Keeping video-only file.")

            if self.temp_video_path and os.path.exists(self.temp_video_path):
                return self.temp_video_path

            return None

        self.last_saved_voice_path = self._save_voice_copy(audio_path)

        merged_path = self._merge_video_audio(
            self.temp_video_path,
            audio_path,
            self.final_video_path,
            remove_audio=not keep_audio,
            shortest=not keep_audio
        )

        return merged_path

    def _save_voice_copy(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return None

        if not self.voice_audio_path:
            return None

        try:
            shutil.copyfile(audio_path, self.voice_audio_path)
            return self.voice_audio_path
        except Exception as e:
            print(f"Voice copy save failed: {e}")
            return None

    def _merge_video_audio(
        self,
        video_path,
        audio_path,
        output_path,
        remove_audio=True,
        shortest=True
    ):
        if not video_path or not audio_path:
            return video_path

        if not os.path.exists(video_path):
            print("Video file does not exist.")
            return None

        if not os.path.exists(audio_path):
            print("Audio file does not exist.")
            return video_path

        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
        ]

        if shortest:
            command.append("-shortest")

        command.append(output_path)

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if os.path.exists(video_path):
                os.remove(video_path)

            if remove_audio and os.path.exists(audio_path):
                os.remove(audio_path)

            return output_path

        except Exception as e:
            print(f"FFmpeg merge failed: {e}")
            return video_path

    def cleanup_temp_audio(self):
        """
        Remove the last temporary audio file after other recorders have used it.
        """

        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            os.remove(self.temp_audio_path)

    def get_recording_seconds(self):
        if not self.is_recording or self.record_start_time is None:
            return 0

        return int(time.time() - self.record_start_time)
