import os
import math
import subprocess
import json


# ==========================================================
# RUN COMMAND WITH LOW MEMORY OUTPUT HANDLING
# ==========================================================

def run_command(command):
    """
    Runs a command without storing large stdout/stderr in memory.
    This is better for Raspberry Pi.
    """

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


# ==========================================================
# GET VIDEO DURATION USING FFPROBE
# ==========================================================

def get_video_duration(input_video_path):
    """
    Gets video duration using ffprobe instead of MoviePy.
    This reduces memory usage.
    """

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        input_video_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# ==========================================================
# CREATE WAVEFORM IMAGE USING FFMPEG
# ==========================================================

def create_waveform_image(audio_path, output_waveform_path):
    """
    Creates waveform image using FFmpeg instead of librosa/matplotlib.
    This is much lighter for Raspberry Pi memory.
    """

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-filter_complex", "showwavespic=s=1280x300",
        "-frames:v", "1",
        output_waveform_path
    ]

    return run_command(command)


# ==========================================================
# CUT VIDEO, SEPARATE AUDIO, AND CREATE WAVEFORM
# ==========================================================

def cut_and_separate_audio_video(
    input_video_path,
    output_folder="Separated_Output",
    cut_seconds=2
):
    """
    This function:
    1. Cuts the input video every 2 seconds.
    2. Saves video-only clips.
    3. Saves audio-only clips.
    4. Creates waveform image for each audio file.
    5. Uses FFmpeg only to reduce memory usage.
    """

    # --------------------------------------------------
    # Check input video
    # --------------------------------------------------
    if not os.path.exists(input_video_path):
        print("Error: Input video file does not exist.")
        return

    # --------------------------------------------------
    # Output folders
    # --------------------------------------------------
    video_output_folder = os.path.join(output_folder, "Videos")
    audio_output_folder = os.path.join(output_folder, "Audios")
    waveform_output_folder = os.path.join(output_folder, "Audio_Waveforms")

    os.makedirs(video_output_folder, exist_ok=True)
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(waveform_output_folder, exist_ok=True)

    # --------------------------------------------------
    # Get video name
    # --------------------------------------------------
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # --------------------------------------------------
    # Get duration using ffprobe
    # --------------------------------------------------
    duration = get_video_duration(input_video_path)
    total_clips = math.ceil(duration / cut_seconds)

    print(f"Video duration: {duration:.2f} seconds")
    print(f"Total clips: {total_clips}")

    # --------------------------------------------------
    # Process each clip
    # --------------------------------------------------
    for i in range(total_clips):

        start_time = i * cut_seconds
        end_time = min((i + 1) * cut_seconds, duration)
        segment_duration = end_time - start_time

        start_label = int(start_time)
        end_label = int(end_time)

        clip_name = f"{video_name}_{start_label}s_to_{end_label}s"

        output_video_path = os.path.join(
            video_output_folder,
            clip_name + "_video.mp4"
        )

        output_audio_path = os.path.join(
            audio_output_folder,
            clip_name + "_audio.wav"
        )

        output_waveform_path = os.path.join(
            waveform_output_folder,
            clip_name + "_waveform.png"
        )

        print(f"\nProcessing clip: {start_label}s to {end_label}s")

        # --------------------------------------------------
        # Save video-only clip
        #
        # -an removes audio.
        # -c:v copy avoids re-encoding, reducing CPU and memory.
        # --------------------------------------------------
        video_command = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_video_path,
            "-t", str(segment_duration),
            "-an",
            "-c:v", "copy",
            output_video_path
        ]

        if run_command(video_command):
            print(f"Video saved: {output_video_path}")
        else:
            print(f"Video failed: {output_video_path}")

        # --------------------------------------------------
        # Save audio-only clip
        #
        # Mono audio uses less space than stereo.
        # 16000 Hz is enough for many sound-analysis tasks.
        # Change 16000 to 44100 if you need original-quality audio.
        # --------------------------------------------------
        audio_command = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", input_video_path,
            "-t", str(segment_duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio_path
        ]

        audio_success = run_command(audio_command)

        if audio_success:
            print(f"Audio saved: {output_audio_path}")

            # --------------------------------------------------
            # Create waveform image using FFmpeg
            # --------------------------------------------------
            waveform_success = create_waveform_image(
                output_audio_path,
                output_waveform_path
            )

            if waveform_success:
                print(f"Waveform saved: {output_waveform_path}")
            else:
                print(f"Waveform failed: {output_waveform_path}")

        else:
            print(f"Audio failed: {output_audio_path}")

    print("\nFinished processing.")


# ==========================================================
# EXAMPLE USAGE
# ==========================================================

input_video = "Jeniffer_Lopez_Play_Live.mp4"

cut_and_separate_audio_video(
    input_video_path=input_video,
    output_folder="Separated_Output",
    cut_seconds=2
)