import os
import math
import subprocess
import json
import sys
import glob
from datetime import datetime


def get_app_root():
    """
    Return the CoBas Battery Reader V1.0 folder.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(script_dir))


def get_default_captures_dir():
    """
    Return the default Captures folder used by the CoBas app.
    """

    return os.path.join(get_app_root(), "Captures")


def get_default_output_folder():
    """
    Return the default folder for separated video/audio/waveform output.
    """

    return os.path.join(get_default_captures_dir(), "Separated_Output")


# ==========================================================
# FIND LAST CAPTURED VIDEO
# ==========================================================

def find_last_captured_video(captures_dir="Captures"):
    """
    Finds the most recently captured video file.
    
    Searches for common video formats in the specified directory.
    Returns the path to the most recent video file, or None if no videos found.
    """
    
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv"]
    
    # Handle both absolute and relative paths
    if not os.path.isabs(captures_dir):
        captures_dir = get_default_captures_dir()
    
    if not os.path.exists(captures_dir):
        print(f"Warning: Captures directory not found: {captures_dir}")
        return None
    
    latest_video = None
    latest_time = 0
    
    for ext in video_extensions:
        pattern = os.path.join(captures_dir, ext)
        for video_path in glob.glob(pattern):
            mod_time = os.path.getmtime(video_path)
            if mod_time > latest_time:
                latest_time = mod_time
                latest_video = video_path
    
    if latest_video:
        mod_date = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Found latest video: {os.path.basename(latest_video)}")
        print(f"Modified: {mod_date}")
        return latest_video
    
    print(f"No video files found in {captures_dir}")
    return None


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

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe could not read video duration: {input_video_path}")

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Invalid ffprobe duration output for {input_video_path}") from e


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
    output_folder=None,
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

    if output_folder is None:
        output_folder = get_default_output_folder()

    # --------------------------------------------------
    # Check input video
    # --------------------------------------------------
    if not os.path.exists(input_video_path):
        print("Error: Input video file does not exist.")
        return

    if cut_seconds <= 0:
        print("Error: cut_seconds must be greater than 0.")
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
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    
    # Get video path from command-line argument or find last captured
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        print(f"Processing specified video: {input_video}")
    else:
        # Find the last captured video in the Captures folder
        input_video = find_last_captured_video()
        
        if input_video is None:
            print("Error: No video file specified and no recent videos found in Captures folder.")
            sys.exit(1)
    
    # Determine output folder
    output_folder = get_default_output_folder()
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file not found: {input_video}")
        sys.exit(1)
    
    # Get cut duration from command-line argument (default: 2 seconds)
    cut_seconds = 2
    if "--cut-seconds" in sys.argv:
        try:
            idx = sys.argv.index("--cut-seconds")
            cut_seconds = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("Warning: Invalid --cut-seconds argument, using default of 2")
            cut_seconds = 2
    
    # Get output folder from command-line argument if provided
    if "--output-folder" in sys.argv:
        try:
            idx = sys.argv.index("--output-folder")
            output_folder = sys.argv[idx + 1]
        except IndexError:
            print("Warning: Invalid --output-folder argument, using default")
    
    print(f"Processing: {input_video}")
    print(f"Output folder: {output_folder}")
    print(f"Cut duration: {cut_seconds} seconds\n")
    
    cut_and_separate_audio_video(
        input_video_path=input_video,
        output_folder=output_folder,
        cut_seconds=cut_seconds
    )
