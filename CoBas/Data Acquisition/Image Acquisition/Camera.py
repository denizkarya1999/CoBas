import cv2
import numpy as np
import time
import os
import tkinter as tk
from PIL import Image, ImageTk

os.makedirs("Samples", exist_ok=True)

DEVICE_ID = 0

cap = None
current_thermal = None
video_writer = None
is_recording = False


def open_camera():
    camera = cv2.VideoCapture(DEVICE_ID, cv2.CAP_V4L2)

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 342)
    camera.set(cv2.CAP_PROP_FPS, 5)

    time.sleep(1)

    return camera


def restart_camera():
    global cap

    if is_recording:
        root.after(5000, restart_camera)
        return

    status_label.config(text="Restarting camera...")

    if cap is not None:
        cap.release()

    time.sleep(0.5)
    cap = open_camera()

    status_label.config(text="Camera restarted" if cap.isOpened() else "Camera restart failed")

    root.after(5000, restart_camera)


def process_frame(frame):
    if len(frame.shape) == 3 and frame.shape[2] == 2:
        gray = frame[:, :, 0]
    elif len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    display = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    display = display.astype(np.uint8)

    return cv2.applyColorMap(display, cv2.COLORMAP_INFERNO)


def update_frame():
    global current_thermal, video_writer

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()

        if ret and frame is not None:
            current_thermal = process_frame(frame)

            if is_recording and video_writer is not None:
                video_writer.write(current_thermal)

            rgb = cv2.cvtColor(current_thermal, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((512, 684))

            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)

    root.after(100, update_frame)


def capture_image():
    if current_thermal is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"Samples/gc001_capture_{timestamp}.png"

        cv2.imwrite(save_path, current_thermal)

        status_label.config(text=f"Saved image: {save_path}")
    else:
        status_label.config(text="No frame available")


def toggle_video_recording():
    global video_writer, is_recording

    if not is_recording:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"Samples/gc001_video_{timestamp}.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(save_path, fourcc, 5.0, (256, 342))

        is_recording = True
        video_button.config(text="Stop Video")
        status_label.config(text=f"Recording video: {save_path}")

    else:
        is_recording = False

        if video_writer is not None:
            video_writer.release()
            video_writer = None

        video_button.config(text="Start Video")
        status_label.config(text="Video saved")


def close_app():
    global cap, video_writer, is_recording

    is_recording = False

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
    root.destroy()


root = tk.Tk()
root.title("CoBas Camera")

camera_label = tk.Label(root)
camera_label.pack(padx=10, pady=10)

capture_button = tk.Button(
    root,
    text="Capture Image",
    command=capture_image,
    font=("Arial", 14)
)
capture_button.pack(pady=5)

video_button = tk.Button(
    root,
    text="Start Video",
    command=toggle_video_recording,
    font=("Arial", 14)
)
video_button.pack(pady=5)

status_label = tk.Label(root, text="Starting camera...", font=("Arial", 11))
status_label.pack(pady=5)

root.protocol("WM_DELETE_WINDOW", close_app)

cap = open_camera()

status_label.config(text="Camera running" if cap.isOpened() else "Could not open camera")

update_frame()
root.after(5000, restart_camera)

root.mainloop()