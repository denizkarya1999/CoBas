import os
import sys
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================
# SCRIPT PATHS
# ==========================================================

SCRIPTS = {
    "Image / Video Capturing": {
        "path": os.path.join(
            BASE_DIR,
            "Image Acquisition",
            "Camera.py"
        ),
        "description": "Open the thermal camera UI and capture images or videos.",
        "folder": os.path.join(
            BASE_DIR,
            "Image Acquisition",
            "Samples"
        )
    },

    "Sound Generation": {
        "path": os.path.join(
            BASE_DIR,
            "Sound Acquisition",
            "pulse_protocol_generator.py"
        ),
        "description": "Generate the beacon and chirp protocol WAV file.",
        "folder": os.path.join(
            BASE_DIR,
            "Sound Acquisition",
            "Inputs"
        )
    },

    "Sound Processing": {
        "path": os.path.join(
            BASE_DIR,
            "Sound Acquisition",
            "protocol_visualization.py"
        ),
        "description": "Process the generated audio and create waveform output.",
        "folder": os.path.join(
            BASE_DIR,
            "Sound Acquisition",
            "Outputs"
        )
    },
}


# ==========================================================
# GLOBAL STATE
# ==========================================================

running_process = None
current_script_name = None


# ==========================================================
# COLORS
# ==========================================================

BG_COLOR = "#111827"
PANEL_COLOR = "#1f2937"
CARD_COLOR = "#273449"
TEXT_COLOR = "#f9fafb"
MUTED_TEXT_COLOR = "#cbd5e1"
ACCENT_COLOR = "#38bdf8"
SUCCESS_COLOR = "#22c55e"
WARNING_COLOR = "#facc15"
ERROR_COLOR = "#ef4444"
CONSOLE_BG = "#020617"
CONSOLE_FG = "#d1fae5"


# ==========================================================
# LOGGING
# ==========================================================

def write_console(text):

    output_box.configure(state="normal")
    output_box.insert(tk.END, text)
    output_box.see(tk.END)
    output_box.configure(state="disabled")


def clear_console():

    output_box.configure(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.configure(state="disabled")


# ==========================================================
# STATUS HANDLING
# ==========================================================

def set_status(text, color=ACCENT_COLOR):

    status_value_label.config(
        text=text,
        fg=color
    )


def set_buttons_state(is_running):

    image_button.config(
        state="disabled" if is_running else "normal"
    )

    sound_generation_button.config(
        state="disabled" if is_running else "normal"
    )

    sound_processing_button.config(
        state="disabled" if is_running else "normal"
    )

    stop_button.config(
        state="normal" if is_running else "disabled"
    )


# ==========================================================
# CHECK SCRIPT FILE
# ==========================================================

def validate_script(script_name):

    script_path = SCRIPTS[script_name]["path"]

    if not os.path.exists(script_path):

        messagebox.showerror(
            "Script Not Found",
            f"Could not find:\n\n{script_path}"
        )

        return False

    return True


# ==========================================================
# RUN SCRIPT
# ==========================================================

def run_script(script_name):

    global running_process
    global current_script_name

    if running_process is not None and running_process.poll() is None:

        messagebox.showwarning(
            "Process Already Running",
            "Another script is already running. Stop it before starting a new one."
        )

        return

    if not validate_script(script_name):
        return

    script_path = SCRIPTS[script_name]["path"]
    script_dir = os.path.dirname(script_path)

    clear_console()

    write_console("============================================================\n")
    write_console(f"Starting: {script_name}\n")
    write_console(f"Script: {script_path}\n")
    write_console("============================================================\n\n")

    set_status(
        f"Running: {script_name}",
        WARNING_COLOR
    )

    set_buttons_state(True)

    current_script_name = script_name

    def worker():

        global running_process
        global current_script_name

        try:

            running_process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if os.name != "nt" else None
            )

            for line in running_process.stdout:

                root.after(
                    0,
                    write_console,
                    line
                )

            return_code = running_process.wait()

            if return_code == 0:

                root.after(
                    0,
                    write_console,
                    "\nProcess finished successfully.\n"
                )

                root.after(
                    0,
                    set_status,
                    f"Finished: {script_name}",
                    SUCCESS_COLOR
                )

            else:

                root.after(
                    0,
                    write_console,
                    f"\nProcess exited with code: {return_code}\n"
                )

                root.after(
                    0,
                    set_status,
                    f"Stopped / Error: {script_name}",
                    ERROR_COLOR
                )

        except Exception as e:

            root.after(
                0,
                write_console,
                f"\nExecution Error:\n{e}\n"
            )

            root.after(
                0,
                set_status,
                "Execution Error",
                ERROR_COLOR
            )

        finally:

            running_process = None
            current_script_name = None

            root.after(
                0,
                set_buttons_state,
                False
            )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ==========================================================
# STOP SCRIPT
# ==========================================================

def stop_script():

    global running_process
    global current_script_name

    if running_process is None or running_process.poll() is not None:

        set_status(
            "No active process",
            MUTED_TEXT_COLOR
        )

        return

    try:

        write_console("\nStopping process...\n")

        if os.name == "nt":

            running_process.terminate()

        else:

            os.killpg(
                os.getpgid(running_process.pid),
                signal.SIGTERM
            )

        set_status(
            "Process stopped",
            ERROR_COLOR
        )

    except Exception as e:

        write_console(
            f"\nCould not stop process:\n{e}\n"
        )

        set_status(
            "Stop failed",
            ERROR_COLOR
        )


# ==========================================================
# OPEN FOLDER
# ==========================================================

def open_folder(folder_path):

    if not os.path.exists(folder_path):

        try:

            os.makedirs(
                folder_path,
                exist_ok=True
            )

        except Exception as e:

            messagebox.showerror(
                "Folder Error",
                f"Could not create/open folder:\n\n{folder_path}\n\n{e}"
            )

            return

    try:

        if os.name == "nt":

            os.startfile(folder_path)

        elif sys.platform == "darwin":

            subprocess.Popen(
                ["open", folder_path]
            )

        else:

            subprocess.Popen(
                ["xdg-open", folder_path]
            )

    except Exception as e:

        messagebox.showerror(
            "Open Folder Error",
            f"Could not open folder:\n\n{folder_path}\n\n{e}"
        )


# ==========================================================
# EXIT APP
# ==========================================================

def close_app():

    global running_process

    if running_process is not None and running_process.poll() is None:

        answer = messagebox.askyesno(
            "Process Running",
            "A script is still running. Do you want to stop it and close the app?"
        )

        if not answer:
            return

        stop_script()

    root.destroy()


# ==========================================================
# CREATE CARD
# ==========================================================

def create_action_card(
    parent,
    title,
    description,
    button_text,
    command,
    row
):

    card = tk.Frame(
        parent,
        bg=CARD_COLOR,
        highlightthickness=1,
        highlightbackground="#334155"
    )

    card.grid(
        row=row,
        column=0,
        sticky="ew",
        padx=18,
        pady=8
    )

    card.grid_columnconfigure(
        0,
        weight=1
    )

    title_label = tk.Label(
        card,
        text=title,
        font=("Arial", 13, "bold"),
        fg=TEXT_COLOR,
        bg=CARD_COLOR,
        anchor="w"
    )

    title_label.grid(
        row=0,
        column=0,
        sticky="w",
        padx=15,
        pady=(12, 2)
    )

    description_label = tk.Label(
        card,
        text=description,
        font=("Arial", 10),
        fg=MUTED_TEXT_COLOR,
        bg=CARD_COLOR,
        anchor="w"
    )

    description_label.grid(
        row=1,
        column=0,
        sticky="w",
        padx=15,
        pady=(0, 12)
    )

    button = ttk.Button(
        card,
        text=button_text,
        command=command,
        width=24
    )

    button.grid(
        row=0,
        column=1,
        rowspan=2,
        padx=15,
        pady=12
    )

    return button


# ==========================================================
# MAIN WINDOW
# ==========================================================

root = tk.Tk()

root.title("CoBas Data Acquisition")

window_width = 980
window_height = 680

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_pos = int((screen_width / 2) - (window_width / 2))
y_pos = int((screen_height / 2) - (window_height / 2))

root.geometry(
    f"{window_width}x{window_height}+{x_pos}+{y_pos}"
)

root.resizable(False, False)

root.configure(
    bg=BG_COLOR
)

root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# ==========================================================
# STYLE
# ==========================================================

style = ttk.Style()

style.theme_use("clam")

style.configure(
    "TButton",
    font=("Arial", 10, "bold"),
    padding=8,
    background="#334155",
    foreground=TEXT_COLOR,
    borderwidth=0
)

style.map(
    "TButton",
    background=[
        ("active", "#475569"),
        ("disabled", "#1e293b")
    ],
    foreground=[
        ("disabled", "#64748b")
    ]
)


# ==========================================================
# HEADER
# ==========================================================

header_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

header_frame.pack(
    fill="x",
    padx=24,
    pady=(18, 8)
)

title_label = tk.Label(
    header_frame,
    text="CoBas Data Acquisition",
    font=("Arial", 24, "bold"),
    fg=TEXT_COLOR,
    bg=BG_COLOR
)

title_label.pack(
    anchor="w"
)

subtitle_label = tk.Label(
    header_frame,
    text="Shell interface for image/video capture, sound protocol generation, and sound processing.",
    font=("Arial", 10),
    fg=MUTED_TEXT_COLOR,
    bg=BG_COLOR
)

subtitle_label.pack(
    anchor="w",
    pady=(4, 0)
)


# ==========================================================
# MAIN PANEL
# ==========================================================

main_panel = tk.Frame(
    root,
    bg=PANEL_COLOR,
    highlightthickness=1,
    highlightbackground="#334155"
)

main_panel.pack(
    fill="x",
    padx=24,
    pady=10
)

main_panel.grid_columnconfigure(
    0,
    weight=1
)


# ==========================================================
# ACTION CARDS
# ==========================================================

image_button = create_action_card(
    main_panel,
    "Image / Video Capturing",
    SCRIPTS["Image / Video Capturing"]["description"],
    "Run Camera",
    lambda: run_script("Image / Video Capturing"),
    0
)

sound_generation_button = create_action_card(
    main_panel,
    "Sound Generation",
    SCRIPTS["Sound Generation"]["description"],
    "Generate Sound",
    lambda: run_script("Sound Generation"),
    1
)

sound_processing_button = create_action_card(
    main_panel,
    "Sound Processing",
    SCRIPTS["Sound Processing"]["description"],
    "Process Sound",
    lambda: run_script("Sound Processing"),
    2
)


# ==========================================================
# CONTROL PANEL
# ==========================================================

control_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

control_frame.pack(
    fill="x",
    padx=24,
    pady=(4, 8)
)

stop_button = ttk.Button(
    control_frame,
    text="Stop Running Script",
    command=stop_script,
    width=24,
    state="disabled"
)

stop_button.grid(
    row=0,
    column=0,
    padx=(0, 8)
)

clear_button = ttk.Button(
    control_frame,
    text="Clear Console",
    command=clear_console,
    width=18
)

clear_button.grid(
    row=0,
    column=1,
    padx=8
)

samples_button = ttk.Button(
    control_frame,
    text="Open Samples",
    command=lambda: open_folder(
        SCRIPTS["Image / Video Capturing"]["folder"]
    ),
    width=18
)

samples_button.grid(
    row=0,
    column=2,
    padx=8
)

inputs_button = ttk.Button(
    control_frame,
    text="Open Inputs",
    command=lambda: open_folder(
        SCRIPTS["Sound Generation"]["folder"]
    ),
    width=18
)

inputs_button.grid(
    row=0,
    column=3,
    padx=8
)

outputs_button = ttk.Button(
    control_frame,
    text="Open Outputs",
    command=lambda: open_folder(
        SCRIPTS["Sound Processing"]["folder"]
    ),
    width=18
)

outputs_button.grid(
    row=0,
    column=4,
    padx=8
)


# ==========================================================
# STATUS BAR
# ==========================================================

status_frame = tk.Frame(
    root,
    bg=PANEL_COLOR,
    highlightthickness=1,
    highlightbackground="#334155"
)

status_frame.pack(
    fill="x",
    padx=24,
    pady=(4, 8)
)

status_title_label = tk.Label(
    status_frame,
    text="Status:",
    font=("Arial", 10, "bold"),
    fg=TEXT_COLOR,
    bg=PANEL_COLOR
)

status_title_label.pack(
    side="left",
    padx=(12, 6),
    pady=8
)

status_value_label = tk.Label(
    status_frame,
    text="Ready",
    font=("Arial", 10, "bold"),
    fg=ACCENT_COLOR,
    bg=PANEL_COLOR
)

status_value_label.pack(
    side="left",
    pady=8
)


# ==========================================================
# CONSOLE PANEL
# ==========================================================

console_frame = tk.Frame(
    root,
    bg=PANEL_COLOR,
    highlightthickness=1,
    highlightbackground="#334155"
)

console_frame.pack(
    fill="both",
    expand=False,
    padx=24,
    pady=(0, 18)
)

console_title_label = tk.Label(
    console_frame,
    text="Console Output",
    font=("Arial", 11, "bold"),
    fg=TEXT_COLOR,
    bg=PANEL_COLOR
)

console_title_label.pack(
    anchor="w",
    padx=12,
    pady=(10, 4)
)

output_box = tk.Text(
    console_frame,
    height=13,
    width=115,
    bg=CONSOLE_BG,
    fg=CONSOLE_FG,
    insertbackground=TEXT_COLOR,
    font=("Consolas", 10),
    relief="flat",
    state="disabled",
    wrap="word"
)

output_box.pack(
    padx=12,
    pady=(0, 12)
)


# ==========================================================
# INITIAL CONSOLE MESSAGE
# ==========================================================

write_console("CoBas Data Acquisition Shell is ready.\n")
write_console(f"Base directory: {BASE_DIR}\n\n")
write_console("Available actions:\n")
write_console("1. Image / Video Capturing\n")
write_console("2. Sound Generation\n")
write_console("3. Sound Processing\n")


# ==========================================================
# START UI
# ==========================================================

root.mainloop()