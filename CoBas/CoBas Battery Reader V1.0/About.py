from tkinter import messagebox


def show_about_window():
    """
    Show the About popup for CoBas_V1.
    """

    messagebox.showinfo(
        "About CoBas_V1",
        "CoBas_V1\n\n"
        "Camera-based battery reader prototype.\n\n"
        "Current features:\n"
        "- Automatic camera initialization on startup\n"
        "- Fixed-size dashboard GUI\n"
        "- Toolbar with Settings and About\n"
        "- Settings-based camera source selection\n"
        "- Live camera preview\n"
        "- Restart Tracking\n"
        "- Stop Tracking\n"
        "- Zoom In / Zoom Out / Reset Zoom\n"
        "- Flip Camera\n"
        "- Take Photo\n"
        "- Record Video with seconds timer\n\n"
        "Code structure:\n"
        "- CoBas_V1.py handles the main GUI\n"
        "- Settings.py handles camera source settings\n"
        "- About.py handles the About popup\n"
        "- Style.py handles GUI styles\n"
        "- Camera.py handles camera operations\n\n"
        "Future version:\n"
        "- Battery detection\n"
        "- ML model inference\n"
        "- Battery percentage estimation\n"
        "- Sound/video fusion"
    )