import tkinter as tk
from tkinter import ttk

from Style import COLORS


def show_about_window(parent=None):
    """
    Show a clean Windows-style About dialog for CoBas Battery Reader.
    """

    about_window = tk.Toplevel(parent)
    about_window.title("About CoBas")
    about_window.geometry("500x350")
    about_window.resizable(False, False)
    about_window.configure(bg=COLORS["main_bg"])

    # Keep About window above the main app window.
    if parent is not None:
        about_window.transient(parent)
        about_window.grab_set()

    # --------------------------------------------------
    # Main Container
    # --------------------------------------------------

    main_frame = ttk.Frame(about_window, style="Panel.TFrame")
    main_frame.pack(fill="both", expand=True, padx=16, pady=16)

    # --------------------------------------------------
    # Header Section
    # --------------------------------------------------

    header_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    header_frame.pack(fill="x", pady=(0, 14))

    ttk.Label(
        header_frame,
        text="CoBas Battery Reader",
        style="PanelTitle.TLabel"
    ).pack(anchor="w")

    ttk.Label(
        header_frame,
        text="Version 1.0",
        style="PanelText.TLabel"
    ).pack(anchor="w", pady=(2, 0))

    # --------------------------------------------------
    # Description Section
    # --------------------------------------------------

    description_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    description_frame.pack(fill="x", pady=(0, 14))

    description = (
        "CoBas is a contactless battery sensing prototype that combines "
        "near-ultrasonic acoustic signals, IWR6843AOP range-angle responses, "
        "and synchronized thermal imaging to support lithium-ion battery "
        "state-of-charge monitoring."
    )

    ttk.Label(
        description_frame,
        text=description,
        style="PanelText.TLabel",
        wraplength=450,
        justify="left"
    ).pack(anchor="w")

    # --------------------------------------------------
    # Authors Section
    # --------------------------------------------------

    authors_section = ttk.Frame(main_frame, style="Panel.TFrame")
    authors_section.pack(fill="x", pady=(0, 14))

    ttk.Label(
        authors_section,
        text="Authors",
        style="PanelTitle.TLabel"
    ).pack(anchor="w", pady=(0, 6))

    authors_frame = ttk.Frame(authors_section, style="Panel.TFrame")
    authors_frame.pack(fill="x")

    authors_left = (
        "Deniz Karya Acikbas\n"
        "Pedro Callado de Paiva\n"
        "Selase Doku\n"
        "Nitin Shankar Madhu\n"
        "Chuka Ezeoke\n"
        "Clark Friese"
    )

    authors_right = (
        "Ahmad Jayeb\n"
        "Lucas Hammermeister\n"
        "Ningyue Mao\n"
        "Xuan Zhou\n"
        "Xiao Zhang"
    )

    ttk.Label(
        authors_frame,
        text=authors_left,
        style="PanelText.TLabel",
        justify="left"
    ).pack(side="left", anchor="nw", expand=True, fill="x")

    ttk.Label(
        authors_frame,
        text=authors_right,
        style="PanelText.TLabel",
        justify="left"
    ).pack(side="left", anchor="nw", expand=True, fill="x")

    # --------------------------------------------------
    # Lab / Copyright Section
    # --------------------------------------------------

    footer_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    footer_frame.pack(fill="x", pady=(0, 10))

    copyright_text = (
        "© 2026 Trustworthy AIoT Lab, University of Michigan-Dearborn.\n"
        "All rights reserved."
    )

    ttk.Label(
        footer_frame,
        text=copyright_text,
        style="PanelText.TLabel",
        wraplength=450,
        justify="left"
    ).pack(anchor="w")

    # --------------------------------------------------
    # Button Row
    # --------------------------------------------------

    button_frame = ttk.Frame(main_frame, style="Panel.TFrame")
    button_frame.pack(fill="x", side="bottom")

    ttk.Button(
        button_frame,
        text="OK",
        style="Settings.TButton",
        command=about_window.destroy
    ).pack(side="right")
