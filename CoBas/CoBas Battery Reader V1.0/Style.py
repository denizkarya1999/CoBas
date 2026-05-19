from tkinter import ttk


# --------------------------------------------------
# Window Settings
# --------------------------------------------------

WINDOW = {
    "width": 980,
    "height": 620
}


# --------------------------------------------------
# Preview Settings
# --------------------------------------------------

PREVIEW = {
    "width": 690,
    "height": 420
}


# --------------------------------------------------
# Colors
# --------------------------------------------------

COLORS = {
    "main_bg": "#0f172a",
    "toolbar_bg": "#020617",
    "panel_bg": "#111827",
    "preview_bg": "#020617",

    "text": "#f8fafc",
    "muted_text": "#94a3b8",
    "panel_text": "#cbd5e1",
    "info_text": "#e5e7eb",

    "accent": "#38bdf8",
    "error": "#f87171"
}


# --------------------------------------------------
# Fonts
# --------------------------------------------------

FONTS = {
    "header": ("Arial", 18, "bold"),
    "subheader": ("Arial", 9),

    "toolbar_title": ("Arial", 11, "bold"),
    "toolbar_text": ("Arial", 8),

    "panel_title": ("Arial", 11, "bold"),
    "panel_text": ("Arial", 8),

    "status": ("Arial", 9, "bold"),
    "info": ("Arial", 8),

    "button": ("Arial", 8),
    "button_bold": ("Arial", 8, "bold"),

    "preview_text": ("Arial", 13, "bold")
}


# --------------------------------------------------
# Spacing
# --------------------------------------------------

SPACING = {
    "main_padx": 10,
    "main_pady": 8,

    "panel_padx": 10,
    "panel_pady": 6,

    "toolbar_padx": 10,
    "toolbar_pady": 4,

    "button_pady": 2
}


# --------------------------------------------------
# Style Function
# --------------------------------------------------

def apply_styles(root):
    """
    Apply all Tkinter/ttk styles.

    This file works like a CSS file for the Tkinter app.
    """

    style = ttk.Style()

    try:
        style.theme_use("clam")
    except Exception:
        pass

    root.configure(bg=COLORS["main_bg"])

    # Frames.
    style.configure(
        "Main.TFrame",
        background=COLORS["main_bg"]
    )

    style.configure(
        "Toolbar.TFrame",
        background=COLORS["toolbar_bg"]
    )

    style.configure(
        "Panel.TFrame",
        background=COLORS["panel_bg"]
    )

    # Labels.
    style.configure(
        "Header.TLabel",
        background=COLORS["main_bg"],
        foreground=COLORS["text"],
        font=FONTS["header"]
    )

    style.configure(
        "SubHeader.TLabel",
        background=COLORS["main_bg"],
        foreground=COLORS["muted_text"],
        font=FONTS["subheader"]
    )

    style.configure(
        "ToolbarTitle.TLabel",
        background=COLORS["toolbar_bg"],
        foreground=COLORS["text"],
        font=FONTS["toolbar_title"]
    )

    style.configure(
        "ToolbarText.TLabel",
        background=COLORS["toolbar_bg"],
        foreground=COLORS["muted_text"],
        font=FONTS["toolbar_text"]
    )

    style.configure(
        "PanelTitle.TLabel",
        background=COLORS["panel_bg"],
        foreground=COLORS["text"],
        font=FONTS["panel_title"]
    )

    style.configure(
        "PanelText.TLabel",
        background=COLORS["panel_bg"],
        foreground=COLORS["panel_text"],
        font=FONTS["panel_text"]
    )

    style.configure(
        "Status.TLabel",
        background=COLORS["panel_bg"],
        foreground=COLORS["accent"],
        font=FONTS["status"]
    )

    style.configure(
        "Info.TLabel",
        background=COLORS["panel_bg"],
        foreground=COLORS["info_text"],
        font=FONTS["info"]
    )

    # Buttons.
    style.configure(
        "Primary.TButton",
        font=FONTS["button_bold"],
        padding=5
    )

    style.configure(
        "Danger.TButton",
        font=FONTS["button_bold"],
        padding=5
    )

    style.configure(
        "Tool.TButton",
        font=FONTS["button"],
        padding=4
    )

    style.configure(
        "Toolbar.TButton",
        font=FONTS["button"],
        padding=4
    )

    # Combobox.
    style.configure(
        "TCombobox",
        padding=3
    )

    return style