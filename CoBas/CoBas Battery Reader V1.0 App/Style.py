from tkinter import ttk


# --------------------------------------------------
# Window Settings
# --------------------------------------------------
# Fixed size of the main application window.
WINDOW = {
    "width": 980,
    "height": 620
}


# --------------------------------------------------
# Preview Settings
# --------------------------------------------------
# Display size of each live sensor preview inside the GUI.
PREVIEW = {
    "width": 660,
    "height": 380
}


# --------------------------------------------------
# Colors
# --------------------------------------------------
# Central color palette used across the application.
COLORS = {
    # Main backgrounds
    "main_bg": "#0b1120",
    "toolbar_bg": "#020617",
    "panel_bg": "#111827",
    "panel_bg_light": "#1f2937",
    "preview_bg": "#020617",

    # Text colors
    "text": "#f8fafc",
    "muted_text": "#94a3b8",
    "panel_text": "#cbd5e1",
    "info_text": "#e5e7eb",

    # Status colors
    "accent": "#38bdf8",
    "success": "#22c55e",
    "warning": "#facc15",
    "error": "#f87171",
    "idle": "#94a3b8",
    "record": "#ef4444",

    # General button colors
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "danger": "#dc2626",
    "danger_hover": "#b91c1c",
    "tool": "#1f2937",
    "tool_hover": "#2563eb",

    # Start / Stop Tracking colors
    "start": "#16a34a",
    "start_hover": "#15803d",
    "stop": "#dc2626",
    "stop_hover": "#b91c1c",

    # Toolbar button colors
    "settings": "#4f46e5",
    "settings_hover": "#4338ca"
}


# --------------------------------------------------
# Fonts
# --------------------------------------------------
# Central font settings.
FONTS = {
    "toolbar_title": ("Arial", 12, "bold"),
    "toolbar_text": ("Arial", 8),

    "panel_title": ("Arial", 10, "bold"),
    "panel_text": ("Arial", 7),

    "status": ("Arial", 8, "bold"),
    "info": ("Arial", 7),

    "button": ("Arial", 8),
    "button_bold": ("Arial", 8, "bold"),

    "preview_text": ("Arial", 13, "bold")
}


# --------------------------------------------------
# Spacing
# --------------------------------------------------
# Central spacing values used by the GUI.
SPACING = {
    "main_padx": 8,
    "main_pady": 6,

    "panel_padx": 9,
    "panel_pady": 4,

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

    This file works like a CSS-style configuration file
    for the Tkinter app.
    """

    style = ttk.Style()

    # The "clam" theme allows better color customization for ttk widgets.
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Set main root background.
    root.configure(bg=COLORS["main_bg"])

    # --------------------------------------------------
    # Frames
    # --------------------------------------------------

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

    style.configure(
        "SoftPanel.TFrame",
        background=COLORS["panel_bg_light"]
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

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
        "Info.TLabel",
        background=COLORS["panel_bg"],
        foreground=COLORS["info_text"],
        font=FONTS["info"]
    )

    # --------------------------------------------------
    # Helper Function for Colored Buttons
    # --------------------------------------------------

    def configure_colored_button(style_name, normal_color, hover_color, bold=False):
        """
        Create a colored ttk button style.

        Parameters:
            style_name:
                Name of the ttk style.

            normal_color:
                Default button color.

            hover_color:
                Button color when active or pressed.

            bold:
                Whether to use bold button font.
        """

        style.configure(
            style_name,
            font=FONTS["button_bold"] if bold else FONTS["button"],
            padding=(6, 4),
            foreground=COLORS["text"],
            background=normal_color,
            borderwidth=0,
            focusthickness=0
        )

        style.map(
            style_name,
            foreground=[
                ("active", COLORS["text"]),
                ("pressed", COLORS["text"])
            ],
            background=[
                ("active", hover_color),
                ("pressed", hover_color)
            ]
        )

    # --------------------------------------------------
    # Tracking Button Styles
    # --------------------------------------------------

    # Green Start Tracking button.
    configure_colored_button(
        "Start.TButton",
        COLORS["start"],
        COLORS["start_hover"],
        bold=True
    )

    # Red Stop Tracking button.
    configure_colored_button(
        "Stop.TButton",
        COLORS["stop"],
        COLORS["stop_hover"],
        bold=True
    )

    # --------------------------------------------------
    # General Button Styles
    # --------------------------------------------------

    # Normal dark/blue tool button.
    configure_colored_button(
        "Tool.TButton",
        COLORS["tool"],
        COLORS["tool_hover"]
    )

    # Indigo settings/about toolbar buttons.
    configure_colored_button(
        "Settings.TButton",
        COLORS["settings"],
        COLORS["settings_hover"]
    )

    # Optional general primary button.
    configure_colored_button(
        "Primary.TButton",
        COLORS["primary"],
        COLORS["primary_hover"],
        bold=True
    )

    # Optional general danger button.
    configure_colored_button(
        "Danger.TButton",
        COLORS["danger"],
        COLORS["danger_hover"],
        bold=True
    )

    # Optional toolbar fallback button.
    configure_colored_button(
        "Toolbar.TButton",
        COLORS["tool"],
        COLORS["tool_hover"]
    )

    # --------------------------------------------------
    # Combobox
    # --------------------------------------------------

    style.configure(
        "TCombobox",
        padding=3
    )

    style.configure(
        "ThermalScale.TRadiobutton",
        background=COLORS["panel_bg"],
        foreground=COLORS["panel_text"],
        font=FONTS["panel_text"],
        padding=(1, 0)
    )
    style.map(
        "ThermalScale.TRadiobutton",
        background=[
            ("active", COLORS["panel_bg"]),
            ("disabled", COLORS["panel_bg"])
        ],
        foreground=[
            ("active", COLORS["text"]),
            ("disabled", COLORS["muted_text"])
        ]
    )

    return style
