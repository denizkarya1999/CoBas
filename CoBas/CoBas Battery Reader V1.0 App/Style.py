"""Shared touchscreen styles; button heights use pixels through negative font sizes.

The layout was designed at 800 x 480. configure_for_display() rescales every
font, spacing and preview size to the touch panel that is attached, and px()
scales any other pixel size written for the 800 x 480 design.
"""
from tkinter import ttk

BASE_WINDOW = {'width': 800, 'height': 480}
WINDOW = {'width': 800, 'height': 480, 'x': 0, 'y': 0}
PREVIEW = {'width': 740, 'height': 250}
SCALE = 1.0
COLORS = {
    'main_bg': '#0b1120', 'toolbar_bg': '#020617', 'panel_bg': '#111827',
    'panel_bg_light': '#1f2937', 'preview_bg': '#020617', 'text': '#f8fafc',
    'muted_text': '#94a3b8', 'panel_text': '#cbd5e1', 'info_text': '#e5e7eb',
    'accent': '#38bdf8', 'success': '#22c55e', 'warning': '#facc15',
    'error': '#f87171', 'idle': '#94a3b8', 'record': '#ef4444',
    'primary': '#2563eb', 'primary_hover': '#1d4ed8', 'danger': '#dc2626',
    'danger_hover': '#b91c1c', 'tool': '#1f2937', 'tool_hover': '#334155',
    'start': '#15803d', 'start_hover': '#166534', 'stop': '#dc2626',
    'stop_hover': '#b91c1c', 'settings': '#4f46e5', 'settings_hover': '#4338ca',
}
_BASE_FONTS = {
    'toolbar_title': ('Arial', -20, 'bold'), 'toolbar_text': ('Arial', -16),
    'panel_title': ('Arial', -19, 'bold'), 'panel_text': ('Arial', -16),
    'status': ('Arial', -16, 'bold'), 'info': ('Arial', -15),
    'button': ('Arial', -18), 'button_bold': ('Arial', -18, 'bold'),
    'preview_text': ('Arial', -18, 'bold'),
}
_BASE_SPACING = {'main_padx': 10, 'main_pady': 6, 'panel_padx': 10, 'panel_pady': 6,
                 'toolbar_padx': 8, 'toolbar_pady': 4, 'button_pady': 4}
FONTS = dict(_BASE_FONTS)
SPACING = dict(_BASE_SPACING)


def px(value):
    """Scale a pixel size written for the 800 x 480 design to the current display."""
    return max(1, round(value * SCALE))


def is_compact():
    """True on small panels (below three quarters of the design size), where optional labels and some padding go."""
    return SCALE < 0.75


def configure_for_display(width, height, x=0, y=0):
    """Size the layout for a display of width x height placed at x, y; call before building widgets."""
    global SCALE
    SCALE = max(0.5, min(width / BASE_WINDOW['width'], height / BASE_WINDOW['height'], 1.5))
    WINDOW.update(width=width, height=height, x=x, y=y)
    PREVIEW.update(width=px(740), height=px(250))
    for name, (family, size, *weight) in _BASE_FONTS.items():
        FONTS[name] = (family, -max(9, round(-size * SCALE)), *weight)
    for name, value in _BASE_SPACING.items():
        SPACING[name] = max(2, round(value * SCALE))
    return SCALE


def apply_styles(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    root.configure(bg=COLORS['main_bg'])
    for name, color in [('Main', 'main_bg'), ('Toolbar', 'toolbar_bg'), ('Panel', 'panel_bg'), ('SoftPanel', 'panel_bg_light')]:
        style.configure(f'{name}.TFrame', background=COLORS[color])
    for name, bg, fg, font in [('ToolbarTitle', 'toolbar_bg', 'text', 'toolbar_title'),
            ('ToolbarText', 'toolbar_bg', 'muted_text', 'toolbar_text'),
            ('PanelTitle', 'panel_bg', 'text', 'panel_title'),
            ('PanelText', 'panel_bg', 'panel_text', 'panel_text'), ('Info', 'panel_bg', 'info_text', 'info')]:
        style.configure(f'{name}.TLabel', background=COLORS[bg], foreground=COLORS[fg], font=FONTS[font])
    style.configure('Number.TLabel', background=COLORS['preview_bg'], foreground=COLORS['accent'], font=('Arial', -px(30), 'bold'), padding=px(6))
    button_padding = (px(12), px(11 if is_compact() else 13))
    for name, color, bold in [('Start','start',True), ('Stop','stop',True), ('Tool','tool',False),
                              ('Settings','settings',False), ('Primary','primary',True), ('Danger','danger',True), ('Toolbar','tool',False)]:
        style.configure(f'{name}.TButton', font=FONTS['button_bold' if bold else 'button'],
                        padding=button_padding, foreground=COLORS['text'], background=COLORS[color], borderwidth=0)
        style.map(f'{name}.TButton', background=[('disabled', '#273244'), ('pressed', COLORS[color+'_hover']), ('active', COLORS[color+'_hover'])],
                  foreground=[('disabled', COLORS['muted_text'])])
    tab_padding = (px(22), px(10 if is_compact() else 14))
    style.configure('Touch.TNotebook', background=COLORS['main_bg'], borderwidth=0, tabmargins=(0,0,0,px(5)))
    style.configure('Touch.TNotebook.Tab', font=FONTS['button_bold'], padding=tab_padding, background=COLORS['tool'], foreground=COLORS['panel_text'])
    style.map('Touch.TNotebook.Tab', padding=[('selected', tab_padding)], background=[('selected', COLORS['primary'])], foreground=[('selected', COLORS['text'])])
    style.configure('Vertical.TScrollbar', width=px(30), arrowsize=px(26))
    style.configure('ThermalScale.TRadiobutton', font=FONTS['button'], padding=button_padding, background=COLORS['panel_bg'], foreground=COLORS['text'])
    style.map('ThermalScale.TRadiobutton', foreground=[('disabled', COLORS['muted_text'])])
    return style
