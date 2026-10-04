"""Shared touchscreen styles; button heights use pixels through negative font sizes."""
from tkinter import ttk

WINDOW = {'width': 800, 'height': 480}
PREVIEW = {'width': 740, 'height': 250}
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
FONTS = {
    'toolbar_title': ('Arial', -20, 'bold'), 'toolbar_text': ('Arial', -16),
    'panel_title': ('Arial', -19, 'bold'), 'panel_text': ('Arial', -16),
    'status': ('Arial', -16, 'bold'), 'info': ('Arial', -15),
    'button': ('Arial', -18), 'button_bold': ('Arial', -18, 'bold'),
    'preview_text': ('Arial', -18, 'bold'),
}
SPACING = {'main_padx': 10, 'main_pady': 6, 'panel_padx': 10, 'panel_pady': 6,
           'toolbar_padx': 8, 'toolbar_pady': 4, 'button_pady': 4}


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
    style.configure('Number.TLabel', background=COLORS['preview_bg'], foreground=COLORS['accent'], font=('Arial', -30, 'bold'), padding=6)
    for name, color, bold in [('Start','start',True), ('Stop','stop',True), ('Tool','tool',False),
                              ('Settings','settings',False), ('Primary','primary',True), ('Danger','danger',True), ('Toolbar','tool',False)]:
        style.configure(f'{name}.TButton', font=FONTS['button_bold' if bold else 'button'],
                        padding=(12, 13), foreground=COLORS['text'], background=COLORS[color], borderwidth=0)
        style.map(f'{name}.TButton', background=[('disabled', '#273244'), ('pressed', COLORS[color+'_hover']), ('active', COLORS[color+'_hover'])],
                  foreground=[('disabled', COLORS['muted_text'])])
    style.configure('Touch.TNotebook', background=COLORS['main_bg'], borderwidth=0, tabmargins=(0,0,0,5))
    style.configure('Touch.TNotebook.Tab', font=FONTS['button_bold'], padding=(22,14), background=COLORS['tool'], foreground=COLORS['panel_text'])
    style.map('Touch.TNotebook.Tab', padding=[('selected', (22,14))], background=[('selected', COLORS['primary'])], foreground=[('selected', COLORS['text'])])
    style.configure('Vertical.TScrollbar', width=30, arrowsize=26)
    style.configure('ThermalScale.TRadiobutton', font=FONTS['button'], padding=(12,13), background=COLORS['panel_bg'], foreground=COLORS['text'])
    style.map('ThermalScale.TRadiobutton', foreground=[('disabled', COLORS['muted_text'])])
    return style
