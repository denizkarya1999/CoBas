"""Touch-sized microphone and speaker chooser with paged device buttons."""
from tkinter import ttk
from TouchUI import TouchDialog, dialogs


class SettingsWindow:
    PAGE_SIZE = 2

    def __init__(self, parent, app):
        self.app = app
        self.dialog = TouchDialog(parent, 'Audio settings', width=740, height=440)
        self.window = self.dialog
        self.kind = 'input'
        self.page = 0
        self.selected = {
            'input': app.audio.microphone_device_id,
            'output': app.audio.speaker_device_id,
        }
        self.devices = {'input': [], 'output': []}
        modes = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        modes.pack(fill='x', pady=(0, 6))
        self.mode_buttons = {}
        for kind, title in (('input', 'Microphone'), ('output', 'Speaker')):
            button = ttk.Button(modes, text=title, style='Tool.TButton',
                                command=lambda k=kind: self.change_kind(k))
            button.pack(side='left', fill='x', expand=True, padx=3)
            self.mode_buttons[kind] = button
        ttk.Button(modes, text='Refresh', style='Tool.TButton', command=self.refresh).pack(side='right', padx=3)
        self.hint = ttk.Label(self.dialog.body, style='PanelText.TLabel')
        self.hint.pack(anchor='w', pady=(0, 6))
        self.choices = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        self.choices.pack(fill='both', expand=True)
        page_row = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        page_row.pack(fill='x', pady=6)
        self.previous = ttk.Button(page_row, text='Previous', style='Tool.TButton', command=lambda: self.change_page(-1))
        self.previous.pack(side='left')
        self.page_label = ttk.Label(page_row, style='PanelText.TLabel', anchor='center')
        self.page_label.pack(side='left', fill='x', expand=True)
        self.next = ttk.Button(page_row, text='Next', style='Tool.TButton', command=lambda: self.change_page(1))
        self.next.pack(side='right')
        actions = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        actions.pack(fill='x')
        ttk.Button(actions, text='Cancel', style='Tool.TButton', command=self.dialog.cancel).pack(side='left', fill='x', expand=True, padx=(0, 4))
        ttk.Button(actions, text='Apply audio', style='Primary.TButton', command=self.apply).pack(side='left', fill='x', expand=True, padx=(4, 0))
        self.refresh()
        self.dialog.show()

    def refresh(self):
        try:
            devices = {
                'input': self.app.audio.get_input_microphones(),
                'output': self.app.audio.get_output_speakers(),
            }
        except Exception as error:
            dialogs.showwarning('Audio unavailable', str(error), parent=self.dialog)
            return False
        self.devices = devices
        self.render()
        return True

    def change_kind(self, kind):
        self.kind = kind
        self.page = 0
        self.render()

    def change_page(self, delta):
        self.page += delta
        self.render()

    def select(self, device_id):
        self.selected[self.kind] = device_id
        self.render()

    def render(self):
        for child in self.choices.winfo_children():
            child.destroy()
        devices = self.devices[self.kind]
        pages = max(1, (len(devices) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page = max(0, min(self.page, pages - 1))
        for kind, button in self.mode_buttons.items():
            button.configure(style='Primary.TButton' if kind == self.kind else 'Tool.TButton')
        self.hint.configure(text='Choose the microphone for recording.' if self.kind == 'input'
                            else 'Choose the speaker for chirp playback.')
        for device in devices[self.page*self.PAGE_SIZE:(self.page+1)*self.PAGE_SIZE]:
            selected = device['id'] == self.selected[self.kind]
            label = ('✓  ' if selected else '') + device['name']
            if len(label) > 64:
                label = label[:61] + '…'
            ttk.Button(self.choices, text=label, style='Primary.TButton' if selected else 'Tool.TButton',
                       command=lambda i=device['id']: self.select(i)).pack(fill='x', pady=3)
        self.page_label.configure(text=f'{self.page + 1} / {pages}')
        self.previous.configure(state='disabled' if self.page == 0 else 'normal')
        self.next.configure(state='disabled' if self.page == pages - 1 else 'normal')

    def apply(self):
        if self.app.pulse_sequence_active:
            dialogs.showwarning('Capture active', 'Stop tracking before changing audio devices.', parent=self.dialog)
            return
        if not self.refresh():
            return
        choices = {}
        for kind in ('input', 'output'):
            device = next((item for item in self.devices[kind] if item['id'] == self.selected[kind]), None)
            if device is None:
                self.change_kind(kind)
                title = 'Microphone' if kind == 'input' else 'Speaker'
                dialogs.showwarning(f'{title} unavailable', f'Select an available {title.lower()} or Automatic USB.', parent=self.dialog)
                return
            choices[kind] = device
        if self.app.apply_audio_sources_from_settings(choices['input'], choices['output']):
            self.dialog.finish(True)
