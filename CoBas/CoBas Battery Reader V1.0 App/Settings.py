"""Touch-sized microphone chooser with paged device buttons."""
from tkinter import ttk
from TouchUI import TouchDialog, dialogs


class SettingsWindow:
    def __init__(self, parent, app):
        self.app = app
        self.dialog = TouchDialog(parent, 'Microphone settings', width=740, height=440)
        self.window = self.dialog
        self.microphones = app.audio.get_input_microphones()
        self.page = 0
        self.selected_id = app.audio.microphone_device_id
        ttk.Label(self.dialog.body, text='Choose the microphone used to record each chirp.', style='PanelText.TLabel').pack(anchor='w', pady=(0,8))
        self.choices = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        self.choices.pack(fill='both', expand=True)
        self.page_row = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        self.page_row.pack(fill='x', pady=6)
        self.previous = ttk.Button(self.page_row, text='Previous', style='Tool.TButton', command=lambda: self.change_page(-1))
        self.previous.pack(side='left')
        self.page_label = ttk.Label(self.page_row, style='PanelText.TLabel', anchor='center')
        self.page_label.pack(side='left', fill='x', expand=True)
        self.next = ttk.Button(self.page_row, text='Next', style='Tool.TButton', command=lambda: self.change_page(1))
        self.next.pack(side='right')
        actions = ttk.Frame(self.dialog.body, style='Panel.TFrame')
        actions.pack(fill='x')
        ttk.Button(actions, text='Cancel', style='Tool.TButton', command=self.dialog.cancel).pack(side='left', fill='x', expand=True, padx=(0,4))
        ttk.Button(actions, text='Apply microphone', style='Primary.TButton', command=self.apply).pack(side='left', fill='x', expand=True, padx=(4,0))
        self.render()
        self.dialog.show()

    def change_page(self, delta):
        self.page += delta
        self.render()

    def select(self, device_id):
        self.selected_id = device_id
        self.render()

    def render(self):
        for child in self.choices.winfo_children():
            child.destroy()
        pages = max(1, (len(self.microphones) + 2) // 3)
        self.page = max(0, min(self.page, pages - 1))
        for device in self.microphones[self.page*3:self.page*3+3]:
            selected = device['id'] == self.selected_id
            name = device['name']
            label = ('✓  ' if selected else '') + name
            if len(label) > 64:
                label = label[:61] + '…'
            ttk.Button(self.choices, text=label, style='Primary.TButton' if selected else 'Tool.TButton',
                       command=lambda i=device['id']: self.select(i)).pack(fill='x', pady=3)
        self.page_label.configure(text=f'{self.page + 1} / {pages}')
        self.previous.configure(state='disabled' if self.page == 0 else 'normal')
        self.next.configure(state='disabled' if self.page == pages - 1 else 'normal')

    def apply(self):
        if self.app.pulse_sequence_active:
            dialogs.showwarning('Capture active', 'Stop tracking before changing the microphone.', parent=self.dialog)
            return
        device = next((item for item in self.microphones if item['id'] == self.selected_id), None)
        if device is None:
            self.selected_id = None
            self.render()
            dialogs.showwarning('Microphone unavailable', 'Select an available microphone before applying.', parent=self.dialog)
            return
        self.app.apply_microphone_source_from_settings(device['id'], device['name'])
        self.dialog.finish(True)
