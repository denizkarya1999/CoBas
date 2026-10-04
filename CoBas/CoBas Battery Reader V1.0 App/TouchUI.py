"""Touch-sized controls and dialogs for the 800 x 480 CoBas display."""
import math
import tkinter as tk
from tkinter import ttk
from Style import COLORS, FONTS


class TouchDialog(tk.Toplevel):
    def __init__(self, parent, title, width=680, height=410):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.configure(bg=COLORS['panel_bg'])
        self.overrideredirect(True)
        self.parent = parent
        self.previous_grab = self.grab_current()
        self.result = None
        if parent.winfo_viewable():
            self.transient(parent)
        pw = parent.winfo_width() if parent.winfo_viewable() else 800
        ph = parent.winfo_height() if parent.winfo_viewable() else 480
        width, height = min(width, pw), min(height, ph)
        x = parent.winfo_x() + max(0, (pw - width) // 2)
        y = parent.winfo_y() + max(0, (ph - height) // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Escape>', lambda event: self.cancel())
        self.body = ttk.Frame(self, style='Panel.TFrame', padding=12)
        self.body.pack(fill='both', expand=True)
        ttk.Label(self.body, text=title, style='PanelTitle.TLabel').pack(anchor='w', pady=(0, 8))

    def show(self):
        self.deiconify()
        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.wait_window()
        return self.result

    def finish(self, result=None):
        self.result = result
        self.grab_release()
        self.destroy()
        if self.previous_grab is not None and self.previous_grab.winfo_exists():
            self.previous_grab.grab_set()

    def cancel(self):
        self.finish(None)


def scroll_text(parent, text, height=6):
    frame = ttk.Frame(parent, style='Panel.TFrame')
    frame.pack(fill='both', expand=True, pady=(0, 8))
    view = tk.Text(frame, wrap='word', font=FONTS['panel_text'], height=height,
                   bg=COLORS['panel_bg'], fg=COLORS['text'], bd=0,
                   highlightthickness=0, padx=4, pady=4)
    bar = ttk.Scrollbar(frame, command=view.yview)
    bar.pack(side='right', fill='y')
    view.pack(side='left', fill='both', expand=True)
    view.configure(yscrollcommand=bar.set)
    view.insert('1.0', text)
    view.configure(state='disabled')
    view.bind('<ButtonPress-1>', lambda e: view.scan_mark(e.x, e.y))
    view.bind('<B1-Motion>', lambda e: view.scan_dragto(e.x, e.y))
    return view


class NumberDialog(TouchDialog):
    def __init__(self, parent, title, initial='', minimum=0, maximum=100, decimal=False):
        super().__init__(parent, title, width=500, height=456)
        self.minimum, self.maximum, self.decimal = minimum, maximum, decimal
        self.value = tk.StringVar(value=str(initial))
        self.replace_next = True
        self.error = tk.StringVar(value=f'Enter {minimum:g} to {maximum:g}' if decimal else f'Enter {minimum} to {maximum}')
        ttk.Label(self.body, textvariable=self.value, style='Number.TLabel', anchor='e').pack(fill='x')
        ttk.Label(self.body, textvariable=self.error, style='PanelText.TLabel').pack(fill='x', pady=4)
        keys = ttk.Frame(self.body, style='Panel.TFrame')
        keys.pack(fill='both', expand=True)
        layout = [('7','8','9','Delete'), ('4','5','6','Clear'), ('1','2','3','.' if decimal else ''), ('0','0','','')]
        for c in range(4):
            keys.columnconfigure(c, weight=1, uniform='key')
        for r, row in enumerate(layout):
            keys.rowconfigure(r, weight=1, minsize=48)
            for c, key in enumerate(row):
                if not key or (r == 3 and c == 1):
                    continue
                ttk.Button(keys, text=key, style='Tool.TButton', command=lambda k=key: self.type_key(k)).grid(
                    row=r, column=c, columnspan=2 if r == 3 else 1, sticky='nsew', padx=3, pady=3)
        actions = ttk.Frame(self.body, style='Panel.TFrame')
        actions.pack(fill='x', pady=(8, 0))
        ttk.Button(actions, text='Cancel', style='Tool.TButton', command=self.cancel).pack(side='left', fill='x', expand=True, padx=(0,4))
        ttk.Button(actions, text='Done', style='Primary.TButton', command=self.accept).pack(side='left', fill='x', expand=True, padx=(4,0))
        self.bind('<Key>', self.keyboard)

    def keyboard(self, event):
        if event.keysym in ('Return', 'KP_Enter'):
            self.accept()
        elif event.keysym == 'BackSpace':
            self.type_key('⌫')
        elif event.char in '0123456789.' and event.char:
            self.type_key(event.char)

    def type_key(self, key):
        current = self.value.get()
        if key == 'Clear':
            current = ''
        elif key in ('⌫', 'Delete'):
            current = current[:-1]
        elif key == '.' and (not self.decimal or ('.' in current and not self.replace_next)):
            return
        else:
            current = ('' if self.replace_next else current) + key
        self.replace_next = False
        self.value.set(current[:10])

    def accept(self):
        try:
            text = self.value.get()
            value = float(text) if self.decimal else int(text)
            if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
                raise ValueError
        except ValueError:
            self.error.set(f'Use a number from {self.minimum:g} to {self.maximum:g}.')
            return
        self.finish(value)


def ask_number(parent, title, **kwargs):
    return NumberDialog(parent, title, **kwargs).show()


class NumberControl(ttk.Frame):
    def __init__(self, parent, variable, title, minimum=1, maximum=999, decimal=False, step=1):
        super().__init__(parent, style='Panel.TFrame')
        self.variable, self.title = variable, title
        self.minimum, self.maximum, self.decimal, self.step = minimum, maximum, decimal, step
        self.disabled = False
        self.columnconfigure(1, weight=1)
        self.minus = ttk.Button(self, text='−', width=3, style='Tool.TButton', command=lambda: self.adjust(-step))
        self.number = ttk.Button(self, textvariable=variable, width=5, style='Tool.TButton', command=self.edit)
        self.plus = ttk.Button(self, text='+', width=3, style='Tool.TButton', command=lambda: self.adjust(step))
        for c, button in enumerate((self.minus, self.number, self.plus)):
            button.grid(row=0, column=c, sticky='ew', padx=3)

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf or {}, **kwargs)
        if 'state' in options:
            state = options.pop('state')
            self.disabled = state == 'disabled'
            for button in (self.minus, self.number, self.plus):
                button.configure(state=state)
        if options:
            super().configure(**options)

    config = configure

    def adjust(self, amount):
        if self.disabled:
            return
        try:
            value = float(self.variable.get())
        except ValueError:
            value = self.minimum
        value = max(self.minimum, min(self.maximum, value + amount))
        self.variable.set(f'{value:g}' if self.decimal else str(int(value)))

    def edit(self):
        if self.disabled:
            return
        value = ask_number(self.winfo_toplevel(), self.title, initial=self.variable.get(), minimum=self.minimum,
                           maximum=self.maximum, decimal=self.decimal)
        if value is not None:
            self.variable.set(f'{value:g}' if self.decimal else str(value))


class TouchMessages:
    @staticmethod
    def _show(title, message, parent=None, choice=None):
        dialog = TouchDialog(parent or tk._default_root, title)
        scroll_text(dialog.body, message)
        actions = ttk.Frame(dialog.body, style='Panel.TFrame')
        actions.pack(fill='x')
        if choice:
            ttk.Button(actions, text='No' if choice == 'yesno' else 'Cancel', style='Tool.TButton',
                       command=lambda: dialog.finish(False)).pack(side='left', fill='x', expand=True, padx=(0,6))
        ttk.Button(actions, text='Yes' if choice == 'yesno' else 'OK', style='Primary.TButton',
                   command=lambda: dialog.finish(True)).pack(side='right', fill='x', expand=True)
        return dialog.show()

    showinfo = _show
    showwarning = _show
    showerror = _show

    def askokcancel(self, title, message, parent=None):
        return bool(self._show(title, message, parent, 'okcancel'))

    def askyesno(self, title, message, parent=None):
        return bool(self._show(title, message, parent, 'yesno'))


dialogs = TouchMessages()
