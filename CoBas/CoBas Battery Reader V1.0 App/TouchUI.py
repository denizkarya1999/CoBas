"""Touch-sized controls and dialogs for the CoBas touch display (designed at 800 x 480, scaled by Style.px)."""
import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from Style import BASE_WINDOW, COLORS, DESIGNS, FONTS, WINDOW, px


def chars_for(width, font='button'):
    """How many average characters of a font fit in width pixels, for labels that cannot wrap (buttons)."""
    sample = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    per_char = tkfont.Font(font=FONTS[font]).measure(sample) / len(sample)
    return max(8, int(width / per_char))


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
        if parent.winfo_viewable():
            pw, ph, left, top = parent.winfo_width(), parent.winfo_height(), parent.winfo_x(), parent.winfo_y()
        else:
            pw, ph, left, top = WINDOW['width'], WINDOW['height'], WINDOW['x'], WINDOW['y']
        # Sizes are written for the 800 x 480 design. On a portrait panel a dialog
        # is as wide as the panel and keeps the share of the height it was designed
        # with: one that covers most of a landscape panel covers most of a portrait one.
        height = round(height * BASE_WINDOW['height'] / DESIGNS['landscape']['height'])
        width, height = min(px(width), pw), min(px(height), ph)
        self.parent_box = (pw, ph, left, top)
        self.size = (width, height)
        self.place_dialog()
        self.text_width = width - 2 * px(12) - px(6)  # what a wrapped label in the body may span
        self.button_chars = chars_for(self.text_width - 2 * px(12))  # what fits on a full-width button
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Escape>', lambda event: self.cancel())
        self.body = ttk.Frame(self, style='Panel.TFrame', padding=px(12))
        self.body.pack(fill='both', expand=True)
        ttk.Label(self.body, text=title, style='PanelTitle.TLabel').pack(anchor='w', pady=(0, 8))

    def place_dialog(self):
        width, height = self.size
        pw, ph, left, top = self.parent_box
        self.geometry(f'{width}x{height}+{left + max(0, (pw - width) // 2)}+{top + max(0, (ph - height) // 2)}')

    def fit_height(self):
        """Grow to the content when it is taller than the height the dialog was given, never past the panel."""
        self.update_idletasks()
        needed = min(self.winfo_reqheight(), self.parent_box[1])
        if needed > self.size[1]:
            self.size = (self.size[0], needed)
            self.place_dialog()

    def show(self):
        self.fit_height()
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
        self.error = tk.StringVar(value=f'Enter {self.range_description()}')
        ttk.Label(self.body, textvariable=self.value, style='Number.TLabel', anchor='e').pack(fill='x')
        ttk.Label(self.body, textvariable=self.error, style='PanelText.TLabel').pack(fill='x', pady=4)
        keys = ttk.Frame(self.body, style='Panel.TFrame')
        keys.pack(fill='both', expand=True)
        layout = [('7','8','9','Delete'), ('4','5','6','Clear'), ('1','2','3','.' if decimal else ''), ('0','0','','')]
        for c in range(4):
            keys.columnconfigure(c, weight=1, uniform='key')
        for r, row in enumerate(layout):
            keys.rowconfigure(r, weight=1, minsize=px(48))
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

    def range_description(self):
        minimum = f'{self.minimum:g}' if self.decimal else str(self.minimum)
        if self.maximum is None:
            return f'{minimum} or more'
        maximum = f'{self.maximum:g}' if self.decimal else str(self.maximum)
        return f'{minimum} to {maximum}'

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
        self.value.set(current if self.maximum is None else current[:10])

    def accept(self):
        try:
            text = self.value.get()
            value = float(text) if self.decimal else int(text)
            if (self.decimal and not math.isfinite(value)) or value < self.minimum:
                raise ValueError
            if self.maximum is not None and value > self.maximum:
                raise ValueError
        except ValueError:
            relation = 'of' if self.maximum is None else 'from'
            self.error.set(f'Use a number {relation} {self.range_description()}.')
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
            value = float(self.variable.get()) if self.decimal else int(self.variable.get())
            if self.decimal and not math.isfinite(value):
                raise ValueError
        except ValueError:
            value = self.minimum
        value = max(self.minimum, value + amount)
        if self.maximum is not None:
            value = min(self.maximum, value)
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
