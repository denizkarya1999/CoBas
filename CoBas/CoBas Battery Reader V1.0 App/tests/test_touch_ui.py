"""Display-free checks for the touchscreen number controls."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

APP = Path(os.environ.get('COBAS_APP_SOURCE', Path(__file__).resolve().parents[1]))
UNDER_TEST = Path(os.environ.get('COBAS_APP_UNDER_TEST', APP))
sys.path.insert(0, str(APP))
spec = importlib.util.spec_from_file_location('touch_ui_under_test', UNDER_TEST / 'TouchUI.py')
touch_ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(touch_ui)


class Value:
    def __init__(self, value=''):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


class NumberControlTests(unittest.TestCase):
    def dialog(self, text='', minimum=1, maximum=None, decimal=False):
        dialog = object.__new__(touch_ui.NumberDialog)
        dialog.minimum = minimum
        dialog.maximum = maximum
        dialog.decimal = decimal
        dialog.value = Value(text)
        dialog.error = Value()
        dialog.replace_next = True
        dialog.finish = MagicMock()
        return dialog

    def control(self, value, minimum=1, maximum=None, decimal=False):
        control = object.__new__(touch_ui.NumberControl)
        control.variable = Value(value)
        control.minimum = minimum
        control.maximum = maximum
        control.decimal = decimal
        control.disabled = False
        return control

    def test_keypad_accepts_entire_unbounded_chirp_count(self):
        dialog = self.dialog('3')
        requested = '9007199254740993'
        for digit in requested:
            dialog.type_key(digit)
        self.assertEqual(dialog.value.get(), requested)
        dialog.accept()
        dialog.finish.assert_called_once_with(int(requested))

    def test_unbounded_integer_does_not_require_float_conversion(self):
        requested = 10 ** 400
        dialog = self.dialog(requested)
        dialog.accept()
        dialog.finish.assert_called_once_with(requested)

    def test_unbounded_dialog_describes_minimum(self):
        dialog = self.dialog('0')
        self.assertEqual(dialog.range_description(), '1 or more')
        dialog.accept()
        dialog.finish.assert_not_called()
        self.assertEqual(dialog.error.get(), 'Use a number of 1 or more.')

    def test_unbounded_dialog_rejects_invalid_integer(self):
        for text in ('', '-1', '1.5', 'inf', 'nan'):
            with self.subTest(text=text):
                dialog = self.dialog(text)
                dialog.accept()
                dialog.finish.assert_not_called()

    def test_bounded_battery_dialog_preserves_limits(self):
        for number in (0, 100):
            with self.subTest(number=number):
                dialog = self.dialog(number, minimum=0, maximum=100)
                dialog.accept()
                dialog.finish.assert_called_once_with(number)
        for number in (-1, 101):
            with self.subTest(number=number):
                dialog = self.dialog(number, minimum=0, maximum=100)
                dialog.accept()
                dialog.finish.assert_not_called()
                self.assertEqual(dialog.error.get(), 'Use a number from 0 to 100.')

    def test_thermal_dialog_preserves_decimal_bounds_and_finite_check(self):
        for text in ('-0.1', '60.1', 'nan', 'inf', '-inf'):
            with self.subTest(text=text):
                dialog = self.dialog(text, minimum=0, maximum=60, decimal=True)
                dialog.accept()
                dialog.finish.assert_not_called()
        dialog = self.dialog('2.5', minimum=0, maximum=60, decimal=True)
        dialog.accept()
        dialog.finish.assert_called_once_with(2.5)

    def test_adjust_preserves_large_integer_exactly(self):
        control = self.control('9007199254740993')
        control.adjust(1)
        self.assertEqual(control.variable.get(), '9007199254740994')
        control.adjust(-1)
        self.assertEqual(control.variable.get(), '9007199254740993')

    def test_adjust_has_no_chirp_ceiling(self):
        control = self.control('999')
        control.adjust(1)
        self.assertEqual(control.variable.get(), '1000')
        control = self.control('1')
        control.adjust(-1)
        self.assertEqual(control.variable.get(), '1')

    def test_bounded_adjust_preserves_battery_and_thermal_limits(self):
        for maximum, decimal in ((100, False), (60, True)):
            with self.subTest(maximum=maximum):
                control = self.control(maximum, minimum=0, maximum=maximum, decimal=decimal)
                control.adjust(1)
                self.assertEqual(control.variable.get(), str(maximum))
                control.variable.set(0)
                control.adjust(-1)
                self.assertEqual(control.variable.get(), '0')
        control = self.control('2.5', minimum=0, maximum=60, decimal=True)
        control.adjust(0.5)
        self.assertEqual(control.variable.get(), '3')

    def test_edit_passes_unbounded_range_and_preserves_integer(self):
        control = self.control('200')
        control.title = 'Chirps'
        control.winfo_toplevel = MagicMock(return_value='parent')
        requested = 9007199254740993
        with patch.object(touch_ui, 'ask_number', return_value=requested) as ask:
            control.edit()
        ask.assert_called_once_with('parent', 'Chirps', initial='200', minimum=1,
                                    maximum=None, decimal=False)
        self.assertEqual(control.variable.get(), str(requested))


if __name__ == '__main__':
    unittest.main()
