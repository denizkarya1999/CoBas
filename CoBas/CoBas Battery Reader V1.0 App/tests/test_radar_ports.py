"""Hardware-free tests for USB identity, pairing, and failed-open cleanup."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

RAW_IQ = Path(__file__).resolve().parents[2] / 'mmWave/CoBas MMWave Processing/Raw IQ Signals'
sys.path.insert(0, str(RAW_IQ))
import iq_logic
from radar_ports import RadarPorts, discover_radar_ports


def port(device, interface, board='3-1', serial='radar-one', **changes):
    values = dict(device=device, vid=0x10C4, pid=0xEA70,
                  location=f'{board}:1.{interface}', serial_number=serial)
    values.update(changes)
    return SimpleNamespace(**values)


class RadarPortTests(unittest.TestCase):
    def discover(self, ports, **kwargs):
        return discover_radar_ports(ports, environ=kwargs.pop('environ', {}),
                                    by_id_directory=Path('/nonexistent-cobas-test'), **kwargs)

    def test_reversed_enumeration_and_numbering_use_interface_roles(self):
        result = self.discover([port('/dev/ttyUSB2', 1), port('/dev/ttyUSB9', 0)])
        self.assertEqual(result, RadarPorts('/dev/ttyUSB9', '/dev/ttyUSB2'))

    def test_unrelated_usb_devices_are_ignored(self):
        result = self.discover([port('/dev/ttyUSB0', 0, vid=0x1234),
                                port('/dev/ttyUSB6', 0), port('/dev/ttyUSB7', 1)])
        self.assertEqual(result, RadarPorts('/dev/ttyUSB6', '/dev/ttyUSB7'))

    def test_missing_radar_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'No CP2105 mmWave radar'):
            self.discover([])

    def test_missing_data_interface_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'could not be paired'):
            self.discover([port('/dev/ttyUSB0', 0)])

    def test_interfaces_from_different_boards_are_never_paired(self):
        with self.assertRaisesRegex(RuntimeError, 'Multiple CP2105'):
            self.discover([port('/dev/ttyUSB0', 0),
                           port('/dev/ttyUSB1', 1, board='3-2')])

    def test_multiple_complete_boards_fail_even_with_duplicate_serials(self):
        with self.assertRaisesRegex(RuntimeError, 'Multiple CP2105'):
            self.discover([port('/dev/ttyUSB0', 0), port('/dev/ttyUSB1', 1),
                           port('/dev/ttyUSB2', 0, board='3-2'),
                           port('/dev/ttyUSB3', 1, board='3-2')])

    def test_serial_selector_picks_one_complete_board(self):
        result = self.discover([port('/dev/ttyUSB0', 0), port('/dev/ttyUSB1', 1),
                               port('/dev/ttyUSB2', 0, board='3-2', serial='selected'),
                               port('/dev/ttyUSB3', 1, board='3-2', serial='selected')],
                              environ={'COBAS_RADAR_SERIAL': 'selected'})
        self.assertEqual(result, RadarPorts('/dev/ttyUSB2', '/dev/ttyUSB3'))

    def test_unknown_interface_metadata_does_not_guess_numbering(self):
        with self.assertRaisesRegex(RuntimeError, 'could not be paired'):
            self.discover([port('/dev/ttyUSB0', 0, location=None),
                           port('/dev/ttyUSB1', 1, location=None)])

    def test_sysfs_interface_number_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for interface in (0, 1):
                path = root / f'interface{interface}'
                path.mkdir()
                (path / 'bInterfaceNumber').write_text(f'{interface:02x}')
            result = self.discover([
                port('/dev/ttyUSB8', 0, location=None, usb_interface_path=root / 'interface0'),
                port('/dev/ttyUSB5', 1, location=None, usb_interface_path=root / 'interface1'),
            ])
        self.assertEqual(result, RadarPorts('/dev/ttyUSB8', '/dev/ttyUSB5'))

    def test_existing_persistent_symlinks_are_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            by_id = root / 'by-id'
            by_id.mkdir()
            control, data = root / 'ttyUSB8', root / 'ttyUSB5'
            control.touch()
            data.touch()
            (by_id / 'radar-if00').symlink_to(control)
            (by_id / 'radar-if01').symlink_to(data)
            result = discover_radar_ports([port(str(control), 0), port(str(data), 1)],
                                          environ={}, by_id_directory=by_id)
            self.assertEqual(result, RadarPorts(str(by_id / 'radar-if00'),
                                                str(by_id / 'radar-if01')))

    def test_both_explicit_ports_bypass_discovery(self):
        result = self.discover([], environ={'COBAS_RADAR_CLI_PORT': '/dev/radar-cli',
                                           'COBAS_RADAR_DATA_PORT': '/dev/radar-data'})
        self.assertEqual(result, RadarPorts('/dev/radar-cli', '/dev/radar-data'))

    def test_partial_override_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Set both'):
            self.discover([], environ={'COBAS_RADAR_CLI_PORT': '/dev/radar-cli'})

    def test_same_interface_override_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'different interfaces'):
            self.discover([], environ={'COBAS_RADAR_CLI_PORT': '/dev/ttyUSB0',
                                      'COBAS_RADAR_DATA_PORT': '/dev/ttyUSB0'})

    def test_same_interface_symlink_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'ttyUSB0'
            original.touch()
            alias = Path(directory) / 'alias'
            alias.symlink_to(original)
            with self.assertRaisesRegex(RuntimeError, 'different interfaces'):
                self.discover([], environ={'COBAS_RADAR_CLI_PORT': str(alias),
                                          'COBAS_RADAR_DATA_PORT': str(original)})


class RadarOpenTests(unittest.TestCase):
    def test_rediscovery_on_restart_uses_new_ports(self):
        pairs = [RadarPorts('cli-before', 'data-before'), RadarPorts('cli-after', 'data-after')]
        handles = [MagicMock(is_open=True) for _ in range(4)]
        serial = SimpleNamespace(Serial=MagicMock(side_effect=handles))
        with patch.dict(sys.modules, {'serial': serial}), \
                patch.object(iq_logic, 'discover_radar_ports', side_effect=pairs) as discovery, \
                patch.object(iq_logic.RadarUARTSource, '_send_command'), \
                patch.object(iq_logic.time, 'sleep'):
            source = iq_logic.RadarUARTSource()
            with source:
                self.assertEqual(source.cli_port, 'cli-before')
            with source:
                self.assertEqual(source.cli_port, 'cli-after')
        self.assertEqual(discovery.call_count, 2)
        self.assertEqual([call.args[0] for call in serial.Serial.call_args_list],
                         ['data-before', 'cli-before', 'data-after', 'cli-after'])
        for handle in handles:
            handle.close.assert_called_once()

    def test_control_open_failure_closes_already_open_data_port(self):
        data = MagicMock(is_open=True)
        serial = SimpleNamespace(Serial=MagicMock(side_effect=[data, OSError('permission denied')]))
        with patch.dict(sys.modules, {'serial': serial}), \
                patch.object(iq_logic, 'discover_radar_ports', return_value=RadarPorts('cli', 'data')):
            with self.assertRaisesRegex(OSError, 'permission denied'):
                iq_logic.RadarUARTSource().__enter__()
        data.close.assert_called_once()

    def test_configuration_failure_closes_both_ports(self):
        data, control = MagicMock(is_open=True), MagicMock(is_open=True)
        serial = SimpleNamespace(Serial=MagicMock(side_effect=[data, control]))
        with patch.dict(sys.modules, {'serial': serial}), \
                patch.object(iq_logic, 'discover_radar_ports', return_value=RadarPorts('cli', 'data')), \
                patch.object(iq_logic.RadarUARTSource, '_send_command', side_effect=RuntimeError('rejected')), \
                patch.object(iq_logic.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                iq_logic.RadarUARTSource().__enter__()
        data.close.assert_called_once()
        control.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
