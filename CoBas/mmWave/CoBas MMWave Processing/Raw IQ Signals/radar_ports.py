"""Find the two CP2105 radar interfaces without depending on USB socket order."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

RADAR_VENDOR_ID = 0x10C4
RADAR_PRODUCT_ID = 0xEA70


@dataclass(frozen=True)
class RadarPorts:
    cli_port: str
    data_port: str


def _interface_identity(port) -> tuple[str, int] | None:
    """Read interface identity, never infer its role from ttyUSB numbering."""
    # PySerial's Linux location includes the USB interface, e.g. 3-1:1.0.
    location = getattr(port, "location", None) or ""
    match = re.fullmatch(r"(.+):\d+\.(\d+)", location)
    if match:
        return match.group(1), int(match.group(2))

    interface_path = getattr(port, "usb_interface_path", None)
    if interface_path:
        path = Path(interface_path)
        try:
            return str(path.parent), int((path / "bInterfaceNumber").read_text().strip(), 16)
        except (OSError, ValueError):
            pass
    return None


def _stable_path(device: str, by_id_directory: Path) -> str:
    """Prefer an existing persistent symlink for this already-identified port."""
    target = Path(device).resolve()
    try:
        for link in sorted(by_id_directory.iterdir()):
            if link.is_symlink() and link.resolve() == target:
                return str(link)
    except OSError:
        pass
    return device


def discover_radar_ports(
    ports: Iterable | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    by_id_directory: Path = Path("/dev/serial/by-id"),
) -> RadarPorts:
    """Return one matching board's control/data pair, or explain the ambiguity.

    Enumeration only reads USB metadata; no serial port is opened or probed.
    Call this each time the radar starts so replugging and renumbering work.
    """
    env = os.environ if environ is None else environ
    cli_override = env.get("COBAS_RADAR_CLI_PORT", "").strip()
    data_override = env.get("COBAS_RADAR_DATA_PORT", "").strip()
    if cli_override or data_override:
        if not cli_override or not data_override:
            raise RuntimeError(
                "Set both COBAS_RADAR_CLI_PORT and COBAS_RADAR_DATA_PORT, "
                "or clear both to detect the radar automatically."
            )
        if Path(cli_override).resolve() == Path(data_override).resolve():
            raise RuntimeError("Radar control and data ports must be different interfaces.")
        return RadarPorts(cli_override, data_override)

    if ports is None:
        from serial.tools import list_ports

        ports = list_ports.comports()
    selected_serial = env.get("COBAS_RADAR_SERIAL", "").strip()
    boards: dict[str, dict[int, list[str]]] = {}
    candidates: list[str] = []
    for port in ports:
        if (getattr(port, "vid", None), getattr(port, "pid", None)) != (
            RADAR_VENDOR_ID, RADAR_PRODUCT_ID
        ):
            continue
        if selected_serial and getattr(port, "serial_number", None) != selected_serial:
            continue
        candidates.append(port.device)
        identity = _interface_identity(port)
        if identity is None:
            continue
        board, interface = identity
        if interface not in (0, 1):
            continue
        board_ports = boards.setdefault(board, {})
        devices = board_ports.setdefault(interface, [])
        if port.device not in devices:
            devices.append(port.device)

    complete = [
        interfaces for interfaces in boards.values()
        if len(interfaces.get(0, [])) == 1 and len(interfaces.get(1, [])) == 1
    ]
    if len(complete) == 1 and len(boards) == 1 and len(candidates) == 2:
        return RadarPorts(
            _stable_path(complete[0][0][0], by_id_directory),
            _stable_path(complete[0][1][0], by_id_directory),
        )
    if len(boards) > 1:
        raise RuntimeError(
            "Multiple CP2105 radar boards are connected. Unplug the unused radar "
            "or set COBAS_RADAR_SERIAL to the intended board's USB serial number."
        )
    if candidates:
        raise RuntimeError(
            "The CP2105 radar was found, but its control interface 00 and data "
            f"interface 01 could not be paired ({', '.join(candidates)}). "
            "Reconnect the radar USB cable; both interfaces must be available. "
            "If USB interface metadata is unavailable, explicitly set both "
            "COBAS_RADAR_CLI_PORT and COBAS_RADAR_DATA_PORT."
        )
    selector = f" with USB serial {selected_serial}" if selected_serial else ""
    raise RuntimeError(
        f"No CP2105 mmWave radar{selector} was detected. Connect the powered "
        "IWR6843AOP to any USB socket using its data-capable USB cable."
    )
