"""Resolve audio devices at use time, independently of USB port/card numbering."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys

import sounddevice as sd


class AudioDeviceError(ValueError):
    """An audio selection is absent, ambiguous, or incompatible."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    selector: str
    channels: int
    is_usb: bool


def _alsa_cards(proc_root=Path("/proc/asound"), sys_root=Path("/sys/class/sound")):
    """Read stable ALSA IDs and USB identity without opening any audio stream."""
    cards = {}
    for card in proc_root.glob("card[0-9]*"):
        try:
            index = int(card.name[4:])
            card_id = (card / "id").read_text().strip()
            device_path = (sys_root / card.name / "device").resolve()
            is_usb = (card / "usbbus").exists() or any(
                part.startswith("usb") for part in device_path.parts
            )
            cards[index] = {"id": card_id, "is_usb": is_usb}
        except (OSError, ValueError):
            continue
    return cards


def _devices(kind):
    if kind not in ("input", "output"):
        raise ValueError("Audio device kind must be 'input' or 'output'.")
    cards = _alsa_cards()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info[f"max_{kind}_channels"] < 1:
            continue
        name = str(info["name"])
        hardware = re.search(r"\(hw:(\d+),(\d+)\)", name)
        card = cards.get(int(hardware[1])) if hardware else None
        stable_name = re.sub(r"\s*\(hw:\d+,\d+\)", "", name).strip()
        if card and card["id"]:
            selector = f"alsa:{card['id']}:{hardware[2]}"
            is_usb = card["is_usb"]
        else:
            selector = f"name:{stable_name}"
            is_usb = "usb" in name.lower()
        devices.append({
            "index": index,
            "name": name,
            "stable_name": stable_name,
            "selector": selector,
            "is_usb": is_usb,
            "max_channels": int(info[f"max_{kind}_channels"]),
        })
    return devices


def list_audio_choices(kind="output", fresh=True):
    """Return labels and persistent selectors for a settings dropdown.

    The saved selector is resolved afresh in the playback process. The displayed
    card number is informational; it is never persisted as device identity.
    The GUI uses a fresh process because PortAudio caches its hardware list.
    Reinitializing PortAudio in the GUI would disrupt any active audio streams.
    """
    if kind not in ("input", "output"):
        raise ValueError("Audio device kind must be 'input' or 'output'.")
    if fresh:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--list", kind],
                capture_output=True, text=True, check=True, timeout=10,
            )
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise AudioDeviceError(f"Could not refresh {kind} audio devices: {exc}") from exc
    return [
        {"label": device["name"], "selector": device["selector"]}
        for device in _devices(kind)
    ]


def _compatible_device(device, kind, samplerate):
    # Stereo playback is needed by common USB speakers including UACDemoV1.0.
    # Preserve mono recordings by taking channel zero if capture needs stereo.
    channel_options = (2, 1) if kind == "output" else (1, 2)
    check = sd.check_output_settings if kind == "output" else sd.check_input_settings
    errors = []
    for channels in channel_options:
        if channels > device["max_channels"]:
            continue
        try:
            check(
                device=device["index"], channels=channels,
                samplerate=samplerate, dtype="float32",
            )
        except (sd.PortAudioError, ValueError) as exc:
            errors.append(str(exc))
            continue
        return AudioDevice(
            index=device["index"], name=device["name"],
            selector=device["selector"], channels=channels,
            is_usb=device["is_usb"],
        )
    detail = "; ".join(errors) or "No supported mono or stereo channel layout"
    raise AudioDeviceError(
        f"{device['name']} cannot provide {kind} at {samplerate} Hz: {detail}"
    )


def _default_index(kind):
    try:
        return int(sd.default.device[0 if kind == "input" else 1])
    except (TypeError, ValueError, IndexError):
        return -1


def resolve_audio_device(kind, selector=None, samplerate=48_000):
    """Resolve a stable name/card ID and validate 48 kHz mono/stereo support.

    Auto prefers compatible USB hardware independently for input and output.
    Multiple compatible USB devices require a selection unless one is the
    system default. Without USB hardware, a compatible system default is used.
    An explicitly selected missing device never silently switches to another.
    Integer indices remain accepted for existing command-line callers only.
    """
    devices = _devices(kind)
    if selector not in (None, "", "auto"):
        value = str(selector).strip()
        if value.isdecimal():
            matches = [d for d in devices if d["index"] == int(value)]
        elif value.startswith("name:"):
            matches = [d for d in devices if d["stable_name"] == value[5:]]
        elif value.startswith("alsa:"):
            matches = [d for d in devices if d["selector"] == value]
        else:
            matches = [d for d in devices if d["name"] == value or d["stable_name"] == value]
        if not matches:
            raise AudioDeviceError(
                f"Selected {kind} device '{selector}' is not connected. "
                "Reconnect it or choose another device in Settings."
            )
        if len(matches) != 1:
            raise AudioDeviceError(
                f"Selected {kind} device '{selector}' matches multiple devices. "
                "Choose a specific audio device in Settings."
            )
        return _compatible_device(matches[0], kind, samplerate)

    usb = [device for device in devices if device["is_usb"]]
    if usb:
        compatible, errors = [], []
        for device in usb:
            try:
                compatible.append(_compatible_device(device, kind, samplerate))
            except AudioDeviceError as exc:
                errors.append(str(exc))
        if len(compatible) == 1:
            return compatible[0]
        default = _default_index(kind)
        for device in compatible:
            if device.index == default:
                return device
        if compatible:
            names = ", ".join(device.name for device in compatible)
            raise AudioDeviceError(
                f"Multiple compatible USB {kind} devices are connected: {names}. "
                "Choose the desired device in Settings."
            )
        raise AudioDeviceError("No compatible USB " + kind + " device. " + "; ".join(errors))

    default = _default_index(kind)
    selected = next((d for d in devices if d["index"] == default), None)
    if selected is not None:
        return _compatible_device(selected, kind, samplerate)
    raise AudioDeviceError(
        f"No USB {kind} device or usable system default is connected. "
        "Connect a device or choose one in Settings."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List current CoBas audio devices.")
    parser.add_argument("--list", choices=("input", "output"), required=True)
    args = parser.parse_args()
    print(json.dumps(list_audio_choices(args.list, fresh=False)))
