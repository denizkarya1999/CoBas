"""Find the touch display the app should fill, and turn it the way the operator wants.

The touch display is an SPI or DSI panel beside an HDMI monitor. Its rotation
is a CoBas setting (portrait unless changed in Settings), kept in
~/.config/cobas/display.json and applied with wlr-randr as the output's
transform. labwc turns touch input with the output it is mapped to, so the
touch calibration the udev rule gives the digitiser is right in every rotation.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOUCH_OUTPUT_PREFIXES = ("SPI", "DSI")
# Name an output explicitly (as wlr-randr or xrandr lists it) to override the choice.
OUTPUT_ENVIRONMENT_VARIABLE = "COBAS_TOUCH_OUTPUT"
ROTATED_TRANSFORMS = {"90", "270", "flipped-90", "flipped-270"}
# Degrees the panel is turned, and wlr-randr's name for that output transform.
TRANSFORMS = {0: "normal", 90: "90", 180: "180", 270: "270"}
ROTATIONS = tuple(TRANSFORMS)
ROTATION_NAMES = {0: "Landscape", 90: "Portrait", 180: "Landscape, upside down", 270: "Portrait, upside down"}
DEFAULT_ROTATION = 90  # the panel stands upright unless the operator chooses otherwise

SETTINGS_FILE = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "cobas" / "display.json"


def _run(command, timeout=5):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True).stdout


def _wayland_outputs():
    outputs = []
    for output in json.loads(_run(["wlr-randr", "--json"])):
        if not output.get("enabled"):
            continue
        mode = next((m for m in output.get("modes", []) if m.get("current")), None)
        if mode is None:
            continue
        scale = float(output.get("scale") or 1.0)
        width, height = round(mode["width"] / scale), round(mode["height"] / scale)
        transform = str(output.get("transform", "normal"))
        if transform in ROTATED_TRANSFORMS:
            width, height = height, width
        position = output.get("position") or {}
        outputs.append({"name": output["name"], "x": int(position.get("x", 0)), "y": int(position.get("y", 0)),
                        "width": width, "height": height, "transform": transform})
    return outputs


def _x11_outputs():
    outputs = []
    for line in _run(["xrandr", "--listmonitors"]).splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        size, x, y = parts[2].split("+")  # e.g. 480/73x320/49+1920+0
        width, height = (part.split("/")[0] for part in size.split("x"))
        outputs.append({"name": parts[-1], "x": int(x), "y": int(y), "width": int(width), "height": int(height),
                        "transform": "normal"})
    return outputs


def can_rotate():
    """Rotation is done through wlr-randr, so it needs a Wayland session."""
    return bool(shutil.which("wlr-randr") and os.environ.get("WAYLAND_DISPLAY"))


def list_outputs():
    """Return the enabled outputs as dicts of name, x, y, width, height (logical pixels) and transform."""
    for available, reader in ((can_rotate(), _wayland_outputs), (shutil.which("xrandr"), _x11_outputs)):
        if not available:
            continue
        try:
            outputs = reader()
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            continue
        if outputs:
            return outputs
    return []


def find_touch_display():
    """Return the output the touch UI should fill, or None to keep the default size and position."""
    outputs = list_outputs()
    wanted = os.environ.get(OUTPUT_ENVIRONMENT_VARIABLE, "").strip()
    if wanted:
        for output in outputs:
            if output["name"] == wanted:
                return output
    for output in outputs:
        if output["name"].upper().startswith(TOUCH_OUTPUT_PREFIXES):
            return output
    return None


def rotation_of(display):
    """The display's rotation in degrees, or None for a transform the app does not offer (a flipped one)."""
    for degrees, transform in TRANSFORMS.items():
        if display.get("transform") == transform:
            return degrees
    return None


# --------------------------------------------------------------------------- #
# The saved rotation
# --------------------------------------------------------------------------- #

def load_rotation():
    """The rotation the operator chose, or the default when nothing was saved yet."""
    try:
        rotation = int(json.loads(SETTINGS_FILE.read_text())["rotation"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_ROTATION
    return rotation if rotation in TRANSFORMS else DEFAULT_ROTATION


def save_rotation(rotation):
    if rotation not in TRANSFORMS:
        raise ValueError(f"rotation must be one of {ROTATIONS}, not {rotation!r}")
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"rotation": int(rotation)}, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Rotating the display
# --------------------------------------------------------------------------- #

def set_output_rotation(output_name, rotation):
    _run(["wlr-randr", "--output", output_name, "--transform", TRANSFORMS[rotation]], timeout=10)


def wait_for_rotation(output_name, rotation, timeout=3.0):
    """The output once the compositor reports the new transform (or as it is when that takes too long)."""
    deadline = time.monotonic() + timeout
    while True:
        display = next((o for o in list_outputs() if o["name"] == output_name), None)
        if (display and rotation_of(display) == rotation) or time.monotonic() > deadline:
            return display
        time.sleep(0.2)


def rotate_touch_display(display, rotation):
    """Turn the panel; returns the display as it is afterwards (touch follows the output in labwc)."""
    if rotation_of(display) != rotation:
        set_output_rotation(display["name"], rotation)
    return wait_for_rotation(display["name"], rotation) or display


def prepare_touch_display():
    """The touch display, turned as the saved setting says; None when there is no touch display."""
    display = find_touch_display()
    if display is None or not can_rotate():
        return display
    rotation = load_rotation()
    try:
        return rotate_touch_display(display, rotation)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[WARNING] The touch display could not be turned to {ROTATION_NAMES[rotation]}: {error}")
        return display


def main(arguments):
    """Show the touch display, or turn it: python Display.py [0|90|180|270]."""
    if arguments:
        rotation = int(arguments[0])
        save_rotation(rotation)
        display = prepare_touch_display()
        if display is None:
            print("No touch display (SPI or DSI output) is attached.")
            return 1
        print(f"{display['name']}: {ROTATION_NAMES.get(rotation_of(display), display['transform'])}, "
              f"{display['width']} x {display['height']} at {display['x']},{display['y']}")
        return 0 if rotation_of(display) == rotation else 1
    display = find_touch_display()
    print(f"Saved rotation: {ROTATION_NAMES[load_rotation()]} ({load_rotation()} degrees)")
    if display is None:
        print("No touch display (SPI or DSI output) is attached.")
        return 1
    print(f"{display['name']}: transform {display['transform']}, {display['width']} x {display['height']} "
          f"at {display['x']},{display['y']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
