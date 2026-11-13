"""Find the touch display the app should fill: an SPI or DSI panel beside an HDMI monitor."""
import json
import os
import shutil
import subprocess

TOUCH_OUTPUT_PREFIXES = ("SPI", "DSI")
# Name an output explicitly (as wlr-randr or xrandr lists it) to override the choice.
OUTPUT_ENVIRONMENT_VARIABLE = "COBAS_TOUCH_OUTPUT"
ROTATED_TRANSFORMS = {"90", "270", "flipped-90", "flipped-270"}


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout


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
        if str(output.get("transform", "normal")) in ROTATED_TRANSFORMS:
            width, height = height, width
        position = output.get("position") or {}
        outputs.append({"name": output["name"], "x": int(position.get("x", 0)),
                        "y": int(position.get("y", 0)), "width": width, "height": height})
    return outputs


def _x11_outputs():
    outputs = []
    for line in _run(["xrandr", "--listmonitors"]).splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        size, x, y = parts[2].split("+")  # e.g. 480/73x320/49+1920+0
        width, height = (part.split("/")[0] for part in size.split("x"))
        outputs.append({"name": parts[-1], "x": int(x), "y": int(y), "width": int(width), "height": int(height)})
    return outputs


def list_outputs():
    """Return the enabled outputs as dicts of name, x, y, width and height (logical pixels)."""
    for available, reader in ((shutil.which("wlr-randr") and os.environ.get("WAYLAND_DISPLAY"), _wayland_outputs),
                              (shutil.which("xrandr"), _x11_outputs)):
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
