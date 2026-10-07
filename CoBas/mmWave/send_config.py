#!/usr/bin/env python3
"""Send a TI mmWave .cfg file to the IWR6843AOP CLI UART."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parent / "CoBas MMWave Processing" / "Raw IQ Signals"))
from radar_ports import discover_radar_ports


def default_port() -> str:
    return discover_radar_ports().cli_port


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="TI mmWave configuration (.cfg)")
    parser.add_argument("--port", help="Explicit control port; otherwise detect the CP2105 radar")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between commands")
    args = parser.parse_args()
    port = args.port or default_port()

    with open(args.config, encoding="utf-8") as stream:
        commands = [
            line.strip()
            for line in stream
            if line.strip() and not line.lstrip().startswith("%")
        ]

    with serial.Serial(port, 115200, timeout=0.25, write_timeout=1) as uart:
        uart.reset_input_buffer()
        for command in commands:
            print(f"> {command}")
            uart.write((command + "\n").encode("ascii"))
            uart.flush()
            time.sleep(args.delay)
            reply = uart.read(4096).decode("utf-8", errors="replace").strip()
            if reply:
                print(reply)
            if "Error" in reply:
                raise RuntimeError(f"Radar rejected command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
