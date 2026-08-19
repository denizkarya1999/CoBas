#!/usr/bin/env python3
"""Send a TI mmWave .cfg file to the IWR6843AOP CLI UART."""

from __future__ import annotations

import argparse
import glob
import time

import serial


def default_port() -> str:
    matches = sorted(
        glob.glob(
            "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_"
            "Bridge_Controller_*-if00-port0"
        )
    )
    return matches[0] if matches else "/dev/ttyUSB0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="TI mmWave configuration (.cfg)")
    parser.add_argument("--port", default=default_port())
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between commands")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        commands = [
            line.strip()
            for line in stream
            if line.strip() and not line.lstrip().startswith("%")
        ]

    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=1) as uart:
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
