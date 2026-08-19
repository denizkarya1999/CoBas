#!/usr/bin/env python3
"""Start the hard-coded IWR6843AOP USB1 I/Q logger terminal interface."""

from __future__ import annotations

import sys

from iq_logic import (
    CaptureInfo,
    CaptureResult,
    IQSample,
    capture,
    log_path_for_name,
)


# Printing all USB1 symbols can overflow a terminal and cause serial data loss.
# Show antenna 0 at every eighth range bin while logging the complete matrix.
TERMINAL_RANGE_BIN_STEP = 8


def prompt_log_name() -> str:
    """Ask until the user supplies a safe, unused CSV filename."""
    while True:
        log_name = input("Enter log name: ").strip()
        try:
            log_path = log_path_for_name(log_name)
        except ValueError as error:
            print(f"Invalid log name: {error}")
            continue
        if log_path.exists():
            print(f"A log named '{log_path.name}' already exists. Choose another name.")
            continue
        return log_name


def display_startup(info: CaptureInfo) -> None:
    print(f"Radar control: {info.cli_port} at 115200 baud")
    print(f"Complex I/Q data: {info.data_port} at 921600 baud")
    print("Signal: zero-Doppler range-FFT symbols, signed int16 Im/Re")
    print(f"CSV log: {info.log_path}")
    print("Applying the embedded radar configuration and starting the sensor...")
    print("Press Ctrl+C to stop.\n")


def display_sample(sample: IQSample) -> None:
    if sample.virtual_antenna != 0 or sample.range_bin % TERMINAL_RANGE_BIN_STEP:
        return
    print(
        f"frame={sample.frame_number:8d}  "
        f"range={sample.range_bin:02d}  ant={sample.virtual_antenna:02d}  "
        f"I={sample.i:7d}  Q={sample.q:7d}"
    )


def display_summary(result: CaptureResult) -> None:
    if result.interrupted:
        print("\nCapture stopped by user.")
    print(
        f"Logged {result.sample_count} complex sample(s) from "
        f"{result.frame_count} frame(s) to {result.log_path}"
    )
    if result.malformed_packets or result.discarded_bytes:
        print(
            f"Parser warning: {result.malformed_packets} malformed packet(s), "
            f"{result.discarded_bytes} discarded byte(s).",
            file=sys.stderr,
        )


def main() -> int:
    """Run the fixed USB0/USB1 capture; no arguments are accepted or needed."""
    try:
        log_name = prompt_log_name()
        result = capture(
            log_name,
            on_started=display_startup,
            on_sample=display_sample,
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCapture cancelled before startup.")
        return 130
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    display_summary(result)
    return 130 if result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
