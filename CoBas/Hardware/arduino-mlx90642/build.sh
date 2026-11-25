#!/usr/bin/env bash
# Compile the CoBas thermal camera firmware for the Arduino Nano ESP32 and,
# unless --compile-only is given, flash it to the board over USB.
#
#   ./build.sh                 compile and flash (port found by Arduino's USB ID)
#   ./build.sh --compile-only  just compile
#   ./build.sh --port /dev/ttyACM1
#
# Needs arduino-cli with the arduino:esp32 core:
#   arduino-cli core update-index && arduino-cli core install arduino:esp32
set -Eeuo pipefail

SKETCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd -- "$SKETCH_DIR/../mlx90642-library" && pwd)"
FQBN="arduino:esp32:nano_nora"
BUILD_DIR="${COBAS_ARDUINO_BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/cobas/arduino-mlx90642}"
PORT=""
UPLOAD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile-only) UPLOAD=0 ;;
        --port) PORT="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

command -v arduino-cli >/dev/null || { echo "arduino-cli is not on PATH" >&2; exit 1; }

# The vendor driver includes <MLX90642.h> and <MLX90642_depends.h> from inc/,
# and MLX90642_vendor.c includes <src/MLX90642.c>.
INCLUDES="-I$LIB_DIR -I$LIB_DIR/inc"
arduino-cli compile \
    --fqbn "$FQBN" \
    --build-path "$BUILD_DIR" \
    --build-property "compiler.c.extra_flags=$INCLUDES" \
    --build-property "compiler.cpp.extra_flags=$INCLUDES" \
    "$SKETCH_DIR"

if [[ "$UPLOAD" -eq 0 ]]; then
    exit 0
fi

if [[ -z "$PORT" ]]; then
    # Arduino's USB vendor ID is 2341; the radar's CP2105 ports are not matched.
    for candidate in /dev/serial/by-id/usb-*ARDUINO* /dev/serial/by-id/usb-Arduino*; do
        if [[ -e "$candidate" ]]; then
            PORT="$(readlink -f "$candidate")"
            break
        fi
    done
fi
[[ -n "$PORT" ]] || { echo "no Arduino serial port found; pass --port" >&2; exit 1; }

echo "flashing $PORT"
arduino-cli upload --fqbn "$FQBN" --port "$PORT" --input-dir "$BUILD_DIR" "$SKETCH_DIR"
