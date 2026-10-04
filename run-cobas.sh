#!/usr/bin/env bash
set -euo pipefail
COBAS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MLX90642_SHARED_LIB="$COBAS_ROOT/.venv/lib/cobas/libmlx90642.so"
# Use the connected USB audio adapter for CoBas playback and capture.
export ALSA_CARD="${COBAS_ALSA_CARD:-Device}"
export ALSA_PCM_CARD="$ALSA_CARD"
cd "$COBAS_ROOT/CoBas/CoBas Battery Reader V1.0 App"
exec "$COBAS_ROOT/.venv/bin/python" CoBas_V1.py "$@"
