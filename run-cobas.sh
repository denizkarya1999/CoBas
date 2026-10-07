#!/usr/bin/env bash
set -euo pipefail
COBAS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MLX90642_SHARED_LIB="$COBAS_ROOT/.venv/lib/cobas/libmlx90642.so"
# Resolve speaker and microphone independently by their stable identities.
# Retain the optional explicit ALSA override for installations that use it.
if [[ -n "${COBAS_ALSA_CARD:-}" ]]; then
    export ALSA_CARD="$COBAS_ALSA_CARD"
    export ALSA_PCM_CARD="$COBAS_ALSA_CARD"
fi
cd "$COBAS_ROOT/CoBas/CoBas Battery Reader V1.0 App"
exec "$COBAS_ROOT/.venv/bin/python" CoBas_V1.py "$@"
