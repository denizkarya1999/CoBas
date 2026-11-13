#!/usr/bin/env bash
# Start the CoBas Battery Reader automatically when the desktop session starts,
# that is, whenever the Raspberry Pi boots to its desktop.
#
#   ./install-autostart.sh            install or refresh the autostart entry
#   ./install-autostart.sh --remove   stop starting the app at boot
#
# The entry is an XDG autostart file, which the Raspberry Pi desktop runs at
# login. It points at run-cobas.sh in this checkout, so run the script again
# if the repository moves.
set -euo pipefail
COBAS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
ENTRY="$AUTOSTART_DIR/cobas.desktop"

if [[ "${1:-}" == "--remove" ]]; then
    rm -f "$ENTRY"
    echo "Removed $ENTRY"
    exit 0
fi

mkdir -p "$AUTOSTART_DIR"
cat > "$ENTRY" <<ENTRY_END
[Desktop Entry]
Type=Application
Name=CoBas
Comment=CoBas Battery Reader, started with the desktop
Exec="$COBAS_ROOT/run-cobas.sh"
Path=$COBAS_ROOT
Icon=battery
Terminal=false
ENTRY_END
echo "Installed $ENTRY"
