#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="MLX90642 Thermal Camera"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="${SCRIPT_DIR}/thermal_camera_app.py"
DRIVER_LIB="/usr/local/lib/libmlx90642.so"
LAUNCHER="/usr/local/bin/cobas-thermal-camera"
BOOT_CONFIG=""
REAL_USER="${SUDO_USER:-${USER:-}}"
NEEDS_REBOOT=0

log() {
    printf '\n==> %s\n' "$*"
}

warn() {
    printf '\nWARNING: %s\n' "$*" >&2
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

rerun_with_sudo_if_needed() {
    if [[ "${EUID}" -ne 0 ]]; then
        log "Requesting sudo permissions for system driver installation"
        exec sudo -E bash "$0" "$@"
    fi
}

detect_target_user() {
    if [[ -z "${REAL_USER}" || "${REAL_USER}" == "root" ]]; then
        REAL_USER="$(logname 2>/dev/null || printf 'root')"
    fi

    if ! id "${REAL_USER}" >/dev/null 2>&1; then
        warn "Could not find login user '${REAL_USER}'. User group setup will be skipped."
        REAL_USER="root"
    fi
}

check_os() {
    if [[ ! -r /etc/os-release ]]; then
        warn "Could not read /etc/os-release; continuing anyway."
        return
    fi

    # shellcheck disable=SC1091
    . /etc/os-release

    if [[ "${ID:-}" != "ubuntu" ]]; then
        warn "This script is intended for Ubuntu 26.04 LTS on Raspberry Pi 5. Detected ID='${ID:-unknown}'."
    fi

    if [[ "${VERSION_ID:-}" != "26.04" ]]; then
        warn "This script is intended for Ubuntu 26.04 LTS. Detected VERSION_ID='${VERSION_ID:-unknown}'."
    fi
}

install_packages() {
    log "Installing system packages"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y \
        build-essential \
        ffmpeg \
        gcc \
        i2c-tools \
        libi2c-dev \
        make \
        python3 \
        python3-tk \
        python3-venv
}

install_driver_library() {
    if [[ ! -f "$APP_FILE" ]]; then
        warn "Could not find ${APP_FILE}; skipping driver library installation."
        return
    fi

    log "Building and installing ${APP_NAME} driver library"

    local build_dir
    local build_output
    build_dir="$(mktemp -d /tmp/cobas-mlx90642.XXXXXX)"
    build_output="${build_dir}/libmlx90642.so"

    if [[ "${REAL_USER}" != "root" ]]; then
        chown "$REAL_USER:$REAL_USER" "$build_dir"
    fi

    run_as_target_user env MLX90642_SHARED_LIB="$build_output" python3 "$APP_FILE" --build-only
    install -m 0755 "$build_output" "$DRIVER_LIB"
    rm -rf "$build_dir"
    ldconfig

    log "Installed ${DRIVER_LIB}"
}

install_launcher() {
    log "Installing application launcher"

cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

cd "${SCRIPT_DIR}"
exec python3 "${APP_FILE}" "\$@"
EOF

    chmod 0755 "$LAUNCHER"
    log "Installed ${LAUNCHER}"
}

choose_boot_config() {
    if [[ -f /boot/firmware/usercfg.txt ]]; then
        BOOT_CONFIG="/boot/firmware/usercfg.txt"
    elif [[ -f /boot/firmware/config.txt ]]; then
        BOOT_CONFIG="/boot/firmware/config.txt"
    elif [[ -f /boot/config.txt ]]; then
        BOOT_CONFIG="/boot/config.txt"
    else
        warn "Could not find Raspberry Pi boot config. I2C overlay setup was skipped."
        return 1
    fi

    return 0
}

backup_file_once() {
    local file="$1"
    local backup="${file}.cobas-backup-$(date +%Y%m%d-%H%M%S)"

    cp -a "$file" "$backup"
    log "Backed up ${file} to ${backup}"
}

ensure_config_line() {
    local file="$1"
    local line="$2"

    if grep -Fxq "$line" "$file"; then
        return
    fi

    printf '\n%s\n' "$line" >> "$file"
    NEEDS_REBOOT=1
    log "Added '${line}' to ${file}"
}

enable_i2c_boot_overlay() {
    log "Enabling Raspberry Pi I2C boot overlay"

    if ! choose_boot_config; then
        return
    fi

    backup_file_once "$BOOT_CONFIG"
    ensure_config_line "$BOOT_CONFIG" "dtparam=i2c_arm=on"

    if [[ -n "${MLX90642_I2C_BAUDRATE:-}" ]]; then
        ensure_config_line "$BOOT_CONFIG" "dtparam=i2c_arm_baudrate=${MLX90642_I2C_BAUDRATE}"
    fi
}

enable_i2c_kernel_modules() {
    log "Enabling Linux I2C device support"

    install -d -m 0755 /etc/modules-load.d
    printf 'i2c-dev\n' > /etc/modules-load.d/cobas-i2c.conf

    modprobe i2c-dev || warn "Could not load i2c-dev immediately. It should load after reboot."
    modprobe i2c-bcm2835 2>/dev/null || true
}

configure_i2c_permissions() {
    log "Configuring /dev/i2c-* permissions"

    if ! getent group i2c >/dev/null; then
        groupadd --system i2c
    fi

    install -d -m 0755 /etc/udev/rules.d
    printf 'KERNEL=="i2c-[0-9]*", GROUP="i2c", MODE="0660"\n' > /etc/udev/rules.d/99-cobas-i2c.rules
    udevadm control --reload-rules || true
    udevadm trigger || true

    if [[ "${REAL_USER}" != "root" ]]; then
        usermod -aG i2c "$REAL_USER"
        log "Added ${REAL_USER} to the i2c group"
    fi
}

run_as_target_user() {
    if [[ "${REAL_USER}" != "root" ]]; then
        sudo -u "$REAL_USER" env PYTHONDONTWRITEBYTECODE=1 "$@"
    else
        env PYTHONDONTWRITEBYTECODE=1 "$@"
    fi
}

verify_tools() {
    log "Verifying installed tools"

    command -v gcc >/dev/null || die "gcc was not installed"
    command -v ffmpeg >/dev/null || die "ffmpeg was not installed"
    command -v i2cdetect >/dev/null || die "i2c-tools was not installed"
    python3 - <<'PY'
import tkinter
print("tkinter OK")
PY
}

print_i2c_status() {
    log "Checking I2C device nodes"

    if compgen -G "/dev/i2c-*" >/dev/null; then
        ls -l /dev/i2c-*
    else
        warn "No /dev/i2c-* devices are visible yet."
        NEEDS_REBOOT=1
    fi
}

print_next_steps() {
    log "Installation complete"

    if [[ "${REAL_USER}" != "root" ]]; then
        printf 'User: %s\n' "$REAL_USER"
        printf 'Group note: log out and back in, or reboot, so the i2c group membership applies.\n'
    fi

    if [[ "${NEEDS_REBOOT}" -eq 1 ]]; then
        printf 'Reboot required before the camera can use I2C:\n'
        printf '  sudo reboot\n'
    fi

    printf '\nAfter reboot, test the bus:\n'
    printf '  i2cdetect -y 1\n'
    printf '\nRun the desktop camera app:\n'
    printf '  cobas-thermal-camera\n'
    printf '\nOr from the source directory:\n'
    printf '  cd %q\n' "$SCRIPT_DIR"
    printf '  python3 thermal_camera_app.py\n'
    printf '\nIf the sensor is on a different bus:\n'
    printf '  MLX90642_I2C_BUS=/dev/i2c-0 cobas-thermal-camera\n'
}

main() {
    rerun_with_sudo_if_needed "$@"
    detect_target_user
    check_os
    install_packages
    enable_i2c_boot_overlay
    enable_i2c_kernel_modules
    configure_i2c_permissions
    install_driver_library
    install_launcher
    verify_tools
    print_i2c_status
    print_next_steps
}

main "$@"
