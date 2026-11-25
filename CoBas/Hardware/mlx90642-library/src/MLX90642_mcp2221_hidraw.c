#include <stdint.h>
#include <errno.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <poll.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/hidraw.h>

#include "MLX90642_depends.h"

/*
 * Linux userspace transport for the MLX90642 behind an MCP2221A USB-to-I2C
 * bridge.  The bridge is a USB HID device: each command is one 64-byte output
 * report and each answer one 64-byte input report, exchanged through the
 * bridge's /dev/hidraw node.  Only the kernel's generic HID support is needed;
 * 99-mcp2221a.rules makes the node accessible to the desktop user.
 *
 * Framing follows the MCP2221A datasheet (DS20005565) and the Linux
 * hid-mcp2221 driver.  A register read is "write address, no STOP" (0x94)
 * followed by "read with REPEATED START" (0x93) and as many "get I2C data"
 * (0x40) requests as needed for the 60-byte chunks the bridge buffers.
 *
 * The contract is the same as MLX90642_linux_i2c.c: zero on success, -1 on a
 * transport failure that was reported on stderr, with errno describing it.
 */

#define MCP2221_VENDOR_ID 0x04D8
#define MCP2221_PRODUCT_ID 0x00DD
#define HIDRAW_CLASS_DIR "/sys/class/hidraw"
/* A nonempty value names the bridge's node and skips the sysfs search. */
#define HIDRAW_ENV "MLX90642_MCP2221_HIDRAW"

#define REPORT_SIZE 64
#define I2C_CHUNK_SIZE 60

/* Command codes, the first byte of an output report. */
#define CMD_STATUS_SET_PARAMETERS 0x10
#define CMD_I2C_GET_DATA 0x40
#define CMD_I2C_WRITE 0x90
#define CMD_I2C_READ_REPEATED_START 0x93
#define CMD_I2C_WRITE_NO_STOP 0x94

/* CMD_STATUS_SET_PARAMETERS sub-commands and their acknowledgements. */
#define PARAM_CANCEL_TRANSFER 0x10
#define PARAM_SET_SPEED 0x20
#define SPEED_ACCEPTED 0x20

/* Response byte positions. */
#define RSP_ECHO 0
#define RSP_STATUS 1
#define RSP_I2C_STATE 2 /* in data command replies */
#define RSP_GET_DATA_LENGTH 3
#define RSP_GET_DATA_PAYLOAD 4
#define RSP_SPEED_RESULT 3 /* in status replies */
#define RSP_STATUS_I2C_STATE 8
#define RSP_STATUS_ADDRESS_ACK 20
#define RSP_STATUS_SCL_LEVEL 22
#define RSP_STATUS_SDA_LEVEL 23

#define STATUS_OK 0x00
#define STATUS_BUSY 0x01
#define STATUS_GET_DATA_NOT_READY 0x41
#define GET_DATA_LENGTH_ERROR 127
#define ADDRESS_NACK_MASK 0x40

/* I2C engine states that matter to this transport. */
#define I2C_STATE_IDLE 0x00
#define I2C_STATE_START_TIMEOUT 0x12
#define I2C_STATE_REPEATED_START_TIMEOUT 0x17
#define I2C_STATE_ADDRESS_TIMEOUT 0x23
#define I2C_STATE_ADDRESS_NACK 0x25
#define I2C_STATE_DATA_TIMEOUT 0x44
#define I2C_STATE_READ_TIMEOUT 0x52
#define I2C_STATE_READ_MORE 0x54 /* a chunk is ready and more follow */
#define I2C_STATE_READ_LAST 0x55 /* the final chunk is ready */
#define I2C_STATE_STOP_TIMEOUT 0x62

/* The bridge tops out at 400 kHz. The divider formula is the datasheet's. */
#define I2C_SPEED_HZ 400000
#define I2C_CLOCK_DIVIDER ((12000000 / I2C_SPEED_HZ) - 3)

/* One HID round trip takes about a millisecond; waits are in those units. */
#define STEP_DELAY_US 1000
#define RESPONSE_TIMEOUT_MS 1000
#define STALE_RESPONSES_TOLERATED 4
#define BUSY_RETRIES 20
#define SETTLE_POLLS 10
#define READ_TIMEOUT_MS 2000
/* The sensor can briefly NACK while changing measurement/read-window state. */
#define READ_MAX_ATTEMPTS 4
#define READ_RETRY_DELAY_US 2500

static int hid_fd = -1;
static char device_path[PATH_MAX] = "(not opened)";
static int device_from_environment;
/* Why the last hook call failed, in words; MLX90642_TransportLastError returns it. */
static char last_error[256] = "";

static void sleep_us(long microseconds)
{
    struct timespec delay;

    delay.tv_sec = microseconds / 1000000;
    delay.tv_nsec = (microseconds % 1000000) * 1000;
    /* An interrupted delay is not resumed, as in the I2C header transport. */
    nanosleep(&delay, NULL);
}

static long elapsed_ms(const struct timespec *since)
{
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - since->tv_sec) * 1000 +
           (now.tv_nsec - since->tv_nsec) / 1000000;
}

static void note_error(const char *format, ...)
{
    va_list arguments;

    va_start(arguments, format);
    vsnprintf(last_error, sizeof(last_error), format, arguments);
    va_end(arguments);
}

const char *MLX90642_TransportLastError(void)
{
    return last_error;
}

static int fail(const char *what, int error_number)
{
    fprintf(stderr, "mcp2221 %s: %s: %s\n", device_path, what,
            strerror(error_number));
    errno = error_number;
    return -1;
}

static void close_bridge(void)
{
    if (hid_fd >= 0) {
        close(hid_fd);
        hid_fd = -1;
    }
}

static int find_bridge_in_sysfs(char *path, size_t size)
{
    /*
     * Every hidraw node publishes its HID identity in sysfs as
     * HID_ID=<bus>:<vendor>:<product>, readable without any privilege.
     */
    DIR *directory = opendir(HIDRAW_CLASS_DIR);
    struct dirent *entry;
    int found = 0;

    if (directory == NULL) {
        return -1;
    }

    while (!found && (entry = readdir(directory)) != NULL) {
        char uevent_path[PATH_MAX];
        char line[256];
        FILE *uevent;

        if (strncmp(entry->d_name, "hidraw", 6) != 0) {
            continue;
        }

        snprintf(uevent_path, sizeof(uevent_path),
                 HIDRAW_CLASS_DIR "/%s/device/uevent", entry->d_name);
        uevent = fopen(uevent_path, "r");
        if (uevent == NULL) {
            continue;
        }

        while (fgets(line, sizeof(line), uevent) != NULL) {
            unsigned int vendor, product;

            if (sscanf(line, "HID_ID=%*x:%x:%x", &vendor, &product) == 2 &&
                vendor == MCP2221_VENDOR_ID && product == MCP2221_PRODUCT_ID) {
                snprintf(path, size, "/dev/%s", entry->d_name);
                found = 1;
                break;
            }
        }
        fclose(uevent);
    }

    closedir(directory);
    return found ? 0 : -1;
}

static int is_bridge(int fd)
{
    struct hidraw_devinfo info;

    if (ioctl(fd, HIDIOCGRAWINFO, &info) < 0) {
        return 0;
    }

    return (uint16_t)info.vendor == MCP2221_VENDOR_ID &&
           (uint16_t)info.product == MCP2221_PRODUCT_ID;
}

static int read_report(uint8_t *response)
{
    size_t received = 0;

    /* hidraw delivers a whole report per read; the loop also serves pipes. */
    while (received < REPORT_SIZE) {
        struct pollfd waiting;
        ssize_t count;
        int ready;

        waiting.fd = hid_fd;
        waiting.events = POLLIN;
        waiting.revents = 0;
        ready = poll(&waiting, 1, RESPONSE_TIMEOUT_MS);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (ready == 0) {
            errno = ETIMEDOUT;
            return -1;
        }

        count = read(hid_fd, response + received, REPORT_SIZE - received);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (count == 0) {
            errno = EIO;
            return -1;
        }
        received += (size_t)count;
    }

    return 0;
}

static int exchange(const uint8_t *command, uint8_t *response)
{
    /*
     * The bridge uses unnumbered reports, so hidraw expects a leading zero
     * report-ID byte on writes and returns input reports without one.
     */
    uint8_t output[1 + REPORT_SIZE];
    ssize_t written;
    int attempt;

    output[0] = 0;
    memcpy(output + 1, command, REPORT_SIZE);

    written = write(hid_fd, output, sizeof(output));
    if (written < 0) {
        int saved_errno = errno;

        close_bridge();
        return fail("write", saved_errno);
    }
    if (written != (ssize_t)sizeof(output) && written != REPORT_SIZE) {
        close_bridge();
        return fail("short write", EIO);
    }

    for (attempt = 0; attempt <= STALE_RESPONSES_TOLERATED; attempt++) {
        if (read_report(response) != 0) {
            int saved_errno = errno;

            close_bridge();
            return fail("read", saved_errno);
        }
        /* The reply echoes the command code; anything else is left over from
         * a command whose reply was never collected. Skip it. */
        if (response[RSP_ECHO] == command[0]) {
            return 0;
        }
    }

    close_bridge();
    return fail("reply mismatch", EIO);
}

static int send_status_command(int cancel, int set_speed, uint8_t *response)
{
    uint8_t command[REPORT_SIZE];

    memset(command, 0, sizeof(command));
    command[0] = CMD_STATUS_SET_PARAMETERS;
    command[2] = cancel ? PARAM_CANCEL_TRANSFER : 0;
    command[3] = set_speed ? PARAM_SET_SPEED : 0;
    command[4] = set_speed ? I2C_CLOCK_DIVIDER : 0;

    if (exchange(command, response) != 0) {
        return -1;
    }
    if (response[RSP_STATUS] != STATUS_OK) {
        errno = EIO;
        return -1;
    }
    return 0;
}

static int cancel_transfer(void)
{
    /* Also frees the bus after a failed transfer; errno is left untouched. */
    uint8_t response[REPORT_SIZE];
    int saved_errno = errno;
    int result;

    sleep_us(STEP_DELAY_US);
    result = send_status_command(1, 0, response);
    errno = saved_errno;
    return result;
}

static int set_speed(void)
{
    uint8_t response[REPORT_SIZE];
    int attempt;

    for (attempt = 0; attempt < BUSY_RETRIES; attempt++) {
        if (send_status_command(0, 1, response) != 0) {
            return -1;
        }
        if (response[RSP_SPEED_RESULT] == SPEED_ACCEPTED) {
            return 0;
        }
        /* The engine was mid-transfer, for example after an interrupted
         * run of the app; free it and ask again. */
        if (cancel_transfer() != 0) {
            return -1;
        }
        sleep_us(STEP_DELAY_US);
    }

    return fail("set I2C speed", EBUSY);
}

static int open_bridge(void)
{
    const char *configured = getenv(HIDRAW_ENV);
    int fd;

    if (hid_fd >= 0) {
        return 0;
    }

    if (configured != NULL && configured[0] != '\0') {
        snprintf(device_path, sizeof(device_path), "%s", configured);
        device_from_environment = 1;
    } else if (find_bridge_in_sysfs(device_path, sizeof(device_path)) == 0) {
        device_from_environment = 0;
    } else {
        fprintf(stderr,
                "mcp2221: no MCP2221A USB bridge is attached (USB ID "
                "%04x:%04x); set %s to its /dev/hidraw node to override\n",
                MCP2221_VENDOR_ID, MCP2221_PRODUCT_ID, HIDRAW_ENV);
        snprintf(device_path, sizeof(device_path), "(no bridge)");
        errno = ENODEV;
        return -1;
    }

    fd = open(device_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return fail("open", errno);
    }

    /* A node found by searching is checked against the bridge's USB identity
     * so another HID device is never sent I2C commands. An explicitly named
     * node is the operator's choice and may be a test double. */
    if (!device_from_environment && !is_bridge(fd)) {
        close(fd);
        return fail("not an MCP2221A", ENODEV);
    }

    hid_fd = fd;
    if (set_speed() != 0) {
        int saved_errno = errno;

        close_bridge();
        errno = saved_errno;
        return -1;
    }

    return 0;
}

static int engine_state_error(uint8_t state)
{
    switch (state) {
    case I2C_STATE_IDLE:
        return 0;
    case I2C_STATE_ADDRESS_NACK:
    case I2C_STATE_ADDRESS_TIMEOUT:
        return ENXIO;
    case I2C_STATE_START_TIMEOUT:
    case I2C_STATE_REPEATED_START_TIMEOUT:
    case I2C_STATE_DATA_TIMEOUT:
    case I2C_STATE_READ_TIMEOUT:
    case I2C_STATE_STOP_TIMEOUT:
        return ETIMEDOUT;
    default:
        return EIO;
    }
}

static int is_transfer_failure(uint8_t state)
{
    return state == I2C_STATE_ADDRESS_NACK ||
           state == I2C_STATE_ADDRESS_TIMEOUT ||
           state == I2C_STATE_START_TIMEOUT ||
           state == I2C_STATE_REPEATED_START_TIMEOUT ||
           state == I2C_STATE_DATA_TIMEOUT ||
           state == I2C_STATE_READ_TIMEOUT ||
           state == I2C_STATE_STOP_TIMEOUT;
}

static int settle_transfer(void)
{
    /*
     * After the last byte the engine still has to send STOP; wait for it to
     * report idle, and make sure the target acknowledged its address.
     */
    uint8_t response[REPORT_SIZE];
    int poll_count;

    for (poll_count = 0; poll_count < SETTLE_POLLS; poll_count++) {
        uint8_t state;

        if (send_status_command(0, 0, response) != 0) {
            return -1;
        }
        if (response[RSP_STATUS_ADDRESS_ACK] & ADDRESS_NACK_MASK) {
            cancel_transfer();
            errno = ENXIO;
            return -1;
        }

        state = response[RSP_STATUS_I2C_STATE];
        if (state == I2C_STATE_IDLE) {
            return 0;
        }
        if (is_transfer_failure(state)) {
            cancel_transfer();
            errno = engine_state_error(state);
            return -1;
        }
        sleep_us(STEP_DELAY_US);
    }

    cancel_transfer();
    errno = EBUSY;
    return -1;
}

static int submit_data_command(uint8_t *command, uint8_t *response)
{
    /* The engine answers "busy" while it finishes earlier work; wait it out. */
    int busy_retries = 0;

    for (;;) {
        if (exchange(command, response) != 0) {
            return -1;
        }
        if (response[RSP_STATUS] == STATUS_OK) {
            return 0;
        }
        if (response[RSP_STATUS] == STATUS_BUSY &&
            busy_retries++ < BUSY_RETRIES) {
            sleep_us(STEP_DELAY_US);
            continue;
        }

        /* The engine refused the command: NACK, timeout or a stuck bus. */
        errno = engine_state_error(response[RSP_I2C_STATE]);
        if (errno == 0) {
            errno = EBUSY;
        }
        cancel_transfer();
        return -1;
    }
}

static int i2c_write(uint8_t address, const uint8_t *data, size_t length,
                     uint8_t command_code)
{
    size_t sent = 0;

    if (length == 0 || length > UINT16_MAX) {
        errno = EINVAL;
        return -1;
    }

    /* Longer writes continue across reports; the length field always names
     * the whole transfer. The sensor's frames are at most six bytes. */
    do {
        uint8_t command[REPORT_SIZE];
        uint8_t response[REPORT_SIZE];
        size_t chunk = length - sent;

        if (chunk > I2C_CHUNK_SIZE) {
            chunk = I2C_CHUNK_SIZE;
        }

        memset(command, 0, sizeof(command));
        command[0] = command_code;
        command[1] = length & 0xFF;
        command[2] = (length >> 8) & 0xFF;
        command[3] = (uint8_t)(address << 1);
        memcpy(command + 4, data + sent, chunk);

        if (submit_data_command(command, response) != 0) {
            return -1;
        }
        sent += chunk;
        sleep_us(STEP_DELAY_US);
    } while (sent < length);

    /* A write without STOP leaves the engine waiting for the repeated start,
     * so only a complete write is checked for idle and acknowledgement. */
    if (command_code == CMD_I2C_WRITE_NO_STOP) {
        return 0;
    }
    return settle_transfer();
}

static int i2c_read(uint8_t address, uint8_t *data, size_t length,
                    uint8_t command_code)
{
    uint8_t command[REPORT_SIZE];
    uint8_t response[REPORT_SIZE];
    struct timespec started;
    size_t received = 0;

    if (length == 0 || length > UINT16_MAX) {
        errno = EINVAL;
        return -1;
    }

    memset(command, 0, sizeof(command));
    command[0] = command_code;
    command[1] = length & 0xFF;
    command[2] = (length >> 8) & 0xFF;
    command[3] = (uint8_t)((address << 1) | 1);
    if (submit_data_command(command, response) != 0) {
        return -1;
    }

    /* The bridge clocks in up to 60 bytes, hands them over on request, then
     * continues; a reply without data means the chunk is not complete yet. */
    clock_gettime(CLOCK_MONOTONIC, &started);
    while (received < length) {
        uint8_t state;
        uint8_t count;

        sleep_us(STEP_DELAY_US);
        memset(command, 0, sizeof(command));
        command[0] = CMD_I2C_GET_DATA;
        if (exchange(command, response) != 0) {
            return -1;
        }

        /* The engine state is reported whether or not data is ready, and a
         * NACK shows there first: the status byte alone just says "not ready". */
        state = response[RSP_I2C_STATE];
        if (is_transfer_failure(state)) {
            cancel_transfer();
            errno = engine_state_error(state);
            return -1;
        }

        if (response[RSP_STATUS] == STATUS_OK) {
            count = response[RSP_GET_DATA_LENGTH];
            if ((state == I2C_STATE_READ_MORE || state == I2C_STATE_READ_LAST) &&
                count != GET_DATA_LENGTH_ERROR) {
                if (count > I2C_CHUNK_SIZE || received + count > length) {
                    cancel_transfer();
                    errno = EIO;
                    return -1;
                }
                memcpy(data + received, response + RSP_GET_DATA_PAYLOAD, count);
                received += count;
                continue;
            }
        } else if (response[RSP_STATUS] != STATUS_GET_DATA_NOT_READY &&
                   response[RSP_STATUS] != STATUS_BUSY) {
            cancel_transfer();
            errno = EIO;
            return -1;
        }

        if (elapsed_ms(&started) > READ_TIMEOUT_MS) {
            cancel_transfer();
            errno = ETIMEDOUT;
            return -1;
        }
    }

    sleep_us(STEP_DELAY_US);
    return settle_transfer();
}

static void record_failure(const char *what, uint8_t slaveAddr,
                           int error_number)
{
    uint8_t response[REPORT_SIZE];
    int scl = -1;
    int sda = -1;

    /* The status report carries the live SDA/SCL levels: both high means the
     * bus is pulled up and idle, a low line means a short or missing power. */
    if (hid_fd >= 0 && send_status_command(0, 0, response) == 0) {
        scl = response[RSP_STATUS_SCL_LEVEL];
        sda = response[RSP_STATUS_SDA_LEVEL];
    }

    switch (error_number) {
    case ENXIO:
        note_error("%s: no acknowledge from the sensor at 0x%02X "
                   "(SDA=%d SCL=%d)", what, (unsigned)slaveAddr, sda, scl);
        break;
    case ETIMEDOUT:
    case EBUSY:
        note_error("%s: I2C bus timeout at 0x%02X (SDA=%d SCL=%d)", what,
                   (unsigned)slaveAddr, sda, scl);
        break;
    case ESHUTDOWN:
    case ENODEV:
    case ENOENT:
        note_error("%s: the MCP2221A bridge is not on USB (unplugged or "
                   "dropped off; check its cable)", what);
        break;
    case EACCES:
    case EPERM:
        note_error("%s: no permission to open %s; install 99-mcp2221a.rules",
                   what, device_path);
        break;
    default:
        note_error("%s: %s", what, strerror(error_number));
    }

    /* The status exchange above may have changed errno. */
    errno = error_number;
}

static int is_retryable_error(int error_number)
{
    /* These can be transient after a NACK or a bridge that was mid-transfer. */
    return error_number == ENXIO || error_number == EIO ||
           error_number == EAGAIN || error_number == EBUSY ||
           error_number == ETIMEDOUT;
}

int MLX90642_I2CRead(uint8_t slaveAddr, uint16_t startAddress,
                     uint16_t nMemAddressRead, uint16_t *rData)
{
    /*
     * Read framing is one combined transaction:
     *   START + slave(W) + address MSB + address LSB
     *   REPEATED START + slave(R) + 2 bytes per word + STOP
     * The bridge keeps both halves together when the write omits STOP.
     */
    uint8_t register_address[2];
    uint8_t *data;
    size_t read_len;
    int saved_errno = EIO;
    unsigned int attempt;

    /* A zero-word read only has to reach the bridge. */
    if (nMemAddressRead == 0) {
        return open_bridge();
    }

    /* The bridge's length field is 16-bit and the count is in words. */
    if (nMemAddressRead > UINT16_MAX / 2) {
        fprintf(stderr, "i2c read request too large: %u words\n",
                (unsigned)nMemAddressRead);
        errno = EINVAL;
        return -1;
    }

    read_len = (size_t)nMemAddressRead * 2;
    data = malloc(read_len);
    if (data == NULL) {
        perror("malloc read buffer");
        return -1;
    }

    /* MLX90642 memory addresses are transmitted most-significant byte first. */
    register_address[0] = startAddress >> 8;
    register_address[1] = startAddress & 0xFF;

    for (attempt = 1; attempt <= READ_MAX_ATTEMPTS; attempt++) {
        if (open_bridge() == 0 &&
            i2c_write(slaveAddr, register_address, sizeof(register_address),
                      CMD_I2C_WRITE_NO_STOP) == 0 &&
            i2c_read(slaveAddr, data, read_len,
                     CMD_I2C_READ_REPEATED_START) == 0) {
            uint16_t i;

            /* Reassemble each big-endian wire pair into a host-order word. */
            for (i = 0; i < nMemAddressRead; i++) {
                rData[i] = ((uint16_t)data[i * 2] << 8) | data[i * 2 + 1];
            }
            free(data);
            last_error[0] = '\0';
            return 0;
        }

        saved_errno = errno;
        if (!is_retryable_error(saved_errno) || attempt == READ_MAX_ATTEMPTS) {
            break;
        }
        sleep_us(READ_RETRY_DELAY_US * (long)attempt);
    }

    fprintf(stderr,
            "i2c read through mcp2221 failed: device=%s slave=0x%02X "
            "register=0x%04X bytes=%zu attempts=%u: %s\n",
            device_path, (unsigned)slaveAddr, (unsigned)startAddress,
            read_len, attempt > READ_MAX_ATTEMPTS ? READ_MAX_ATTEMPTS : attempt,
            strerror(saved_errno));
    free(data);
    record_failure("i2c read", slaveAddr, saved_errno);
    return -1;
}

static int write_frame(uint8_t slaveAddr, const uint8_t *frame, size_t length,
                       const char *what)
{
    /* Like the I2C header transport, writes are not retried here. */
    if (open_bridge() != 0 ||
        i2c_write(slaveAddr, frame, length, CMD_I2C_WRITE) != 0) {
        int saved_errno = errno;

        fprintf(stderr,
                "%s through mcp2221 failed: device=%s slave=0x%02X "
                "bytes=%zu: %s\n",
                what, device_path, (unsigned)slaveAddr, length,
                strerror(saved_errno));
        record_failure(what, slaveAddr, saved_errno);
        return -1;
    }

    last_error[0] = '\0';
    return 0;
}

int MLX90642_Config(uint8_t slaveAddr, uint16_t writeAddress, uint16_t wData)
{
    /* Configuration frame, all fields big-endian: 0x3A2E opcode + address + value. */
    uint8_t frame[6];

    frame[0] = 0x3A;
    frame[1] = 0x2E;
    frame[2] = writeAddress >> 8;
    frame[3] = writeAddress & 0xFF;
    frame[4] = wData >> 8;
    frame[5] = wData & 0xFF;

    return write_frame(slaveAddr, frame, sizeof(frame), "config write");
}

int MLX90642_I2CCmd(uint8_t slaveAddr, uint16_t i2c_cmd)
{
    /* Command frame: big-endian 0x0180 opcode followed by a big-endian command. */
    uint8_t frame[4];

    frame[0] = 0x01;
    frame[1] = 0x80;
    frame[2] = i2c_cmd >> 8;
    frame[3] = i2c_cmd & 0xFF;

    return write_frame(slaveAddr, frame, sizeof(frame), "command write");
}

int MLX90642_WakeUp(uint8_t slaveAddr)
{
    /* Wake-up is a one-byte write to the device address, then 10 ms of rest. */
    uint8_t dummy = 0x00;

    if (write_frame(slaveAddr, &dummy, 1, "wake write") != 0) {
        return -1;
    }

    sleep_us(10000);
    return 0;
}

void MLX90642_Wait_ms(uint16_t time_ms)
{
    sleep_us((long)time_ms * 1000);
}
