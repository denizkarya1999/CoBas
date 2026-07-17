#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include "MLX90642_depends.h"

/*
 * Linux userspace transport for the MLX90642.  It relies on the i2c-dev
 * interface; in particular, block reads require an adapter that supports the
 * I2C_RDWR ioctl used for combined transactions.  Transport failures are
 * reported to stderr and returned as -1; this layer does not translate errno
 * values into the higher-level MLX90642 error constants.
 */

/* Default Raspberry Pi I2C bus; a nonempty environment override wins below. */
#define I2C_BUS "/dev/i2c-1"

/* The sensor can briefly NACK while changing measurement/read-window state. */
#define I2C_READ_MAX_ATTEMPTS 4
#define I2C_READ_RETRY_DELAY_US 2500

static const char *i2c_bus_path(void)
{
    const char *bus_path = getenv("MLX90642_I2C_BUS");

    /* Treat an unset or empty override as a request to use the default bus. */
    if (bus_path != NULL && bus_path[0] != '\0') {
        return bus_path;
    }

    return I2C_BUS;
}

static int open_i2c(uint8_t slaveAddr)
{
    /*
     * Each bus transaction owns a fresh descriptor, so there is no shared
     * descriptor or selected-slave state between calls.  Linux expects the
     * unshifted 7-bit slave address here, not the address/RW byte seen on the
     * wire.
     */
    int fd = open(i2c_bus_path(), O_RDWR);
    if (fd < 0) {
        perror("open i2c");
        return -1;
    }

    /* This selection is used by the plain write(2) operations below. */
    if (ioctl(fd, I2C_SLAVE, slaveAddr) < 0) {
        perror("ioctl i2c");
        close(fd);
        return -1;
    }

    return fd;
}

static int is_retryable_i2c_error(int error_number)
{
    /* These errors can be transient after a NACK or controller arbitration. */
    return error_number == EREMOTEIO || error_number == EIO ||
           error_number == ENXIO || error_number == EAGAIN ||
           error_number == EBUSY || error_number == ETIMEDOUT;
}

int MLX90642_I2CRead(uint8_t slaveAddr, uint16_t startAddress,
                     uint16_t nMemAddressRead, uint16_t *rData)
{
    /*
     * Read framing is one combined transaction:
     *   START + slave(W) + address MSB + address LSB
     *   REPEATED START + slave(R) + 2 bytes per word + STOP
     * I2C_RDWR keeps both messages together without an intervening STOP.
     * For a nonzero count, rData is assumed to point to enough caller-owned
     * storage; this transport does not validate that pointer.
     */
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    /* A zero-word read succeeds without allocating or dereferencing rData. */
    if (nMemAddressRead == 0) {
        close(fd);
        return 0;
    }

    /* i2c_msg.len is 16-bit and the requested count is measured in words. */
    if (nMemAddressRead > UINT16_MAX / 2) {
        fprintf(stderr, "i2c read request too large: %u words\n",
                (unsigned)nMemAddressRead);
        close(fd);
        return -1;
    }

    /*
     * The temporary byte buffer mirrors the device's raw wire representation.
     */
    size_t read_len = (size_t)nMemAddressRead * 2;
    uint8_t *data = malloc(read_len);
    if (data == NULL) {
        perror("malloc read buffer");
        close(fd);
        return -1;
    }

    /* MLX90642 memory addresses are transmitted most-significant byte first. */
    uint8_t addr_buf[2];
    addr_buf[0] = startAddress >> 8;
    addr_buf[1] = startAddress & 0xFF;

    /*
     * Message zero writes the address; message one reads the contiguous data.
     */
    struct i2c_msg messages[2];
    messages[0].addr = slaveAddr;
    messages[0].flags = 0;
    messages[0].len = (uint16_t)sizeof(addr_buf);
    messages[0].buf = addr_buf;
    messages[1].addr = slaveAddr;
    messages[1].flags = I2C_M_RD;
    messages[1].len = (uint16_t)read_len;
    messages[1].buf = data;

    struct i2c_rdwr_ioctl_data transaction;
    transaction.msgs = messages;
    transaction.nmsgs = 2;

    /*
     * A target can briefly NACK with EREMOTEIO around a measurement boundary.
     * Retry only transient bus failures, reopening the descriptor so the next
     * attempt starts with clean adapter/file state.  I2C_RDWR returns the
     * number of transferred messages; a short nonnegative result is also a
     * failed combined transaction and is retried as EIO.
     */
    int transaction_result = -1;
    int saved_errno = EIO;
    unsigned int attempt;

    for (attempt = 1; attempt <= I2C_READ_MAX_ATTEMPTS; attempt++) {
        errno = 0;
        transaction_result = ioctl(fd, I2C_RDWR, &transaction);
        if (transaction_result == (int)transaction.nmsgs) {
            break;
        }

        saved_errno = transaction_result < 0 ? errno : EIO;
        if (!is_retryable_i2c_error(saved_errno) ||
            attempt == I2C_READ_MAX_ATTEMPTS) {
            fprintf(stderr,
                    "i2c read transaction failed: bus=%s slave=0x%02X "
                    "register=0x%04X bytes=%zu attempts=%u/%u: %s\n",
                    i2c_bus_path(), (unsigned)slaveAddr,
                    (unsigned)startAddress, read_len, attempt,
                    I2C_READ_MAX_ATTEMPTS, strerror(saved_errno));
            free(data);
            close(fd);
            errno = saved_errno;
            return -1;
        }

        close(fd);
        usleep(I2C_READ_RETRY_DELAY_US * attempt);
        fd = open_i2c(slaveAddr);
        if (fd < 0) {
            free(data);
            return -1;
        }
    }

    /* Reassemble each big-endian wire pair into a host-order 16-bit word. */
    for (uint16_t i = 0; i < nMemAddressRead; i++) {
        rData[i] = ((uint16_t)data[i * 2] << 8) | data[i * 2 + 1];
    }

    free(data);
    close(fd);
    return 0;
}

int MLX90642_Config(uint8_t slaveAddr, uint16_t writeAddress, uint16_t wData)
{
    /*
     * Configuration frame (all 16-bit fields are big-endian):
     *   0x3A2E opcode + target memory address + value.
     */
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t buf[6];

    buf[0] = 0x3A;
    buf[1] = 0x2E;
    buf[2] = writeAddress >> 8;
    buf[3] = writeAddress & 0xFF;
    buf[4] = wData >> 8;
    buf[5] = wData & 0xFF;

    /*
     * Reject both syscall errors and short frames; this transport does not
     * retry.
     */
    if (write(fd, buf, 6) != 6) {
        perror("config write");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

int MLX90642_I2CCmd(uint8_t slaveAddr, uint16_t i2c_cmd)
{
    /*
     * Command frame: big-endian 0x0180 opcode followed by a big-endian command.
     */
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t buf[4];

    buf[0] = 0x01;
    buf[1] = 0x80;
    buf[2] = i2c_cmd >> 8;
    buf[3] = i2c_cmd & 0xFF;

    /* Only a complete four-byte frame is accepted as success. */
    if (write(fd, buf, 4) != 4) {
        perror("command write");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

int MLX90642_WakeUp(uint8_t slaveAddr)
{
    /*
     * This backend realizes wake-up as a one-byte write to the device address.
     */
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t dummy = 0x00;
    if (write(fd, &dummy, 1) != 1) {
        perror("wake write");
        close(fd);
        return -1;
    }

    close(fd);
    /* Allow 10 ms for wake-up; an interrupted sleep is not resumed. */
    usleep(10000);
    return 0;
}

void MLX90642_Wait_ms(uint16_t time_ms)
{
    /* POSIX usleep takes microseconds; an interrupted delay is not resumed. */
    usleep(time_ms * 1000);
}
