#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include "MLX90642_depends.h"

#define I2C_BUS "/dev/i2c-1"

static const char *i2c_bus_path(void)
{
    const char *bus_path = getenv("MLX90642_I2C_BUS");

    if (bus_path != NULL && bus_path[0] != '\0') {
        return bus_path;
    }

    return I2C_BUS;
}

static int open_i2c(uint8_t slaveAddr)
{
    int fd = open(i2c_bus_path(), O_RDWR);
    if (fd < 0) {
        perror("open i2c");
        return -1;
    }

    if (ioctl(fd, I2C_SLAVE, slaveAddr) < 0) {
        perror("ioctl i2c");
        close(fd);
        return -1;
    }

    return fd;
}

int MLX90642_I2CRead(uint8_t slaveAddr, uint16_t startAddress,
                     uint16_t nMemAddressRead, uint16_t *rData)
{
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    if (nMemAddressRead == 0) {
        close(fd);
        return 0;
    }

    if (nMemAddressRead > UINT16_MAX / 2) {
        fprintf(stderr, "i2c read request too large: %u words\n",
                (unsigned)nMemAddressRead);
        close(fd);
        return -1;
    }

    size_t read_len = (size_t)nMemAddressRead * 2;
    uint8_t *data = malloc(read_len);
    if (data == NULL) {
        perror("malloc read buffer");
        close(fd);
        return -1;
    }

    uint8_t addr_buf[2];
    addr_buf[0] = startAddress >> 8;
    addr_buf[1] = startAddress & 0xFF;

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

    if (ioctl(fd, I2C_RDWR, &transaction) < 0) {
        perror("i2c read transaction");
        free(data);
        close(fd);
        return -1;
    }

    for (uint16_t i = 0; i < nMemAddressRead; i++) {
        rData[i] = ((uint16_t)data[i * 2] << 8) | data[i * 2 + 1];
    }

    free(data);
    close(fd);
    return 0;
}

int MLX90642_Config(uint8_t slaveAddr, uint16_t writeAddress, uint16_t wData)
{
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t buf[6];

    buf[0] = 0x3A;
    buf[1] = 0x2E;
    buf[2] = writeAddress >> 8;
    buf[3] = writeAddress & 0xFF;
    buf[4] = wData >> 8;
    buf[5] = wData & 0xFF;

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
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t buf[4];

    buf[0] = 0x01;
    buf[1] = 0x80;
    buf[2] = i2c_cmd >> 8;
    buf[3] = i2c_cmd & 0xFF;

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
    int fd = open_i2c(slaveAddr);
    if (fd < 0) return -1;

    uint8_t dummy = 0x00;
    if (write(fd, &dummy, 1) != 1) {
        perror("wake write");
        close(fd);
        return -1;
    }

    close(fd);
    usleep(10000);
    return 0;
}

void MLX90642_Wait_ms(uint16_t time_ms)
{
    usleep(time_ms * 1000);
}
