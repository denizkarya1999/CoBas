/*
  Arduino (Wire) transport for the MLX90642 vendor driver: the platform hooks
  declared in MLX90642_depends.h, with the same contract as the Linux
  transports in ../mlx90642-library/src: 0 on success, -1 on a bus failure.
  Reads are split into chunks that fit the Wire buffer; each chunk is its own
  write-address / repeated-start / read transaction at the next address, which
  the sensor's auto-incrementing memory accepts.
*/

#include <Arduino.h>
#include <Wire.h>

extern "C" {
#include <MLX90642_depends.h>
}

// Words per I2C transaction; 128 words = 256 bytes, inside the 512-byte buffer
// the sketch asks Wire for.
static const uint16_t CHUNK_WORDS = 128;
// The sensor can briefly NACK while changing measurement/read-window state.
static const uint8_t READ_MAX_ATTEMPTS = 4;
static const uint16_t READ_RETRY_DELAY_US = 2500;

static int readChunk(uint8_t slaveAddr, uint16_t address, uint16_t words, uint16_t *out)
{
  Wire.beginTransmission(slaveAddr);
  Wire.write((uint8_t)(address >> 8));
  Wire.write((uint8_t)(address & 0xFF));
  // No STOP: the data phase follows with a repeated START.
  if (Wire.endTransmission(false) != 0) {
    return -1;
  }
  size_t wanted = (size_t)words * 2;
  size_t received = Wire.requestFrom((uint8_t)slaveAddr, wanted, (bool)true);
  if (received != wanted) {
    return -1;
  }
  for (uint16_t i = 0; i < words; i++) {
    int high = Wire.read();
    int low = Wire.read();
    out[i] = (uint16_t)(((high & 0xFF) << 8) | (low & 0xFF));
  }
  return 0;
}

extern "C" int MLX90642_I2CRead(uint8_t slaveAddr, uint16_t startAddress,
                                uint16_t nMemAddressRead, uint16_t *rData)
{
  uint16_t done = 0;
  while (done < nMemAddressRead) {
    uint16_t words = nMemAddressRead - done;
    if (words > CHUNK_WORDS) {
      words = CHUNK_WORDS;
    }
    // Memory addresses count bytes: consecutive words are two apart.
    uint16_t address = (uint16_t)(startAddress + done * 2);
    int result = -1;
    for (uint8_t attempt = 1; attempt <= READ_MAX_ATTEMPTS && result != 0; attempt++) {
      result = readChunk(slaveAddr, address, words, rData + done);
      if (result != 0 && attempt < READ_MAX_ATTEMPTS) {
        delayMicroseconds(READ_RETRY_DELAY_US * attempt);
      }
    }
    if (result != 0) {
      return -1;
    }
    done += words;
  }
  return 0;
}

extern "C" int MLX90642_Config(uint8_t slaveAddr, uint16_t writeAddress, uint16_t wData)
{
  // Configuration frame, all fields big-endian: 0x3A2E opcode + address + value.
  Wire.beginTransmission(slaveAddr);
  Wire.write((uint8_t)0x3A);
  Wire.write((uint8_t)0x2E);
  Wire.write((uint8_t)(writeAddress >> 8));
  Wire.write((uint8_t)(writeAddress & 0xFF));
  Wire.write((uint8_t)(wData >> 8));
  Wire.write((uint8_t)(wData & 0xFF));
  return Wire.endTransmission(true) == 0 ? 0 : -1;
}

extern "C" int MLX90642_I2CCmd(uint8_t slaveAddr, uint16_t i2c_cmd)
{
  // Command frame: big-endian 0x0180 opcode followed by a big-endian command.
  Wire.beginTransmission(slaveAddr);
  Wire.write((uint8_t)0x01);
  Wire.write((uint8_t)0x80);
  Wire.write((uint8_t)(i2c_cmd >> 8));
  Wire.write((uint8_t)(i2c_cmd & 0xFF));
  return Wire.endTransmission(true) == 0 ? 0 : -1;
}

extern "C" int MLX90642_WakeUp(uint8_t slaveAddr)
{
  // Wake-up is a one-byte write to the device address, then 10 ms of rest.
  Wire.beginTransmission(slaveAddr);
  Wire.write((uint8_t)0x00);
  int result = Wire.endTransmission(true) == 0 ? 0 : -1;
  delay(10);
  return result;
}

extern "C" void MLX90642_Wait_ms(uint16_t time_ms)
{
  delay(time_ms);
}
