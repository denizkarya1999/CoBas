/*
  CoBas thermal camera on an Arduino Nano ESP32.

  The board reads the Melexis MLX90642 over its 3.3 V I2C bus (A4 = SDA,
  A5 = SCL) with the vendor driver from ../mlx90642-library and streams every
  frame to the host over USB serial, so the Raspberry Pi only has to read a
  serial port. Build and flash with ./build.sh (arduino-cli); the Arduino IDE
  needs the driver's include path, see the README.

  Wiring: camera SDA -> A4, SCL -> A5, GND -> GND, VCC -> VBUS (5 V).

  Host protocol, all little-endian:
    frame packet (1552 bytes):
      0   "CBT" 0x01           magic and version
      4   uint16 sequence
      6   uint16 pixel count   768
      8   uint16 flags         bit 0: sensor reinitialised since the last frame
      10  uint32 millis        board clock when the frame was read
      14  int16[768] pixels    MLX90642 temperature words, 1/50 degree C
      1550 uint16 crc          CRC-16/CCITT-FALSE over bytes 0..1549
    text lines starting with '#' carry status; the host resynchronises on the
    magic, so lines and packets may interleave.
    single-byte commands from the host: 'I' info line, 'R' reinitialise the
    sensor, 'S' start streaming, 'X' stop streaming.
*/

#include <Wire.h>

extern "C" {
#include <MLX90642.h>
// Defined in the vendor source but left out of its header.
int MLX90642_GetRefreshTime(uint8_t slaveAddr);
}

static const uint8_t SENSOR_ADDRESS = SA_90642_DEFAULT;
static const uint16_t FRAME_PIXELS = MLX90642_TOTAL_NUMBER_OF_PIXELS;
static const uint32_t I2C_CLOCK_HZ = 400000;
static const uint32_t I2C_FALLBACK_CLOCK_HZ = 100000;
static const uint16_t FRAME_FAILURES_BEFORE_REINIT = 3;
static const uint16_t WINDOW_POLL_LIMIT = 1000;
static const uint8_t MAGIC[4] = {'C', 'B', 'T', 0x01};

static uint16_t frame[FRAME_PIXELS];
static uint16_t sequence = 0;
static bool streaming = true;
static bool sensor_ready = false;
static bool reinitialised = false;
static uint16_t frame_failures = 0;
static uint16_t init_failures = 0;
static uint32_t i2c_clock_hz = I2C_CLOCK_HZ;
static int last_status = 0;

static uint16_t crc16(const uint8_t *data, size_t length)
{
  // CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xFFFF.
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void put16(uint8_t *out, uint16_t value)
{
  out[0] = value & 0xFF;
  out[1] = value >> 8;
}

static void put32(uint8_t *out, uint32_t value)
{
  for (int i = 0; i < 4; i++) {
    out[i] = (value >> (8 * i)) & 0xFF;
  }
}

static void sendInfo()
{
  uint16_t id[MLX90642_NUMBER_OF_ID_WORDS] = {0};
  uint8_t fw[2] = {0};
  int refresh_ms = -1;
  if (sensor_ready) {
    MLX90642_GetID(SENSOR_ADDRESS, id);
    MLX90642_GetFWver(SENSOR_ADDRESS, fw);
    refresh_ms = MLX90642_GetRefreshTime(SENSOR_ADDRESS);
  }
  Serial.printf("# cobas-mlx90642 board=nano-esp32 sensor=0x%02X ready=%d streaming=%d "
                "id=%04X-%04X-%04X-%04X sensor_fw=%u.%u refresh_ms=%d i2c_hz=%lu "
                "last_status=%d frames=%u\n",
                SENSOR_ADDRESS, sensor_ready, streaming, id[0], id[1], id[2], id[3],
                fw[0], fw[1], refresh_ms, (unsigned long)i2c_clock_hz, last_status, sequence);
}

static void setI2CClock(uint32_t hz)
{
  i2c_clock_hz = hz;
  Wire.setClock(hz);
}

static bool initialiseSensor()
{
  // The driver synchronises a measurement and waits for the first frame; it
  // gives the sensor little slack beyond its nominal period, so retry.
  last_status = MLX90642_Init(SENSOR_ADDRESS);
  if (last_status == 0) {
    sensor_ready = true;
    reinitialised = true;
    init_failures = 0;
    frame_failures = 0;
    Serial.printf("# sensor ready at 0x%02X, i2c %lu Hz\n", SENSOR_ADDRESS, (unsigned long)i2c_clock_hz);
    return true;
  }
  sensor_ready = false;
  init_failures++;
  if (init_failures == 3 && i2c_clock_hz != I2C_FALLBACK_CLOCK_HZ) {
    // Long jumper wires sometimes need the slower clock.
    setI2CClock(I2C_FALLBACK_CLOCK_HZ);
    Serial.printf("# init keeps failing (%d); trying i2c %lu Hz\n", last_status, (unsigned long)i2c_clock_hz);
  } else {
    Serial.printf("# init failed: status %d (-1 = no acknowledge from 0x%02X, -3 = timeout)\n",
                  last_status, SENSOR_ADDRESS);
  }
  return false;
}

static void sendFrame()
{
  static uint8_t packet[14 + FRAME_PIXELS * 2 + 2];
  memcpy(packet, MAGIC, 4);
  put16(packet + 4, sequence);
  put16(packet + 6, FRAME_PIXELS);
  put16(packet + 8, reinitialised ? 1 : 0);
  put32(packet + 10, millis());
  for (uint16_t i = 0; i < FRAME_PIXELS; i++) {
    put16(packet + 14 + i * 2, frame[i]);
  }
  put16(packet + 14 + FRAME_PIXELS * 2, crc16(packet, 14 + FRAME_PIXELS * 2));
  Serial.write(packet, sizeof(packet));
  reinitialised = false;
  sequence++;
}

static int waitForNextWindow()
{
  // Require a closed-then-open read window so the same frame is never sent
  // twice; mirrors MLX90642_PythonWaitForNextFrame in the Linux build.
  bool saw_closed = false;
  for (uint16_t poll = 0; poll < WINDOW_POLL_LIMIT; poll++) {
    int status = MLX90642_IsReadWindowOpen(SENSOR_ADDRESS);
    if (status < 0) {
      return status;
    }
    if (status == MLX90642_NO) {
      saw_closed = true;
    } else if (saw_closed) {
      return 0;
    }
    handleCommands();
    MLX90642_Wait_ms(MLX90642_POLL_TIME_MS);
  }
  return -MLX90642_TIMEOUT_ERR;
}

static void handleCommands()
{
  while (Serial.available() > 0) {
    int command = Serial.read();
    switch (command) {
      case 'I': sendInfo(); break;
      case 'R': sensor_ready = false; init_failures = 0; setI2CClock(I2C_CLOCK_HZ); break;
      case 'S': streaming = true; break;
      case 'X': streaming = false; break;
      default: break;
    }
  }
}

void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  // Frames are read in chunks that fit this buffer (see MLX90642_arduino_i2c.cpp).
  Wire.setBufferSize(512);
  Wire.begin();
  setI2CClock(I2C_CLOCK_HZ);
  delay(100);
  Serial.println("# cobas-mlx90642 starting");
}

void loop()
{
  handleCommands();

  if (!sensor_ready) {
    if (!initialiseSensor()) {
      delay(500);
    }
    return;
  }

  if (!streaming) {
    delay(20);
    return;
  }

  last_status = MLX90642_GetImage(SENSOR_ADDRESS, frame);
  if (last_status < 0) {
    frame_failures++;
    Serial.printf("# frame read failed: status %d (%u in a row)\n", last_status, frame_failures);
    if (frame_failures >= FRAME_FAILURES_BEFORE_REINIT) {
      sensor_ready = false;
    }
    delay(50);
    return;
  }
  frame_failures = 0;
  digitalWrite(LED_BUILTIN, HIGH);
  sendFrame();
  digitalWrite(LED_BUILTIN, LOW);

  last_status = waitForNextWindow();
  if (last_status < 0) {
    Serial.printf("# waiting for the next frame failed: status %d\n", last_status);
    frame_failures++;
    if (frame_failures >= FRAME_FAILURES_BEFORE_REINIT) {
      sensor_ready = false;
    }
  }
}
