# IWR6843AOP USB1 complex I/Q logger

This no-argument program configures and starts the radar through USB0, decodes
complex range-bin symbols from the SDK binary stream on USB1, displays selected
samples in the terminal, and records every sample in a named CSV under the
`Logs` folder.

## Fixed configuration

All values and the complete radar profile are embedded in `iq_logic.py`:

```text
CLI/control: /dev/ttyUSB0 at 115200 baud
Binary data: /dev/ttyUSB1 at 921600 baud
Range bins: 64
Frame period: 120 ms
Sample layout: signed int16 imaginary, then signed int16 real
```

There are no command-line arguments or external configuration files.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 start_raw_iq_logger.py
```

At startup, enter the desired log name. The `.csv` extension is added
automatically when omitted:

```text
Enter log name: empty_room_test
```

USB1 is opened before the embedded profile is sent through USB0, so early
frames are not lost. Press `Ctrl+C` to stop; the script also sends `sensorStop`
and closes both UARTs.

The terminal shows antenna 0 at every eighth range bin to avoid falling behind
the serial stream. Every sample is written to the CSV using only the values
needed to construct a range-azimuth heatmap:

```text
frame_number,range_bin,virtual_antenna,i,q
```

Magnitude and phase are intentionally omitted because both can be derived from
I and Q. Timestamps and global sample indices are also unnecessary because the
frame, range-bin, and virtual-antenna fields fully locate each matrix value.

New captures are saved as:

```text
Logs/<entered_log_name>.csv
```

Path separators are rejected, and an existing log is never overwritten. The
program asks for another name instead.

## Signal meaning

For the IWR6843AOP SDK 3.x out-of-box firmware, enabling the range-azimuth
output produces TLV type 8: the zero-Doppler range-FFT matrix for all virtual
antennas. The parser also recognizes legacy azimuth TLV type 4. TI stores each
complex symbol as imaginary first and real second; the CSV maps real to `i` and
imaginary to `q`.

These values are complex range-FFT I/Q available through USB1. They are
processed radar-cube symbols, not raw time-domain ADC samples. Raw ADC capture
still requires LVDS/DCA1000 or different radar firmware.

## Code structure

- `start_raw_iq_logger.py`: terminal interface and application entry point.
- `iq_logic.py`: hard-coded radar profile, UART control, TLV parsing, and CSV.
