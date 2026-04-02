# MagicBaton

A gesture-recognition conducting baton powered by dual IMUs and machine learning. Wave, strike, spin — the baton knows what you did.

## Hardware

- **ESP32** microcontroller
- **2x MPU6050** IMUs (I2C addresses `0x68` and `0x69`) mounted on a baton
- I2C: SDA=GPIO21, SCL=GPIO22, 400kHz
- Communication: BLE (device name `MagicBaton`) + USB Serial (115200 baud)

## Data Format

12-axis raw sensor data at ~50Hz:

```
A,ax,ay,az,gx,gy,gz|B,ax,ay,az,gx,gy,gz
```

- **A** = IMU at address `0x68`, **B** = IMU at address `0x69`
- Accelerometer: 16-bit signed, default ±2g (raw ±16384 per g)
- Gyroscope: 16-bit signed, default ±250 deg/s (raw ±32768)

## Project Structure

```
MagicBaton/
├── HW/
│   ├── baton_bluetooth/
│   │   └── baton_bluetooth.ino   # ESP32 firmware (BLE + Serial)
│   ├── baton_ap/
│   │   └── baton_ap.ino          # WiFi AP firmware (alternative)
│   └── baton_imu/
│       └── baton_imu.ino         # WiFi STA firmware (alternative)
├── record.py                      # GUI data recorder for ML training
├── server.py                      # WebSocket bridge (Serial → browser)
├── index.html                     # Real-time dashboard + visualizer
├── data/                          # Recorded gesture data (CSV + metadata)
│   └── metadata.json
└── README.md
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Links2568/MagicBaton.git
cd MagicBaton
pip install -r requirements.txt

# 2. Flash firmware (connect ESP32 via USB)
arduino-cli compile --fqbn esp32:esp32:esp32 HW/baton_bluetooth/baton_bluetooth.ino
arduino-cli upload --fqbn esp32:esp32:esp32 --port /dev/cu.usbserial-0001 HW/baton_bluetooth/baton_bluetooth.ino

# 3. Record gesture data (this is what you need for ML training)
python record.py

# 4. (Optional) Run debug dashboard for real-time visualization
#    This is a separate tool for debugging and future demo purposes.
#    NOT required for data collection.
python server.py        # reads USB Serial, serves WebSocket on :8765
open index.html         # open in browser, connects to server.py
```

## Gesture Reference

10 gestures, mapped to keys `0-9` in the recorder:

### 0 — idle (stationary)

```
  ___
```

Hold the baton still. Record several reps in different orientations: vertical, horizontal, resting on a table. This is the negative class — the model needs to know what "no gesture" looks like.

### 1 — beat (downward strike)

```
    |
    |
    v
```

A conductor's downbeat. Swing the baton downward in a quick, decisive stroke and stop at the bottom. One clean hit.

### 2 — stab (forward thrust)

```
  ------>
```

Thrust the baton forward along its long axis, like a fencing lunge. Both IMUs accelerate in the same direction — this is what distinguishes it from a swing.

### 3 — spin (draw a circle)

```
    ╭──╮
    │  │
    ╰──╯
```

Move the baton tip in a full circle in the air. Clockwise or counterclockwise, one complete loop. Keep it smooth and round.

### 4 — infinity (draw a figure-8)

```
   ╭╮ ╭╮
   ╰╯ ╰╯
```

Trace a horizontal figure-8 (infinity symbol) in the air. Right loop, then left loop, one continuous motion. The direction reversal halfway through is the key signal.

### 5 — slash (diagonal cut)

```
  ╲
   ╲
    ╲
```

A diagonal swipe from upper-left to lower-right (or the reverse). Like a sword slash. Similar to beat but at a ~45-degree angle.

### 6 — shake (tremolo)

```
  ←→←→←→
```

Rapidly shake the baton side to side for 1-2 seconds, like a tremolo. The signal is high-frequency alternating acceleration, sustained over time.

### 7 — flick (wrist snap)

```
  ⤴
```

A single sharp wrist snap — extremely quick and short. Like flicking water off your fingers. Over in a fraction of a second.

### 8 — wing (draw a W)

```
  /\  /\
 /  \/  \
```

Trace the letter W in the air with the baton tip. Zig-zag up and down, roughly 2 seconds for the full shape.

### 9 — slope (draw a checkmark)

```
  \
   \___
```

Diagonal stroke down-left, then a horizontal stroke to the right. Like drawing a checkmark. About 1 second total.

## Recording Tips

- Start with the most distinct gestures: `idle`, `beat`, `stab`, `spin`
- Record 25-30 reps per gesture
- Vary your speed and intensity across reps — don't be too consistent
- Keep the baton orientation the same between recording and inference
- The recorder saves each rep as a separate CSV in `data/`
- `metadata.json` tracks all sessions for easy loading during training

## Architecture

```
ESP32 + 2x MPU6050
       │
       ├── BLE notify ──→ record.py (GUI recorder, saves CSV)
       │
       └── USB Serial ──→ server.py ──→ WebSocket ──→ index.html (dashboard)
```

The dashboard displays real-time sensor waveforms, beat detection (jerk-based), BPM tracking, dynamics (pp-ff), and a visual effects canvas that responds to baton motion.

## License

See [LICENSE](LICENSE).
