# MagicBaton

A gesture-recognition conducting baton powered by dual IMUs and machine learning. Wave, strike, spin — the baton knows what you did.

## Demo

[▶ Watch the demo](docs/demo.mp4)

https://github.com/Links2568/MagicBaton/raw/main/docs/demo.mp4

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
├── firmware/                   # ESP32 Arduino sketches
│   ├── baton_bluetooth/        # BLE + USB Serial (recommended)
│   ├── baton_ap/               # WiFi AP (alternative)
│   └── baton_imu/              # WiFi STA (alternative)
├── dataset/                    # Public gesture dataset (4 subjects, ~934 CSVs)
│   ├── README.md
│   └── {zuchen,yoyo,tiffany,xiaolan}/
├── models/                     # Pre-trained weights
│   ├── linknet.pt              # single-subject CNN
│   ├── linknet_cross.pt        # cross-subject CNN
│   └── beat_svm.pkl            # beat-detection SVM
├── training/                   # Training + plotting scripts
│   ├── train_linknet.py             # single-subject (zuchen)
│   ├── train_linknet_cross.py       # cross-subject, 4-subject pool
│   ├── train_linknet_lopo_compare.py# LOPO vs 5-fold comparison
│   ├── train_linknet_rigid.py       # rigid-body feature ablation
│   ├── train_beat_svm.py            # beat-only SVM classifier
│   └── plot_*.py                    # figure generators
├── realtime/                   # Live demo and data capture
│   ├── server.py               # BLE → WebSocket bridge, runs inference
│   ├── record.py               # GUI data recorder
│   ├── index.html              # Dashboard + MIDI player
│   └── gesture_viz.html        # Gesture visualization
├── results/                    # Experiment figures and reports
├── docs/                       # Images used in this README
└── README.md
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Links2568/MagicBaton.git
cd MagicBaton
pip install -r requirements.txt

# 2. Flash firmware (connect ESP32 via USB)
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/baton_bluetooth/baton_bluetooth.ino
arduino-cli upload --fqbn esp32:esp32:esp32 --port /dev/cu.usbserial-0001 firmware/baton_bluetooth/baton_bluetooth.ino

# 3a. Try the pre-trained models in the live dashboard
python realtime/server.py        # loads models/ and serves WebSocket on :8765
open realtime/index.html         # open in browser, connects to server.py

# 3b. Or record new gesture data for training
python realtime/record.py        # saves to realtime/data/

# 4. Retrain from the public dataset
python training/train_linknet.py          # single-subject CNN/RNN/SVM/RF
python training/train_linknet_cross.py    # 4-subject pooled, LOPO + 5-fold
python training/train_beat_svm.py         # beat-detection SVM for realtime
```

See [`dataset/README.md`](dataset/README.md) for the dataset schema.

## Gesture Reference

8 gestures, mapped to keys `0-7` in the recorder:

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

### 4 — slash (diagonal cut)

```
  ╲
   ╲
    ╲
```

A diagonal swipe from upper-left to lower-right (or the reverse). Like a sword slash. Similar to beat but at a ~45-degree angle.

### 5 — shake (tremolo)

```
  ←→←→←→
```

Rapidly shake the baton side to side for 1-2 seconds, like a tremolo. The signal is high-frequency alternating acceleration, sustained over time.

### 6 — flick (wrist snap)

```
  ⤴
```

A single sharp wrist snap — extremely quick and short. Like flicking water off your fingers. Over in a fraction of a second.

### 7 — wing (draw a W)

```
  /\  /\
 /  \/  \
```

Trace the letter W in the air with the baton tip. Zig-zag up and down, roughly 2 seconds for the full shape.

## How to Record

1. Run `python realtime/record.py` and wait for BLE connection
2. **Click** the gesture you want to record in the left panel
3. Get the baton into the **starting position** for that gesture
4. Press **Space** to start recording
5. **Perform the gesture once**, cleanly
6. Press **Space** to stop recording immediately after the gesture ends
7. Repeat from step 4 for the next rep (same gesture)
8. Click a different gesture to switch, or press **Undo Last** if you messed up

### Important: Keep recordings clean

Each recording should contain **only the gesture itself** — nothing before, nothing after.

- **DO**: Press Space right before you start moving, stop right after you finish
- **DON'T**: Include the wind-up, the recovery, or the return to rest position
- **DON'T**: Include pauses or idle time at the start/end of a recording
- For `stab`: record only the forward thrust, NOT the pull-back
- For `beat`: record only the downswing, NOT the arm raising back up
- For `shake`: record the sustained shaking, trim the start/stop transitions

Bad example (too much noise):
```
[idle...idle...] [GESTURE] [recovery...idle...]
                 ^^^^^^^^^ only this part
```

Good example:
```
[GESTURE]
^^^^^^^^^ the entire recording
```

### General Tips

- Start with the most distinct gestures: `idle`, `beat`, `stab`, `spin`
- Record 25-30 reps per gesture
- Vary your speed and intensity across reps — don't be too consistent
- Keep the baton orientation the same between recording and inference
- The recorder saves each rep as a separate CSV in `realtime/data/`
- `metadata.json` tracks all sessions for easy loading during training

## Architecture

```
ESP32 + 2x MPU6050
       │
       ├── BLE notify ──→ realtime/record.py   (GUI recorder, saves CSV)
       │
       └── USB Serial ──→ realtime/server.py ──→ WebSocket ──→ realtime/index.html (dashboard)
```

The dashboard displays real-time sensor waveforms, beat detection (jerk-based), BPM tracking, dynamics (pp-ff), and a visual effects canvas that responds to baton motion.

## Beat Detection

The dashboard detects conducting beats from IMU data using a **jerk-based rising-edge detector** with cooldown debouncing.

### How it works

Jerk (the rate of change of acceleration) is used instead of raw acceleration because the defining moment of a conductor's beat is the sudden change — the wrist stopping or reversing direction — not the acceleration magnitude itself.

A beat is registered when three conditions are met simultaneously:

1. **Threshold crossing**: `combinedJerk >= beatThreshold` — the current frame's jerk exceeds the detection threshold.
2. **Rising edge**: `prevJerkA < beatThreshold` — the previous frame was still below the threshold. This ensures the beat fires only once at the moment jerk crosses the threshold, not continuously while jerk stays high.
3. **Cooldown**: `(now - lastBeatTime) > 180` — at least 180ms since the last beat. This caps detection at ~333 BPM and prevents a single vigorous swing from producing multiple false triggers due to oscillation.

### On beat trigger

When a beat is detected, the system:

- Records the timestamp in `beatTimes[]` (capped at 16 entries), which feeds the BPM calculator.
- Extracts the hit direction from IMU A's raw accelerometer values (`aa[0] / 16384` for X, `-aa[1] / 16384` for Y, where 16384 is the MPU6050's raw count per g at ±2g range) and spawns directional visual particle effects.
- Plays a beat sound effect only when MIDI playback is inactive, to avoid clashing with the music.
- Maps jerk magnitude to musical dynamics notation (`pp` → `ff`) based on intensity thresholds:

| Jerk         | Dynamics |
|-------------|----------|
| > 15000     | ff       |
| > 10000     | f        |
| > 6000      | mf       |
| > 3000      | p        |
| ≤ 3000      | pp       |

### BPM calculation

Beat timestamps older than 3 seconds are expired first. If at least 2 beats remain, the system computes inter-beat intervals, takes only the **last 4 intervals** for averaging, and converts to BPM via `60000 / avg`. Using only the 4 most recent intervals (roughly 2–3 seconds of data) balances stability against responsiveness — older intervals would slow down the response when the conductor accelerates or decelerates.

If fewer than 2 valid beats exist (e.g., the conductor stops), `currentBPM` drops to 0.

### Background energy

```js
bgEnergy = bgEnergy * 0.92 + (jerkDecay / 12000) * 0.08;
```

An exponential moving average (EMA) that smoothly tracks overall motion intensity. 92% old value + 8% new value means `bgEnergy` responds slowly to changes, driving background visual brightness/activity without flickering on single-frame jerk spikes.

## MIDI Player

The dashboard includes a browser-based MIDI player that can sync its playback speed to the conducting baton in real time.

### Dependencies

- **[@tonejs/midi](https://github.com/Tonejs/Midi)** (`Midi` class): Parses `.mid` files into structured JavaScript objects containing tracks, notes, and header metadata (PPQ, tempos).
- **[Tone.js](https://tonejs.github.io/)** (`Tone.Transport`, `Tone.PolySynth`): Provides the audio engine, scheduling transport, and synthesizer.

### MIDI parsing and note scheduling

MIDI files are parsed into a structure like:

```js
midiParsedData = {
  header: { ppq: 480, tempos: [{ bpm: 76 }] },
  tracks: [
    { notes: [{ name: "C4", ticks: 0, durationTicks: 240, velocity: 0.8 }, ...] },
    ...
  ]
}
```

The original BPM is extracted from `header.tempos[0].bpm` (defaults to 120 if absent). Notes are scheduled onto `Tone.Transport` using **tick-based timing** rather than absolute seconds:

```js
const st = Math.round(note.ticks * tonePPQ / midiPPQ);
const dur = Math.max(1, Math.round(note.durationTicks * tonePPQ / midiPPQ));
Tone.Transport.schedule((time) => {
  midiSynth.triggerAttackRelease(note.name, dur + 'i', time, note.velocity);
}, st + 'i');
```

The PPQ ratio (`tonePPQ / midiPPQ`) rescales between the MIDI file's pulses-per-quarter-note and Tone.js Transport's internal PPQ, preserving all rhythmic relationships. The `'i'` suffix tells Tone.js the unit is ticks (not seconds), so the real-time playback position of each note is determined entirely by `Tone.Transport.bpm`. Changing BPM changes the speed of everything proportionally, without re-scheduling any notes.

### Tempo synchronization

The function `updateMidiTempo(bpm)` bridges beat detection to MIDI playback:

```js
function updateMidiTempo(bpm) {
  if (!midiIsPlaying || !midiFollowBaton || !window.Tone) return;
  if (bpm > 20 && bpm < 300) {
    Tone.Transport.bpm.rampTo(bpm, 0.15);
  } else {
    Tone.Transport.bpm.rampTo(midiOrigBPM, 1.0);
  }
}
```

- **Valid BPM (20–300)**: `rampTo(bpm, 0.15)` smoothly transitions to the new tempo over 0.15 seconds. Smooth ramping is essential because IMU data updates every frame — hard-setting BPM each frame would cause audible clock jitter. 0.15s is fast enough to feel responsive but smooth enough to avoid artifacts.
- **Invalid/zero BPM** (conductor stops or data anomaly): `rampTo(midiOrigBPM, 1.0)` slowly drifts back to the original tempo over 1 second, so the music gracefully returns to its natural pace rather than abruptly snapping.

### Follow Baton toggle

When the user unchecks "Follow Baton", the system immediately hard-sets `Tone.Transport.bpm.value = midiOrigBPM` (no ramp) for instant response to user intent. When re-enabled, real-time tempo sync resumes on the next frame.

### Playback controls

- **Play**: Unlocks the browser audio context (`Tone.start()` — required by Chrome's autoplay policy) and starts `Tone.Transport`.
- **Pause**: Freezes the transport clock; playback resumes from the paused position.
- **Stop**: Resets the transport to position 0 and calls `midiSynth.releaseAll()` to immediately silence any sustained notes.
- **Loop**: Toggles `Tone.Transport.loop` with boundaries set from tick 0 to `maxTick + 1 beat` of padding.

### MIDI library

A built-in library of base64-encoded MIDI files (e.g., Beethoven Symphony No. 7 Mov. 2) is stored in the `MIDI_LIBRARY` object. These are decoded at load time via `atob()` → `Uint8Array` → `ArrayBuffer` → `new Midi()`, following the same parsing and scheduling pipeline as uploaded files.

## License

See [LICENSE](LICENSE).