# MagicBaton Dataset

Gesture recordings from 4 subjects holding the MagicBaton (ESP32 + 2× MPU6050).
Each file is one gesture performance, sampled at ~50 Hz from both IMUs.

## Layout

```
dataset/
├── zuchen/   # 443 recordings
├── yoyo/     # 210 recordings
├── tiffany/  # 141 recordings
└── xiaolan/  # 140 recordings
```

Each subject directory contains per-gesture CSVs (`beat_001.csv`, `beat_002.csv`, …) and a `metadata.json` index.

## CSV schema

Columns (13):

| column                      | description                           |
|-----------------------------|---------------------------------------|
| `timestamp`                 | Unix time (seconds, float)            |
| `ax_a, ay_a, az_a`          | IMU A (addr `0x68`) accelerometer raw |
| `gx_a, gy_a, gz_a`          | IMU A gyroscope raw                   |
| `ax_b, ay_b, az_b`          | IMU B (addr `0x69`) accelerometer raw |
| `gx_b, gy_b, gz_b`          | IMU B gyroscope raw                   |

Accelerometer range is ±2 g (16384 raw counts per g), gyroscope range is ±250 °/s (32768 raw counts per 250 °/s) — MPU6050 defaults.

## Gestures

8 canonical classes used in the training scripts:

| label   | description                       |
|---------|-----------------------------------|
| `idle`  | hold still                        |
| `beat`  | conductor's downbeat              |
| `stab`  | forward thrust along the long axis|
| `spin`  | one full circle with the tip      |
| `slash` | diagonal cut                      |
| `shake` | 1–2 s side-to-side tremolo        |
| `flick` | single wrist snap                 |
| `wing`  | W-shape traced in the air         |

See the main [README](../README.md) for gesture illustrations and recording guidelines.

Subject `zuchen` is the reference set (most reps, all 8 classes) and is what the single-subject scripts (`train_linknet.py`, `train_beat_svm.py`) use. The cross-subject scripts (`train_linknet_cross.py`, `train_linknet_lopo_compare.py`, `train_linknet_rigid.py`) load all four.

## Recording your own

Run `python realtime/record.py` and follow the on-screen prompts. New recordings land in `realtime/data/` (gitignored) and can be dropped into a new `dataset/<subject>/` folder to extend the set.
