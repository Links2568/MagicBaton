"""
MagicBaton WebSocket Bridge
Reads IMU data from ESP32 via BLE and forwards to browser via WebSocket.
Falls back to mock data when no hardware is connected.
Runs SVM-based beat detection + CNN gesture classification on sliding windows.
"""

import asyncio
import collections
import json
import math
import os
import pickle
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import websockets
from bleak import BleakClient, BleakScanner

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Config ---
BLE_DEVICE_NAME = "MagicBaton"
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
WS_HOST = "0.0.0.0"
WS_PORT = 8765
NORM_SCALE = 32768.0

# SVM beat detection (matches train_beat_svm.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVM_MODEL_PATH = os.path.join(_ROOT, "models", "beat_svm.pkl")
SVM_WINDOW = 22          # ~440ms — short enough to isolate a single rapid beat
SVM_STRIDE = 1           # classify every new sample (~20ms) for fast response
SVM_TRIGGER_HIGH = 0.55  # rising-edge threshold: prob must exceed this
SVM_TRIGGER_LOW = 0.30   # must fall below this before another beat can fire
SVM_COOLDOWN = 0.12      # minimum seconds between detections (safety only)

# CNN gesture detection (matches train_linknet_cross.py)
CNN_MODEL_PATH = os.path.join(_ROOT, "models", "linknet_cross.pt")
CNN_GESTURES = ["beat", "stab", "spin", "slash", "shake", "flick", "wing"]
CNN_NUM_CLASSES = len(CNN_GESTURES)
CNN_SEQ_LEN = 128
CNN_JERK_THRESHOLD = 0.15   # jerk magnitude to trigger recording
CNN_RECORD_DURATION = 1.0   # seconds to record after jerk trigger
CNN_CONFIDENCE_THRESHOLD = 0.4
CNN_COOLDOWN = 0.5          # seconds before next detection allowed

connected_clients = set()
latest_data = None
use_mock = True

# SVM state
svm_model = None
svm_scaler = None
svm_buffer = collections.deque(maxlen=SVM_WINDOW)
svm_samples_since_last = 0
svm_last_beat_time = 0.0
svm_armed = True         # True → next rising edge will fire; False → must dip below LOW first
svm_last_event = None

# CNN state
cnn_model = None
cnn_state = "IDLE"          # IDLE -> RECORDING -> COOLDOWN
cnn_prev_diff = None        # previous diff sample for jerk calculation
cnn_record_buf = []         # samples collected after trigger
cnn_trigger_time = 0.0      # when jerk triggered
cnn_last_gesture_time = 0.0


def load_svm():
    global svm_model, svm_scaler
    try:
        with open(SVM_MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        svm_scaler = obj["scaler"]
        svm_model = obj["model"]
        print(f"[SVM] Loaded {SVM_MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"[SVM] Failed to load: {e}", flush=True)


def extract_features(seq):
    """Match train_beat_svm.py exactly: (T, 6) diff-signal -> 82 features."""
    T, C = seq.shape
    feats = []
    for c in range(C):
        ch = seq[:, c]
        m, s = np.mean(ch), np.std(ch)
        feats.extend([
            m, s, np.min(ch), np.max(ch), np.max(ch) - np.min(ch),
            np.median(ch),
            np.mean(((ch - m) / (s + 1e-10)) ** 3),
            np.mean(((ch - m) / (s + 1e-10)) ** 4) - 3,
            np.sum(ch ** 2) / T,
            np.sum(np.diff(np.sign(ch)) != 0) / max(T - 1, 1),
            np.percentile(ch, 25), np.percentile(ch, 75),
            np.percentile(ch, 75) - np.percentile(ch, 25),
        ])
    accel_mag = np.sqrt(np.sum(seq[:, :3] ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(seq[:, 3:] ** 2, axis=1))
    feats.extend([np.mean(accel_mag), np.std(accel_mag),
                  np.mean(gyro_mag), np.std(gyro_mag)])
    return np.array(feats, dtype=np.float32)


def svm_feed(imu_a_accel, imu_a_gyro, imu_b_accel, imu_b_gyro):
    """
    Feed one 12-axis sample into the SVM sliding window.
    Uses hysteresis (rising-edge) detection so a sustained high prob only fires once:
      - fires when armed AND prob crosses SVM_TRIGGER_HIGH
      - re-arms when prob drops below SVM_TRIGGER_LOW
    Returns (beat_bool, prob).
    """
    global svm_samples_since_last, svm_last_beat_time, svm_armed

    if svm_model is None:
        return False, 0.0

    # Build diff signal (IMU_A - IMU_B) / 2 / NORM_SCALE → 6 channels
    a = np.array(imu_a_accel + imu_a_gyro, dtype=np.float32)
    b = np.array(imu_b_accel + imu_b_gyro, dtype=np.float32)
    diff = (a - b) / 2.0 / NORM_SCALE
    svm_buffer.append(diff)
    svm_samples_since_last += 1

    if len(svm_buffer) < SVM_WINDOW or svm_samples_since_last < SVM_STRIDE:
        return False, 0.0
    svm_samples_since_last = 0

    seq = np.stack(list(svm_buffer))
    feats = extract_features(seq).reshape(1, -1)
    feats_scaled = svm_scaler.transform(feats)
    prob = float(svm_model.predict_proba(feats_scaled)[0, 1])

    now = time.time()
    trigger = False
    if svm_armed and prob >= SVM_TRIGGER_HIGH and (now - svm_last_beat_time) >= SVM_COOLDOWN:
        trigger = True
        svm_last_beat_time = now
        svm_armed = False
    elif not svm_armed and prob < SVM_TRIGGER_LOW:
        svm_armed = True
    return trigger, prob


# --- CNN Gesture Detection ---

class LinkNetCNN(nn.Module):
    def __init__(self, in_ch=6, num_classes=CNN_NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = self.dropout(x)
        return self.fc(x)


def load_cnn():
    global cnn_model
    try:
        cnn_model = LinkNetCNN()
        state = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True)
        cnn_model.load_state_dict(state)
        cnn_model.eval()
        print(f"[CNN] Loaded {CNN_MODEL_PATH}", flush=True)
    except Exception as e:
        cnn_model = None
        print(f"[CNN] Failed to load: {e}", flush=True)


@torch.no_grad()
def cnn_feed(imu_a_accel, imu_a_gyro, imu_b_accel, imu_b_gyro):
    """
    Jerk-triggered gesture detection:
      IDLE      — monitor jerk; when jerk > threshold, switch to RECORDING
      RECORDING — collect samples for CNN_RECORD_DURATION seconds, then classify
      COOLDOWN  — wait CNN_COOLDOWN seconds before allowing next trigger
    Returns (gesture_name, confidence, jerk_mag).
    """
    global cnn_state, cnn_prev_diff, cnn_record_buf, cnn_trigger_time, cnn_last_gesture_time

    if cnn_model is None:
        return "idle", 0.0, 0.0

    # Compute diff signal
    a = np.array(imu_a_accel + imu_a_gyro, dtype=np.float32)
    b = np.array(imu_b_accel + imu_b_gyro, dtype=np.float32)
    diff = (a - b) / 2.0 / NORM_SCALE

    # Compute jerk (difference between consecutive diff samples)
    jerk_mag = 0.0
    if cnn_prev_diff is not None:
        jerk = diff - cnn_prev_diff
        jerk_mag = float(np.linalg.norm(jerk))
    cnn_prev_diff = diff.copy()

    now = time.time()

    if cnn_state == "IDLE":
        if jerk_mag >= CNN_JERK_THRESHOLD:
            cnn_state = "RECORDING"
            cnn_record_buf = [diff]
            cnn_trigger_time = now
            print(f"[CNN] jerk={jerk_mag:.3f} -> RECORDING", flush=True)
        return "idle", 0.0, jerk_mag

    elif cnn_state == "RECORDING":
        cnn_record_buf.append(diff)
        elapsed = now - cnn_trigger_time
        if elapsed < CNN_RECORD_DURATION:
            return "idle", 0.0, jerk_mag

        # Time's up — classify
        seq = np.stack(cnn_record_buf)  # (T, 6)

        # Pad or truncate to CNN_SEQ_LEN
        if seq.shape[0] < CNN_SEQ_LEN:
            pad = np.zeros((CNN_SEQ_LEN - seq.shape[0], 6), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:CNN_SEQ_LEN]

        tensor = torch.from_numpy(seq).float().permute(1, 0).unsqueeze(0)  # (1, 6, 128)
        logits = cnn_model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        idx = int(probs.argmax())
        confidence = float(probs[idx])

        cnn_state = "COOLDOWN"
        cnn_last_gesture_time = now
        cnn_record_buf = []

        if confidence >= CNN_CONFIDENCE_THRESHOLD:
            print(f"[CNN] {len(seq)} samples -> {CNN_GESTURES[idx]} ({confidence:.2f})", flush=True)
            return CNN_GESTURES[idx], confidence, jerk_mag
        else:
            print(f"[CNN] {len(seq)} samples -> below conf ({confidence:.2f})", flush=True)
            return "idle", confidence, jerk_mag

    elif cnn_state == "COOLDOWN":
        if (now - cnn_last_gesture_time) >= CNN_COOLDOWN:
            cnn_state = "IDLE"
        return "idle", 0.0, jerk_mag

    return "idle", 0.0, jerk_mag


def parse_imu_line(message):
    try:
        parts = message.split("|")
        if len(parts) < 2:
            return None
        a = parts[0].split(",")
        b = parts[1].split(",")
        return {
            "timestamp": time.time(),
            "IMU_A": {
                "accel": [int(a[1]), int(a[2]), int(a[3])],
                "gyro": [int(a[4]), int(a[5]), int(a[6])],
            },
            "IMU_B": {
                "accel": [int(b[1]), int(b[2]), int(b[3])],
                "gyro": [int(b[4]), int(b[5]), int(b[6])],
            },
        }
    except (IndexError, ValueError):
        return None


def generate_mock_data():
    t = time.time()
    bpm = 90
    freq = bpm / 60.0
    phase = (t * freq) % 1.0
    beat = max(0, math.sin(phase * math.pi * 2) ** 8) * 16000
    sway = math.sin(t * 1.2) * 4000
    na = math.sin(t * 37.3) * 200
    nb = math.sin(t * 41.7) * 200
    return {
        "timestamp": t,
        "IMU_A": {
            "accel": [int(sway+na), int(-beat+na), int(8192+na)],
            "gyro": [int(math.sin(t*freq*math.pi*2)*3000), int(math.cos(t*freq*math.pi*2)*2000), int(na*2)],
        },
        "IMU_B": {
            "accel": [int(-sway+nb), int(beat*0.8+nb), int(-8192+nb)],
            "gyro": [int(math.sin(t*freq*math.pi*2+0.5)*2500), int(math.cos(t*freq*math.pi*2+0.5)*1800), int(nb*2)],
        },
    }


async def ble_task():
    """Connect to MagicBaton via BLE, feed latest_data."""
    global latest_data, use_mock

    while True:
        print("[BLE] Scanning...", flush=True)
        try:
            device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME, timeout=8.0)
            if not device:
                print("[BLE] Not found, retrying...", flush=True)
                await asyncio.sleep(3)
                continue

            print(f"[BLE] Connecting to {device.name} ({device.address})...", flush=True)
            async with BleakClient(device) as client:
                use_mock = False
                print("[BLE] Connected", flush=True)

                def on_notify(sender, data):
                    global latest_data
                    line = data.decode("utf-8", errors="ignore").strip()
                    parsed = parse_imu_line(line)
                    if parsed:
                        latest_data = parsed

                await client.start_notify(BLE_TX_UUID, on_notify)
                while client.is_connected:
                    await asyncio.sleep(0.5)

            use_mock = True
            print("[BLE] Disconnected", flush=True)
        except Exception as e:
            use_mock = True
            print(f"[BLE] Error: {e}", flush=True)
            await asyncio.sleep(3)


async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Client connected ({len(connected_clients)} total)", flush=True)
    try:
        async for _ in websocket:
            pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(connected_clients)} total)", flush=True)


async def broadcast_loop():
    global latest_data
    while True:
        if use_mock:
            latest_data = generate_mock_data()

        if latest_data and connected_clients:
            # Feed SVM detector with this sample
            beat, prob = svm_feed(
                latest_data["IMU_A"]["accel"], latest_data["IMU_A"]["gyro"],
                latest_data["IMU_B"]["accel"], latest_data["IMU_B"]["gyro"],
            )
            # Feed CNN gesture detector
            gesture, gesture_conf, gesture_jerk = cnn_feed(
                latest_data["IMU_A"]["accel"], latest_data["IMU_A"]["gyro"],
                latest_data["IMU_B"]["accel"], latest_data["IMU_B"]["gyro"],
            )

            payload = dict(latest_data)
            payload["svm_beat"] = beat
            payload["svm_prob"] = prob
            payload["gesture"] = gesture
            payload["gesture_confidence"] = gesture_conf
            payload["gesture_jerk"] = gesture_jerk

            msg = json.dumps(payload)
            dead = set()
            for c in connected_clients:
                try:
                    await c.send(msg)
                except Exception:
                    dead.add(c)
            connected_clients.difference_update(dead)

        await asyncio.sleep(0.02)


async def main():
    load_svm()
    load_cnn()
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[WS] Server on ws://{WS_HOST}:{WS_PORT}", flush=True)
    print("[INFO] Scanning for MagicBaton via BLE... Mock data until connected.", flush=True)

    await asyncio.gather(
        ble_task(),
        broadcast_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
