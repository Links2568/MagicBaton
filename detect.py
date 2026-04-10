"""
MagicBaton — Real-Time Gesture Detector (GUI)

Connects via BLE, segments gestures using energy-gated state machine,
classifies with BatonNet, and displays results in real-time.

Usage:
    python detect.py
"""

import asyncio
import collections
import os
import threading
import time
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ──
BLE_DEVICE_NAME = "MagicBaton"
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "batonnet.pt")

GESTURES = [
    "idle", "beat", "stab", "spin",
    "slash", "shake", "flick", "wing",
]
NUM_CLASSES = len(GESTURES)

CHANNELS = [
    "ax_a", "ay_a", "az_a", "gx_a", "gy_a", "gz_a",
    "ax_b", "ay_b", "az_b", "gx_b", "gy_b", "gz_b",
]

# Sliding window parameters
WINDOW_SIZE = 64            # samples per classification window
WINDOW_STRIDE = 16          # classify every N new samples
CONFIDENCE_THRESHOLD = 0.35
COOLDOWN_SECONDS = 0.2

# Model constants (must match train.py)
SEQ_LEN = 128
IN_CHANNELS = 6
NORM_SCALE = 32768.0

# Display
WAVE_LEN = 200


# ── BLE Parsing ──

def parse_line(line):
    """Parse full 12-axis data: accel+gyro from both IMUs."""
    try:
        parts = line.split("|")
        if len(parts) < 2:
            return None
        a = parts[0].split(",")
        b = parts[1].split(",")
        return [
            int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]),
            int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5]), int(b[6]),
        ]
    except (IndexError, ValueError):
        return None


# ── Model ──

class BatonNet(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 24, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(24, 48, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)  # GAP
        x = self.dropout(x)
        return self.fc(x)


def load_model(path=MODEL_PATH):
    """Load BatonNet weights and return model in eval mode."""
    model = BatonNet()
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Model loaded from {path}")
    else:
        print(f"WARNING: Model weights not found at {path}")
    model.eval()
    return model


# ── Preprocessing ──

def preprocess_segment(raw_samples):
    """
    raw_samples: list of 12-element lists (ax_a..gz_b).
    Returns tensor (1, 6, 128) ready for BatonNet.
    """
    arr = np.array(raw_samples, dtype=np.float32)  # (T, 12)
    imu_a = arr[:, :6]
    imu_b = arr[:, 6:]
    diff = (imu_a - imu_b) / 2.0
    diff /= NORM_SCALE  # normalize to ~[-1, 1]

    # Pad or truncate to SEQ_LEN
    t, c = diff.shape
    if t >= SEQ_LEN:
        diff = diff[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - t, c), dtype=np.float32)
        diff = np.concatenate([diff, pad], axis=0)

    # (1, C, T) for Conv1d
    tensor = torch.from_numpy(diff).float().permute(1, 0).unsqueeze(0)
    return tensor


@torch.no_grad()
def run_inference(model, tensor):
    """Returns (class_name, confidence, all_probs)."""
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    idx = int(probs.argmax())
    return GESTURES[idx], float(probs[idx]), probs


# ── Gesture Detector (State Machine) ──

class GestureDetector:
    """
    Sliding-window gesture detector.
    Continuously buffers samples and classifies every WINDOW_STRIDE frames.
    No energy gating — the model decides idle vs gesture.
    """

    def __init__(self, model):
        self.model = model
        self.state = "IDLE"
        self.buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.samples_since_last = 0
        self.cooldown_start = 0.0
        self.last_energy = 0.0       # kept for display compatibility
        self.result = None

    def feed(self, raw_12ch):
        """
        Feed one 12-channel sample. Returns prediction tuple or None.
        """
        self.buffer.append(raw_12ch)
        self.samples_since_last += 1

        # Cooldown: skip classification
        if self.state == "COOLDOWN":
            if time.time() - self.cooldown_start >= COOLDOWN_SECONDS:
                self.state = "IDLE"
            return None

        # Only classify every WINDOW_STRIDE samples and when buffer is full
        if self.samples_since_last < WINDOW_STRIDE or len(self.buffer) < WINDOW_SIZE:
            return None

        self.samples_since_last = 0
        self.state = "ACTIVE"

        seg = list(self.buffer)
        tensor = preprocess_segment(seg)
        class_name, confidence, probs = run_inference(self.model, tensor)

        if class_name == "idle" or confidence < CONFIDENCE_THRESHOLD:
            self.state = "IDLE"
            return None

        self.result = (class_name, confidence, probs, len(seg))
        self.state = "COOLDOWN"
        self.cooldown_start = time.time()
        return self.result


# ── GUI Application ──

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBaton Live Detector")
        self.root.configure(bg="#111118")
        self.root.geometry("960x700")
        self.root.minsize(860, 600)

        # Model
        self.model = load_model()
        self.detector = GestureDetector(self.model)

        # BLE data queue (thread-safe)
        self.ble_queue = collections.deque(maxlen=500)
        self.connected = False
        self.status_text = tk.StringVar(value="Disconnected")

        # Diff waveform display buffers (6 channels)
        self.diff_labels = ["ax", "ay", "az", "gx", "gy", "gz"]
        self.diff_data = {ch: [0.0] * WAVE_LEN for ch in self.diff_labels}
        self.prev_raw = None  # for computing diff display

        # History
        self.history = []  # list of (time_str, class_name, confidence, n_samples)

        self.build_ui()
        self.start_ble_thread()
        self.update_tick()

    # ── UI ──
    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111118")
        style.configure("TLabel", background="#111118", foreground="#ccccdd",
                         font=("Menlo", 11))
        style.configure("Header.TLabel", font=("Menlo", 13, "bold"), foreground="#eeeef4")
        style.configure("Big.TLabel", font=("Menlo", 28, "bold"), foreground="#ff9944")
        style.configure("Conf.TLabel", font=("Menlo", 14), foreground="#888899")
        style.configure("Status.TLabel", font=("Menlo", 10), foreground="#666680")
        style.configure("State.TLabel", font=("Menlo", 11, "bold"), foreground="#44ff88")
        style.configure("History.TLabel", font=("Menlo", 10), foreground="#9999aa")

        # ── Top bar ──
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(top, text="MagicBaton Live Detector", style="Header.TLabel").pack(side="left")
        self.status_label = ttk.Label(top, textvariable=self.status_text, style="Status.TLabel")
        self.status_label.pack(side="right")

        # ── Waveform canvas ──
        wave_frame = ttk.Frame(self.root)
        wave_frame.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        self.canvas = tk.Canvas(wave_frame, bg="#0a0a10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.detect_label_id = None

        # ── Middle row: energy + state | prediction + confidence bars ──
        mid = ttk.Frame(self.root)
        mid.pack(fill="x", padx=16, pady=(4, 4))

        # Left: energy bar + state
        left_mid = ttk.Frame(mid, width=240)
        left_mid.pack(side="left", fill="y", padx=(0, 16))
        left_mid.pack_propagate(False)

        energy_row = ttk.Frame(left_mid)
        energy_row.pack(fill="x", pady=(4, 2))
        ttk.Label(energy_row, text="Energy:", style="TLabel").pack(side="left")
        self.energy_canvas = tk.Canvas(energy_row, width=140, height=16,
                                        bg="#1a1a28", highlightthickness=0)
        self.energy_canvas.pack(side="left", padx=(8, 0))

        state_row = ttk.Frame(left_mid)
        state_row.pack(fill="x", pady=(2, 4))
        ttk.Label(state_row, text="State:", style="TLabel").pack(side="left")
        self.state_label = ttk.Label(state_row, text="IDLE", style="State.TLabel")
        self.state_label.pack(side="left", padx=(8, 0))

        # Right: prediction + confidence bars
        right_mid = ttk.Frame(mid)
        right_mid.pack(side="left", fill="both", expand=True)

        pred_row = ttk.Frame(right_mid)
        pred_row.pack(fill="x", pady=(4, 2))
        self.pred_label = ttk.Label(pred_row, text="---", style="Big.TLabel")
        self.pred_label.pack(side="left")
        self.conf_label = ttk.Label(pred_row, text="", style="Conf.TLabel")
        self.conf_label.pack(side="left", padx=(12, 0))

        # Confidence bars canvas
        self.bars_canvas = tk.Canvas(right_mid, height=110, bg="#111118",
                                      highlightthickness=0)
        self.bars_canvas.pack(fill="x", pady=(2, 4))

        # ── Bottom: history ──
        bot = ttk.Frame(self.root)
        bot.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Label(bot, text="History:", style="TLabel").pack(anchor="w")
        self.history_frame = ttk.Frame(bot)
        self.history_frame.pack(fill="x")

    # ── BLE ──
    def start_ble_thread(self):
        t = threading.Thread(target=self._ble_thread, daemon=True)
        t.start()

    def _ble_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_run())

    async def _ble_run(self):
        from bleak import BleakClient, BleakScanner

        while True:
            self.status_text.set("Scanning...")
            try:
                device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME, timeout=8.0)
                if not device:
                    self.status_text.set("Not found, retrying...")
                    await asyncio.sleep(3)
                    continue

                self.status_text.set("Connecting...")
                async with BleakClient(device) as client:
                    self.connected = True
                    self.status_text.set("Connected")

                    def on_notify(sender, data):
                        line = data.decode("utf-8", errors="ignore").strip()
                        vals = parse_line(line)
                        if vals is not None and len(vals) == 12:
                            self.ble_queue.append(vals)

                    await client.start_notify(BLE_TX_UUID, on_notify)
                    while client.is_connected:
                        await asyncio.sleep(0.5)

                self.connected = False
                self.status_text.set("Disconnected")
            except Exception as e:
                self.connected = False
                self.status_text.set(f"Error: {e}")
                await asyncio.sleep(3)

    # ── Main Update Loop (~30fps) ──
    def update_tick(self):
        # 1. Drain BLE queue
        new_samples = []
        while self.ble_queue:
            try:
                new_samples.append(self.ble_queue.popleft())
            except IndexError:
                break

        # 2. Process each sample
        prediction = None
        for raw in new_samples:
            # Compute diff for display
            arr = np.array(raw, dtype=np.float32)
            imu_a = arr[:6]
            imu_b = arr[6:]
            diff = (imu_a - imu_b) / 2.0 / NORM_SCALE

            for i, ch in enumerate(self.diff_labels):
                self.diff_data[ch].append(float(diff[i]))
                self.diff_data[ch].pop(0)

            # Feed detector
            result = self.detector.feed(raw)
            if result is not None:
                prediction = result

        # 3. Update GUI if prediction
        if prediction is not None:
            class_name, confidence, probs, n_samples = prediction
            self.pred_label.configure(text=class_name.upper())
            self.conf_label.configure(text=f"({confidence * 100:.1f}%)")
            self._draw_confidence_bars(probs)
            # Add to history
            time_str = datetime.now().strftime("%H:%M:%S")
            self.history.insert(0, (time_str, class_name, confidence, n_samples))
            if len(self.history) > 20:
                self.history.pop()
            self._update_history()

        # 4. Redraw waveform + energy
        self._draw_waveform()
        self._draw_energy()
        self._update_state()

        self.root.after(33, self.update_tick)

    # ── Waveform Drawing ──
    def _draw_waveform(self):
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        c.delete("all")

        # 2 panels: top = accel diff (ax, ay, az), bottom = gyro diff (gx, gy, gz)
        accel_colors = {"ax": "#ff4466", "ay": "#44ff88", "az": "#4488ff"}
        gyro_colors = {"gx": "#ff6688", "gy": "#66ffaa", "gz": "#6699ff"}
        panels = [
            ("Accel Diff (A-B)", accel_colors, 1.0),
            ("Gyro Diff (A-B)", gyro_colors, 1.0),
        ]
        panel_h = h / len(panels)

        for pi, (label, colors, val_range) in enumerate(panels):
            y_off = pi * panel_h
            mid_y = y_off + panel_h / 2

            if pi > 0:
                c.create_line(0, y_off, w, y_off, fill="#252535", width=1)
            c.create_line(0, mid_y, w, mid_y, fill="#1a1a28", width=1)
            c.create_text(6, y_off + 4, text=label, fill="#444460",
                          font=("Menlo", 9), anchor="nw")

            for ch, color in colors.items():
                data = self.diff_data[ch]
                points = []
                for i, v in enumerate(data):
                    x = i / WAVE_LEN * w
                    y = mid_y - (v / val_range) * (panel_h * 0.45)
                    y = max(y_off, min(y_off + panel_h, y))
                    points.append(x)
                    points.append(y)
                if len(points) >= 4:
                    c.create_line(points, fill=color, width=1.2, smooth=True)

        # Detection indicator
        if self.detector.state == "ACTIVE":
            c.create_rectangle(0, 0, w, h, outline="#ff9944", width=3)
            c.create_text(w - 10, 14, text="DETECTING",
                          fill="#ff9944", font=("Menlo", 11, "bold"), anchor="ne")

    # ── Energy Bar (disabled — kept as placeholder) ──
    def _draw_energy(self):
        c = self.energy_canvas
        c.delete("all")

    # ── State Label ──
    def _update_state(self):
        state = self.detector.state
        colors = {"IDLE": "#666680", "ACTIVE": "#ff9944", "COOLDOWN": "#4488ff"}
        self.state_label.configure(text=state, foreground=colors.get(state, "#ccccdd"))

    # ── Confidence Bars ──
    def _draw_confidence_bars(self, probs):
        c = self.bars_canvas
        c.delete("all")
        cw = c.winfo_width()
        if cw < 10:
            cw = 600
        bar_h = 10
        spacing = 11
        max_bar_w = cw - 100

        for i, (name, prob) in enumerate(zip(GESTURES, probs)):
            y = i * spacing
            # Label
            c.create_text(50, y + bar_h // 2, text=f"{name:>8s}",
                          fill="#888899", font=("Menlo", 9), anchor="e")
            # Background
            c.create_rectangle(55, y, 55 + max_bar_w, y + bar_h,
                               fill="#1a1a28", outline="")
            # Bar
            bw = int(prob * max_bar_w)
            color = "#ff9944" if prob == probs.max() else "#333366"
            if bw > 0:
                c.create_rectangle(55, y, 55 + bw, y + bar_h,
                                   fill=color, outline="")
            # Percentage
            c.create_text(55 + max_bar_w + 5, y + bar_h // 2,
                          text=f"{prob * 100:4.1f}%", fill="#666680",
                          font=("Menlo", 9), anchor="w")

    # ── History ──
    def _update_history(self):
        for w in self.history_frame.winfo_children():
            w.destroy()

        for i, (t, name, conf, n_samp) in enumerate(self.history[:8]):
            text = f"  {t}  {name:<10s} {conf * 100:5.1f}%   {n_samp:>3d} samples"
            fg = "#ccccdd" if i == 0 else "#666680"
            lbl = ttk.Label(self.history_frame, text=text, style="History.TLabel")
            lbl.configure(foreground=fg)
            lbl.pack(anchor="w")


# ── Main ──

def main():
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
