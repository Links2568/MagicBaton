"""
MagicBaton — Gesture Data Recorder (GUI)

GUI app for recording labeled IMU gesture data for ML training.
Connects via BLE, shows real-time waveform, records gesture reps.

Usage:
    python record.py
"""

import asyncio
import csv
import json
import time
import os
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# ── Config ──
BLE_DEVICE_NAME = "MagicBaton"
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GESTURES = [
    "idle",       # 0 - stationary
    "beat",       # 1 - downward strike
    "stab",       # 2 - forward thrust
    "spin",       # 3 - draw a circle
    "infinity",   # 4 - draw figure-8 (∞)
    "slash",      # 5 - diagonal cut
    "shake",      # 6 - rapid tremolo
    "flick",      # 7 - quick wrist snap
    "wing",       # 8 - draw W shape
    "slope",      # 9 - draw checkmark
]
CHANNELS = [
    "ax_a", "ay_a", "az_a", "gx_a", "gy_a", "gz_a",
    "ax_b", "ay_b", "az_b", "gx_b", "gy_b", "gz_b",
]
COLORS_ACCEL_A = {"ax_a": "#ff4466", "ay_a": "#44ff88", "az_a": "#4488ff"}
COLORS_ACCEL_B = {"ax_b": "#ff8844", "ay_b": "#88ff44", "az_b": "#44bbff"}
COLORS_GYRO_A = {"gx_a": "#ff6688", "gy_a": "#66ffaa", "gz_a": "#6699ff"}
COLORS_GYRO_B = {"gx_b": "#ffaa66", "gy_b": "#aaff66", "gz_b": "#66ddff"}
ALL_COLORS = {**COLORS_ACCEL_A, **COLORS_ACCEL_B, **COLORS_GYRO_A, **COLORS_GYRO_B}
WAVE_LEN = 200


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


def load_metadata():
    path = os.path.join(DATA_DIR, "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"sessions": []}


def save_metadata(meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def next_rep_number(meta, gesture):
    existing = [s["rep"] for s in meta["sessions"] if s["gesture"] == gesture]
    return max(existing, default=0) + 1


def save_session(rows, gesture, rep, meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{gesture}_{rep:03d}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "ax_a", "ay_a", "az_a", "gx_a", "gy_a", "gz_a",
            "ax_b", "ay_b", "az_b", "gx_b", "gy_b", "gz_b",
        ])
        writer.writerows(rows)
    duration = rows[-1][0] - rows[0][0] if len(rows) > 1 else 0
    entry = {
        "file": filename,
        "gesture": gesture,
        "rep": rep,
        "samples": len(rows),
        "duration_s": round(duration, 2),
        "hz": round(len(rows) / duration, 1) if duration > 0 else 0,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta["sessions"].append(entry)
    save_metadata(meta)
    return entry


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBaton Recorder")
        self.root.configure(bg="#111118")
        self.root.geometry("900x620")
        self.root.minsize(800, 550)

        self.meta = load_metadata()
        self.recording = False
        self.rec_buffer = []
        self.wave_data = {ch: [0] * WAVE_LEN for ch in CHANNELS}
        self.connected = False
        self.current_gesture = tk.StringVar(value=GESTURES[0])
        self.status_text = tk.StringVar(value="Disconnected")
        self.sample_count = 0

        self.build_ui()
        self.start_ble_thread()
        self.update_waveform()

    # ── UI ──
    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111118")
        style.configure("TLabel", background="#111118", foreground="#ccccdd",
                         font=("Menlo", 11))
        style.configure("Header.TLabel", font=("Menlo", 13, "bold"), foreground="#eeeef4")
        style.configure("Big.TLabel", font=("Menlo", 28, "bold"), foreground="#ff9944")
        style.configure("Status.TLabel", font=("Menlo", 10), foreground="#666680")
        style.configure("Rec.TButton", font=("Menlo", 13, "bold"), padding=10)
        style.configure("Gest.TRadiobutton", background="#111118", foreground="#ccccdd",
                         font=("Menlo", 11), indicatorsize=0)
        style.map("Gest.TRadiobutton",
                   background=[("selected", "#2a2a40")],
                   foreground=[("selected", "#44ff88")])

        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(top, text="MagicBaton Recorder", style="Header.TLabel").pack(side="left")
        self.status_label = ttk.Label(top, textvariable=self.status_text, style="Status.TLabel")
        self.status_label.pack(side="right")

        # Main area: left (controls) + right (waveform)
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=16, pady=8)

        # Left panel
        left = ttk.Frame(main, width=260)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Gesture selector
        ttk.Label(left, text="Gesture:", style="TLabel").pack(anchor="w", pady=(8, 4))
        for g in GESTURES:
            rb = ttk.Radiobutton(left, text=g, variable=self.current_gesture,
                                  value=g, style="Gest.TRadiobutton")
            rb.pack(anchor="w", padx=8, pady=2)

        # Record button
        self.rec_btn = tk.Button(
            left, text="Start Recording\n(Space)", font=("Menlo", 12, "bold"),
            bg="#1a1a28", fg="#ccccdd", activebackground="#2a2a40",
            activeforeground="#44ff88", relief="flat", bd=0, height=3,
            command=self.toggle_recording
        )
        self.rec_btn.pack(fill="x", pady=(16, 8), ipady=4)

        # Rec info
        self.rec_info = ttk.Label(left, text="", style="Status.TLabel")
        self.rec_info.pack(anchor="w", pady=(0, 8))

        # Stats
        ttk.Label(left, text="Dataset:", style="TLabel").pack(anchor="w", pady=(12, 4))
        self.stats_frame = ttk.Frame(left)
        self.stats_frame.pack(fill="x", padx=4)
        self.update_stats()

        # Delete last button
        self.del_btn = tk.Button(
            left, text="Undo Last", font=("Menlo", 10),
            bg="#1a1a28", fg="#664444", activebackground="#2a2a40",
            activeforeground="#ff4466", relief="flat", bd=0,
            command=self.delete_last
        )
        self.del_btn.pack(fill="x", pady=(8, 0))

        # Right panel: waveform canvas
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(right, bg="#0a0a10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Legend
        legend = ttk.Frame(right)
        legend.pack(fill="x", pady=(4, 0))
        for label, color in [("X", "#ff4466"), ("Y", "#44ff88"), ("Z", "#4488ff")]:
            ttk.Label(legend, text=f" {label} ", foreground=color,
                       font=("Menlo", 9, "bold"), background="#111118").pack(side="left")
        self.live_values = ttk.Label(legend, text="", style="Status.TLabel")
        self.live_values.pack(side="right")

        # Key bindings
        self.root.bind("<space>", lambda e: self.toggle_recording())
        self.root.bind("<KeyPress>", self.on_key)

    def on_key(self, event):
        c = event.char
        for i, g in enumerate(GESTURES):
            if c == str(i):
                self.current_gesture.set(g)
                return

    # ── BLE ──
    def start_ble_thread(self):
        t = threading.Thread(target=self.ble_thread, daemon=True)
        t.start()

    def ble_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_run())

    async def ble_run(self):
        from bleak import BleakClient, BleakScanner

        while True:
            self.status_text.set("Scanning...")
            try:
                device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME, timeout=8.0)
                if not device:
                    self.status_text.set("Not found, retrying...")
                    await asyncio.sleep(3)
                    continue

                self.status_text.set(f"Connecting...")
                async with BleakClient(device) as client:
                    self.connected = True
                    self.status_text.set("Connected")

                    def on_notify(sender, data):
                        line = data.decode("utf-8", errors="ignore").strip()
                        vals = parse_line(line)
                        if vals is None or len(vals) != 12:
                            return
                        for i, ch in enumerate(CHANNELS):
                            self.wave_data[ch].append(vals[i])
                            self.wave_data[ch].pop(0)

                        if self.recording:
                            self.rec_buffer.append([time.time()] + vals)
                            self.sample_count = len(self.rec_buffer)

                    await client.start_notify(BLE_TX_UUID, on_notify)
                    while client.is_connected:
                        await asyncio.sleep(0.5)

                self.connected = False
                self.status_text.set("Disconnected")
            except Exception as e:
                self.connected = False
                self.status_text.set(f"Error: {e}")
                await asyncio.sleep(3)

    # ── Recording ──
    def toggle_recording(self):
        if not self.connected:
            return
        if not self.recording:
            self.recording = True
            self.rec_buffer = []
            self.sample_count = 0
            self.rec_btn.configure(text="Stop Recording\n(Space)", bg="#331a1a", fg="#ff4466")
            gesture = self.current_gesture.get()
            self.rec_info.configure(text=f"Recording [{gesture}]...", foreground="#ff4466")
        else:
            self.recording = False
            self.rec_btn.configure(text="Start Recording\n(Space)", bg="#1a1a28", fg="#ccccdd")
            if len(self.rec_buffer) < 10:
                self.rec_info.configure(text=f"Too short, discarded.", foreground="#666680")
                return
            gesture = self.current_gesture.get()
            rep = next_rep_number(self.meta, gesture)
            entry = save_session(self.rec_buffer, gesture, rep, self.meta)
            self.rec_info.configure(
                text=f"Saved {entry['file']} ({entry['samples']}s, {entry['duration_s']}s)",
                foreground="#44ff88"
            )
            self.update_stats()

    def delete_last(self):
        if not self.meta["sessions"]:
            return
        last = self.meta["sessions"].pop()
        filepath = os.path.join(DATA_DIR, last["file"])
        if os.path.exists(filepath):
            os.remove(filepath)
        save_metadata(self.meta)
        self.rec_info.configure(text=f"Deleted {last['file']}", foreground="#ff8844")
        self.update_stats()

    # ── Stats ──
    def update_stats(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        counts = {}
        for s in self.meta["sessions"]:
            g = s["gesture"]
            counts[g] = counts.get(g, 0) + 1
        if not counts:
            ttk.Label(self.stats_frame, text="  (no data)", style="Status.TLabel").pack(anchor="w")
            return
        for g in GESTURES:
            c = counts.get(g, 0)
            if c == 0:
                continue
            row = ttk.Frame(self.stats_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{g}", width=10, style="TLabel").pack(side="left")
            # Bar
            bar_canvas = tk.Canvas(row, width=80, height=12, bg="#1a1a28", highlightthickness=0)
            bar_canvas.pack(side="left", padx=(4, 4))
            bar_w = min(c * 3, 78)
            color = "#44ff88" if c >= 20 else "#ff9944" if c >= 10 else "#ff4466"
            bar_canvas.create_rectangle(1, 1, bar_w, 11, fill=color, outline="")
            ttk.Label(row, text=f"{c}", style="Status.TLabel", width=4).pack(side="left")

    # ── Waveform ──
    def update_waveform(self):
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            self.root.after(50, self.update_waveform)
            return

        c.delete("all")

        # 4 stacked panels: Accel A, Accel B, Gyro A, Gyro B
        panels = [
            ("Accel A", COLORS_ACCEL_A, 32768),
            ("Accel B", COLORS_ACCEL_B, 32768),
            ("Gyro A", COLORS_GYRO_A, 32768),
            ("Gyro B", COLORS_GYRO_B, 32768),
        ]
        panel_h = h / len(panels)

        for pi, (label, colors, val_range) in enumerate(panels):
            y_off = pi * panel_h
            mid_y = y_off + panel_h / 2

            # Divider
            if pi > 0:
                c.create_line(0, y_off, w, y_off, fill="#252535", width=1)
            # Center line
            c.create_line(0, mid_y, w, mid_y, fill="#1a1a28", width=1)
            # Label
            c.create_text(6, y_off + 4, text=label, fill="#444460",
                          font=("Menlo", 9), anchor="nw")

            # Waveforms (clamped to panel bounds)
            for ch, color in colors.items():
                data = self.wave_data[ch]
                points = []
                for i, v in enumerate(data):
                    x = i / WAVE_LEN * w
                    y = mid_y - (v / val_range) * (panel_h * 0.45)
                    y = max(y_off, min(y_off + panel_h, y))
                    points.append(x)
                    points.append(y)
                if len(points) >= 4:
                    c.create_line(points, fill=color, width=1.2, smooth=True)

        # Recording indicator
        if self.recording:
            c.create_rectangle(0, 0, w, h, outline="#ff4466", width=3)
            c.create_text(w - 10, 14, text=f"REC  {self.sample_count}",
                          fill="#ff4466", font=("Menlo", 11, "bold"), anchor="ne")

        # Live values
        ax = self.wave_data["ax_a"][-1]
        ay = self.wave_data["ay_a"][-1]
        az = self.wave_data["az_a"][-1]
        self.live_values.configure(text=f"A:[{ax:>6},{ay:>6},{az:>6}]")

        self.root.after(33, self.update_waveform)  # ~30fps


def main():
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
