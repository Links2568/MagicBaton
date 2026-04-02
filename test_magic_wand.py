"""
Quick test: download Magic Wand pretrained model and run inference on live MagicBaton data.

Model: TF v2.4.0 magic_wand (wing / ring / slope / unknown)
Input: [1, 128, 3] float32 (128 samples × 3-axis accelerometer)
"""

import asyncio
import re
import sys
import time
import numpy as np

BLE_DEVICE_NAME = "MagicBaton"
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

LABELS = ["wing", "ring", "slope", "unknown"]


def download_model():
    """Download Magic Wand model from TF repo, extract .tflite from C array."""
    import urllib.request
    url = "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.0/tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.cc"
    print("  Downloading Magic Wand model...", flush=True)
    response = urllib.request.urlopen(url)
    text = response.read().decode("utf-8")
    hex_values = re.findall(r"0x([0-9a-fA-F]{2})", text)
    model_bytes = bytes(int(h, 16) for h in hex_values)
    path = "magic_wand_model.tflite"
    with open(path, "wb") as f:
        f.write(model_bytes)
    print(f"  Saved: {path} ({len(model_bytes)} bytes)")
    return path


def load_model(path):
    """Load TFLite model and print input/output info."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"  Input:  {inp['shape']} {inp['dtype']}")
    print(f"  Output: {out['shape']} {out['dtype']}")
    return interpreter


def predict(interpreter, samples):
    """Run inference on [128, 3] accelerometer data."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Normalize: Magic Wand expects values roughly in [-1, 1] range
    # MPU6050 raw accel is ±16384 for ±2g
    data = np.array(samples, dtype=np.float32) / 16384.0

    # Pad or trim to 128 samples
    if len(data) < 128:
        pad = np.zeros((128 - len(data), 3), dtype=np.float32)
        data = np.concatenate([data, pad])
    else:
        data = data[:128]

    data = data.reshape(inp['shape']).astype(inp['dtype'])
    interpreter.set_tensor(inp['index'], data)
    interpreter.invoke()
    probs = interpreter.get_tensor(out['index'])[0]
    return probs


def parse_accel(line):
    """Extract IMU_A accel [ax, ay, az] from BLE data."""
    try:
        parts = line.split("|")
        a = parts[0].split(",")
        return [int(a[1]), int(a[2]), int(a[3])]
    except (IndexError, ValueError):
        return None


async def main():
    from bleak import BleakClient, BleakScanner

    # Download and load model
    model_path = download_model()
    interpreter = load_model(model_path)
    print()

    # Connect BLE
    print(f"  Scanning for '{BLE_DEVICE_NAME}'...", flush=True)
    device = await BleakScanner.find_device_by_name(BLE_DEVICE_NAME, timeout=10.0)
    if not device:
        print("  Not found!")
        return

    buffer = []
    lock = asyncio.Lock()

    def on_data(sender, data):
        line = data.decode("utf-8", errors="ignore").strip()
        vals = parse_accel(line)
        if vals:
            buffer.append(vals)

    print(f"  Connecting...", flush=True)
    async with BleakClient(device) as client:
        await client.start_notify(BLE_TX_UUID, on_data)
        print("  Connected! Performing gesture inference every 2.5 seconds.\n")
        print("  Do wing (W shape), ring (circle), or slope (check mark).")
        print("  Press Ctrl+C to quit.\n")

        try:
            while client.is_connected:
                await asyncio.sleep(2.5)

                if len(buffer) < 20:
                    sys.stdout.write(f"\r  Collecting... ({len(buffer)} samples)")
                    sys.stdout.flush()
                    continue

                # Take last 128 samples (or whatever we have)
                samples = list(buffer[-128:])
                buffer.clear()

                probs = predict(interpreter, samples)
                best = np.argmax(probs)
                conf = probs[best]

                bar = ""
                for i, (label, p) in enumerate(zip(LABELS, probs)):
                    marker = " <<" if i == best else ""
                    bar += f"  {label:>8}: {p:5.1%}{marker}\n"

                sys.stdout.write(f"\r{'=' * 40}\n")
                sys.stdout.write(f"  Prediction: {LABELS[best].upper()} ({conf:.0%})\n")
                sys.stdout.write(bar)
                sys.stdout.write(f"{'=' * 40}\n")
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n  Done.")


if __name__ == "__main__":
    asyncio.run(main())
