"""
MagicBaton WebSocket Bridge
Reads IMU data from ESP32 via BLE and forwards to browser via WebSocket.
Falls back to mock data when no hardware is connected.
"""

import asyncio
import json
import math
import time
import websockets
from bleak import BleakClient, BleakScanner

# --- Config ---
BLE_DEVICE_NAME = "MagicBaton"
BLE_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
WS_HOST = "0.0.0.0"
WS_PORT = 8765

connected_clients = set()
latest_data = None
use_mock = True


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
            msg = json.dumps(latest_data)
            dead = set()
            for c in connected_clients:
                try:
                    await c.send(msg)
                except Exception:
                    dead.add(c)
            connected_clients.difference_update(dead)

        await asyncio.sleep(0.02)


async def main():
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[WS] Server on ws://{WS_HOST}:{WS_PORT}", flush=True)
    print("[INFO] Scanning for MagicBaton via BLE... Mock data until connected.", flush=True)

    await asyncio.gather(
        ble_task(),
        broadcast_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
