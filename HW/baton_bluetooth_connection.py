import serial
import json
import time

# --- Config ---
# Check Bluetooth port on device manager 
SERIAL_PORT = "COM8" 
BAUD_RATE = 115200
JSON_FILE = "imu_data_BLE.json"

def save_to_json(data_dict):
    try:
        with open(JSON_FILE, 'w') as f:
            json.dump(data_dict, f, indent=4)
    except Exception as e:
        print(f"Failed in writing the file: {e}")

try:
    # Initialize Port
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Bluetooth serial on {SERIAL_PORT}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

while True:
    try:
        if ser.in_waiting > 0:
            # 1. read a line
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            # 2. Parse the line A,ax,ay,az,gx,gy,gz|B,ax,ay,az,gx,gy,gz
            try:
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                    
                imu_a = parts[0].split(',')
                imu_b = parts[1].split(',')
                
                output = {
                    "timestamp": time.time(),
                    "IMU_A": {
                        "accel": [int(imu_a[1]), int(imu_a[2]), int(imu_a[3])],
                        "gyro":  [int(imu_a[4]), int(imu_a[5]), int(imu_a[6])]
                    },
                    "IMU_B": {
                        "accel": [int(imu_b[1]), int(imu_b[2]), int(imu_b[3])],
                        "gyro":  [int(imu_b[4]), int(imu_b[5]), int(imu_b[6])]
                    },
                    "status": "online"
                }
                
                # 3. write to the json file
                save_to_json(output)
                print(f"Bluetooth Data Updated: IMU_A_Acc_Z: {imu_a[3]} | IMU_B_Acc_Z: {imu_b[3]}")
                
            except (IndexError, ValueError):
                print("Data format error, skipping...")

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()
        break
    except Exception as e:
        print(f"Loop error: {e}")
        break