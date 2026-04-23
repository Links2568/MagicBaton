import socket
import json
import time

# --- Config ---
UDP_IP = "0.0.0.0" 
UDP_PORT = 4210
JSON_FILE = "imu_data.json"

# Create Sockets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Python Receiver has been initialized...")

def save_to_json(data_dict):
    try:
        
        with open(JSON_FILE, 'w') as f:
            json.dump(data_dict, f, indent=4)
    except Exception as e:
        print(f"Failed to write the json file: {e}")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')
        
        # 数据解析: A,ax,ay,az,gx,gy,gz|B,ax,ay,az,gx,gy,gz
        try:
            parts = message.split('|')
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
                }
            }
            
            save_to_json(output)
            print(f"Updates: | A_AccZ: {imu_a[3]} | B_AccZ: {imu_b[3]}")
            
        except (IndexError, ValueError):
            print("Wrong form, skip this loop")

    except KeyboardInterrupt:
        print("\nThe Program has stopped")
        break