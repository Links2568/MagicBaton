# import socket

# UDP_IP = "0.0.0.0" 
# UDP_PORT = 4210

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# print(f"Listening on port {UDP_PORT}...")

# while True:
#     data, addr = sock.recvfrom(1024)
#     message = data.decode('utf-8')
    
#     
#     parts = message.split('|')
#     print(f"Recv: {parts}")

import socket
import json
import time

# --- Config ---
UDP_IP = "0.0.0.0" 
UDP_PORT = 4210
JSON_FILE = "imu_data_wifi.json"

# Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Python receiver started, monitoring on port: {UDP_PORT}")

def save_to_json(data_dict):
    # Saving data to the file
    try:
        with open(JSON_FILE, 'w') as f:
            json.dump(data_dict, f, indent=4)
    except Exception as e:
        print(f"Failed in writing the file: {e}")

while True:
    try:
        # 1. Getting raw data
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')
        
        # 2. Decoding data from esp32: A,ax,ay,az,gx,gy,gz|B,ax,ay,az,gx,gy,gz
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
                },
                "status": "online"
            }
            
            # 3. Write to json file
            save_to_json(output)
            
            
            print(f"Data has been updated: IMU_A_Acc_Z: {imu_a[3]} | IMU_B_Acc_Z: {imu_b[3]}")
            
        except (IndexError, ValueError):
            print("Wrong data form, jump this loop")

    except KeyboardInterrupt:
        print("\nThe program has been stopped")
        break