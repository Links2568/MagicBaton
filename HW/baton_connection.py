import socket

# 配置与 ESP32 一致
UDP_IP = "0.0.0.0" # 监听所有网卡
UDP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024)
    message = data.decode('utf-8')
    
    # 简单的解析演示
    parts = message.split('|')
    print(f"Recv: {parts}")