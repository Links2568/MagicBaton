#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// --- 修改你的 WiFi 信息 ---
const char* ssid     = "Linksys08735";
const char* password = "8jdghdgrgr";
const char* udpAddress = "192.168.1.119"; // 电脑的局域网 IP
const int udpPort = 4210;

// --- IMU 地址与寄存器 ---
#define IMU_A_ADDR 0x68
#define IMU_B_ADDR 0x69
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_XOUT_H 0x3B

WiFiUDP udp;

void initIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00); // 唤醒
  if (Wire.endTransmission() == 0) {
    Serial.printf("IMU at 0x%02X initialized.\n", addr);
  } else {
    Serial.printf("IMU at 0x%02X FAILED!\n", addr);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // 使用快速模式 I2C

  // 连接 WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected. IP: " + WiFi.localIP().toString());

  // 初始化两个传感器
  initIMU(IMU_A_ADDR);
  initIMU(IMU_B_ADDR);
}

void loop() {
  int16_t axA, ayA, azA, gxA, gyA, gzA;
  int16_t axB, ayB, azB, gxB, gyB, gzB;

  // 读取 IMU A (14字节包含Temp，我们跳过Temp只取前6和后6)
  Wire.beginTransmission(IMU_A_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_A_ADDR, 14);
  axA = Wire.read()<<8|Wire.read(); ayA = Wire.read()<<8|Wire.read(); azA = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); // 跳过 Temp
  gxA = Wire.read()<<8|Wire.read(); gyA = Wire.read()<<8|Wire.read(); gzA = Wire.read()<<8|Wire.read();

  // 读取 IMU B
  Wire.beginTransmission(IMU_B_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_B_ADDR, 14);
  axB = Wire.read()<<8|Wire.read(); ayB = Wire.read()<<8|Wire.read(); azB = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); // 跳过 Temp
  gxB = Wire.read()<<8|Wire.read(); gyB = Wire.read()<<8|Wire.read(); gzB = Wire.read()<<8|Wire.read();

  // 封包格式: A,ax,ay,az,gx,gy,gz|B,ax,ay,az,gx,gy,gz
  char buf[150];
  snprintf(buf, sizeof(buf), "A,%d,%d,%d,%d,%d,%d|B,%d,%d,%d,%d,%d,%d", 
           axA, ayA, azA, gxA, gyA, gzA,
           axB, ayB, azB, gxB, gyB, gzB);

  // 发送 UDP
  udp.beginPacket(udpAddress, udpPort);
  udp.print(buf);
  udp.endPacket();

  // 串口监控（可选，会略微降低频率）
  // Serial.println(buf);

  delay(10); // 约 100Hz 的更新频率
}