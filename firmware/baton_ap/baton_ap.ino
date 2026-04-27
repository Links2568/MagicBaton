#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// --- SoftAP Config ---
const char* ssid     = "ESP32_Baton_Net"; 
const char* password = "12345678";      
const char* udpAddress = "192.168.4.2";    
const int udpPort = 4210;                   

// --- IMU addresses and registers ---
#define IMU_A_ADDR 0x68
#define IMU_B_ADDR 0x69
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_XOUT_H 0x3B

WiFiUDP udp;

void initIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00); //wake the sensors
  if (Wire.endTransmission() == 0) {
    Serial.printf("IMU at 0x%02X initialized.\n", addr); 
  } else {
    Serial.printf("IMU at 0x%02X FAILED!\n", addr); 
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL [cite: 5]
  Wire.setClock(400000);

  // Start SoftAP
  Serial.println("\nConfiguring access point...");
  WiFi.softAP(ssid, password); 

  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(myIP); // defaults to 192.168.4.1

  // Initialize the sensors
  initIMU(IMU_A_ADDR);
  initIMU(IMU_B_ADDR);
}

void loop() {
  int16_t axA, ayA, azA, gxA, gyA, gzA;
  int16_t axB, ayB, azB, gxB, gyB, gzB;

  // Read IMU A
  Wire.beginTransmission(IMU_A_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_A_ADDR, 14);
  axA = Wire.read()<<8|Wire.read(); ayA = Wire.read()<<8|Wire.read(); azA = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); 
  gxA = Wire.read()<<8|Wire.read(); gyA = Wire.read()<<8|Wire.read(); gzA = Wire.read()<<8|Wire.read();

  // Read IMU B
  Wire.beginTransmission(IMU_B_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_B_ADDR, 14);
  axB = Wire.read()<<8|Wire.read(); ayB = Wire.read()<<8|Wire.read(); azB = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); 
  gxB = Wire.read()<<8|Wire.read(); gyB = Wire.read()<<8|Wire.read(); gzB = Wire.read()<<8|Wire.read();

  // package the data
  char buf[150];
  snprintf(buf, sizeof(buf), "A,%d,%d,%d,%d,%d,%d|B,%d,%d,%d,%d,%d,%d", 
           axA, ayA, azA, gxA, gyA, gzA,
           axB, ayB, azB, gxB, gyB, gzB);

   
  udp.beginPacket(udpAddress, udpPort);
  udp.print(buf);
  udp.endPacket();

  delay(10); // 100Hz Updating frequency
}