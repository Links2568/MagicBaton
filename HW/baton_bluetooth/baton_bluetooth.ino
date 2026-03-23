#include "BluetoothSerial.h" 
#include <Wire.h>

// --- Bluetooth config ---
BluetoothSerial SerialBT;
const char* device_name = "ESP32_Baton"; 

// --- IMU address ---
#define IMU_A_ADDR 0x68
#define IMU_B_ADDR 0x69
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_XOUT_H 0x3B

void initIMU(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00); 
  if (Wire.endTransmission() == 0) {
    Serial.printf("IMU at 0x%02X initialized.\n", addr);
  } else {
    Serial.printf("IMU at 0x%02X FAILED!\n", addr);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // Start BT
  if(!SerialBT.begin(device_name)){
    Serial.println("An error occurred initializing Bluetooth");
  } else {
    Serial.println("Bluetooth initialized, ready to pair!");
  }

  initIMU(IMU_A_ADDR);
  initIMU(IMU_B_ADDR);
}

void loop() {
  int16_t axA, ayA, azA, gxA, gyA, gzA;
  int16_t axB, ayB, azB, gxB, gyB, gzB;

  // IMU A 
  Wire.beginTransmission(IMU_A_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_A_ADDR, 14);
  axA = Wire.read()<<8|Wire.read(); ayA = Wire.read()<<8|Wire.read(); azA = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); // 跳过 Temp
  gxA = Wire.read()<<8|Wire.read(); gyA = Wire.read()<<8|Wire.read(); gzA = Wire.read()<<8|Wire.read();

  // IMU B 
  Wire.beginTransmission(IMU_B_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_B_ADDR, 14);
  axB = Wire.read()<<8|Wire.read(); ayB = Wire.read()<<8|Wire.read(); azB = Wire.read()<<8|Wire.read();
  Wire.read(); Wire.read(); // 跳过 Temp
  gxB = Wire.read()<<8|Wire.read(); gyB = Wire.read()<<8|Wire.read(); gzB = Wire.read()<<8|Wire.read();

  // String form
  char buf[150];
  snprintf(buf, sizeof(buf), "A,%d,%d,%d,%d,%d,%d|B,%d,%d,%d,%d,%d,%d", 
           axA, ayA, azA, gxA, gyA, gzA,
           axB, ayB, azB, gxB, gyB, gzB);

  //Send the data
  SerialBT.println(buf);

  delay(10); 
}