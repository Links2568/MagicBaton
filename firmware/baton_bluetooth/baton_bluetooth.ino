#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Wire.h>

// --- BLE UART Service UUIDs ---
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// --- IMU Config ---
#define IMU_A_ADDR 0x68
#define IMU_B_ADDR 0x69
#define REG_PWR_MGMT_1 0x6B
#define REG_ACCEL_XOUT_H 0x3B

BLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE client connected!");
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE client disconnected, restarting advertising...");
    pServer->getAdvertising()->start();
  }
};

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

  // Start BLE
  BLEDevice::init("MagicBaton");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID_TX,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pTxCharacteristic->addDescriptor(new BLE2902());

  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("BLE ready, advertising as 'MagicBaton'...");

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

  // Package and send via BLE notify
  char buf[150];
  snprintf(buf, sizeof(buf), "A,%d,%d,%d,%d,%d,%d|B,%d,%d,%d,%d,%d,%d",
           axA, ayA, azA, gxA, gyA, gzA,
           axB, ayB, azB, gxB, gyB, gzB);

  // Send via BLE (if connected) and Serial (always)
  if (deviceConnected) {
    pTxCharacteristic->setValue(buf);
    pTxCharacteristic->notify();
  }
  Serial.println(buf);

  delay(20); // ~50Hz
}
