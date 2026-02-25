#include <Wire.h>
#include "driver/i2s.h"
#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_bt.h"

#define SAMPLE_RATE 8000
#define CHUNK_SIZE 8 
#define ADXL345_ADDR 0x53

struct __attribute__((packed)) SyncPacket {
    uint16_t header = 0xBBAA;
    int16_t ax, ay, az;
    int16_t audio[CHUNK_SIZE];
};

SyncPacket packet;
QueueHandle_t dataQueue;

// Hàm đọc Raw dữ liệu từ ADXL345 cực nhanh
void readADXLRaw(int16_t &x, int16_t &y, int16_t &z) {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x32); 
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 6);
    if (Wire.available() == 6) {
        x = (int16_t)(Wire.read() | (Wire.read() << 8));
        y = (int16_t)(Wire.read() | (Wire.read() << 8));
        z = (int16_t)(Wire.read() | (Wire.read() << 8));
    }
}

// Bộ lọc High-pass (Giữ lại từ nhánh âm thanh)
int32_t prev_x_f = 0, prev_y_f = 0;
int16_t highPassFilter(int16_t x) {
    int32_t current_x = (int32_t)x;
    int32_t current_y = (243 * (prev_y_f + current_x - prev_x_f)) >> 8;
    prev_x_f = current_x; prev_y_f = current_y;
    return (int16_t)current_y;
}

void dspTask(void *pvParameters) {
    size_t bytes_read;
    int32_t raw_i2s[CHUNK_SIZE];
    
    // Tạo 3 biến tạm để tránh lỗi "cannot bind packed field"
    int16_t tempX, tempY, tempZ;

    while (1) {
        // Đọc I2S
        if (i2s_read(I2S_NUM_0, &raw_i2s, sizeof(raw_i2s), &bytes_read, portMAX_DELAY) == ESP_OK) {
            
            // Đọc vào biến tạm trước
            readADXLRaw(tempX, tempY, tempZ);
            
            // Sau đó mới gán vào packet
            packet.ax = tempX;
            packet.ay = tempY;
            packet.az = tempZ;
            
            for (int i = 0; i < CHUNK_SIZE; i++) {
                int16_t s = (int16_t)(raw_i2s[i] >> 14);
                packet.audio[i] = highPassFilter(s);
            }
            
            xQueueSend(dataQueue, &packet, portMAX_DELAY);
        }
    }
}

void setup() {
    Serial.begin(921600);
    
    // 1. Tốc độ I2C lên 400kHz
    Wire.begin();
    Wire.setClock(400000); 

    // Khởi tạo ADXL345 (Manual)
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2D); Wire.write(8); // Power ON
    Wire.endTransmission();
    
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x31); 
    Wire.write(0x00); // Chuyển về dải +/- 2G (Nhạy nhất có thể)
    Wire.endTransmission();

    // 2. Chế độ RF Quiet
    btStop(); esp_bt_controller_disable();
    WiFi.begin("Coin", "01152718");
    esp_wifi_set_ps(WIFI_PS_MAX_MODEM);

    // 3. I2S với Buffer lớn hơn
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
        .dma_buf_count = 16, // Tăng lên để chống Drop mẫu
        .dma_buf_len = 128,
        .use_apll = false
    };
    i2s_pin_config_t pin_config = {.bck_io_num = 14, .ws_io_num = 15, .data_in_num = 32};
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);

    dataQueue = xQueueCreate(32, sizeof(SyncPacket));
    xTaskCreatePinnedToCore(dspTask, "DSP", 4096, NULL, 10, NULL, 0);
}

void loop() {
    SyncPacket out;
    if (xQueueReceive(dataQueue, &out, portMAX_DELAY)) {
        Serial.write((uint8_t*)&out, sizeof(SyncPacket));
    }
}