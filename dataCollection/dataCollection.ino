// ============================================================
//  Data Collector - Xiao ESP32S3
//  INMP441 (I2S) + ADXL345 (I2C)
//  Protocol: Binary packet stream qua Serial
//
//  Packet format (26 bytes):
//    [0xAA][0xBB]  2 bytes header
//    [ax][ay][az]  6 bytes int16 (raw ADC / 256.0 = g)
//    [s0..s7]      16 bytes int16 audio (sau high-pass filter)
//
//  Pin mapping (Xiao ESP32S3):
//    INMP441: WS=6, BCLK=43, DIN=44
//    ADXL345: SDA=5, SCL=4
// ============================================================

#include <Wire.h>
#include <WiFi.h>
#include "driver/i2s.h"
#include "esp_bt.h"

// ============================================================
// PIN & HARDWARE CONFIG
// ============================================================

#define I2S_WS        6
#define I2S_BCLK      43
#define I2S_DIN       44
#define I2C_SDA       5
#define I2C_SCL       4
#define ADXL345_ADDR  0x53

#define SAMPLE_RATE   8000
#define CHUNK_SIZE    8      // Audio samples moi lan doc I2S

// ============================================================
// PACKET STRUCTURE
// Dung __attribute__((packed)) de khong co padding bytes
// ============================================================

struct __attribute__((packed)) SyncPacket {
    uint16_t header;          // 0xBBAA
    int16_t  ax, ay, az;      // ADXL345 raw / 256.0 = g
    int16_t  audio[CHUNK_SIZE]; // PCM 16-bit sau high-pass
};

// ============================================================
// GLOBAL
// ============================================================

static SyncPacket packet;
static QueueHandle_t dataQueue = nullptr;

// High-pass filter state
static int32_t hpf_prev_x = 0, hpf_prev_y = 0;

// ============================================================
// HIGH-PASS FILTER
// Loai bo DC offset tu microphone
// alpha = 243/256 ~ 0.949 (fc ~ 400Hz tai 8kHz)
// ============================================================

int16_t highPassFilter(int16_t x) {
    int32_t cx = (int32_t)x;
    int32_t cy = (243 * (hpf_prev_y + cx - hpf_prev_x)) >> 8;
    hpf_prev_x = cx;
    hpf_prev_y = cy;
    return (int16_t)cy;
}

// ============================================================
// ADXL345
// ============================================================

void readADXLRaw(int16_t &x, int16_t &y, int16_t &z) {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x32);  // DATA_X0 register
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 6);
    if (Wire.available() == 6) {
        x = (int16_t)(Wire.read() | (Wire.read() << 8));
        y = (int16_t)(Wire.read() | (Wire.read() << 8));
        z = (int16_t)(Wire.read() | (Wire.read() << 8));
    } else {
        x = y = z = 0;
    }
}

void setup_adxl345() {
    // Power on
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2D); Wire.write(0x08);
    Wire.endTransmission();

    // Range: +/-2G, non-full-resolution (10-bit)
    // 0x00 = +-2g, non-full-res -> 1 LSB = 3.9mg -> /256 ~ g
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x31); Wire.write(0x00);
    Wire.endTransmission();

    // Data rate: 200Hz (0x0B) - du de lay ~1000 samples/s voi overhead I2C
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2C); Wire.write(0x0B);
    Wire.endTransmission();

    // Kiem tra DEVID
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x00);
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 1);
    uint8_t devid = Wire.available() ? Wire.read() : 0;
    if (devid == 0xE5) {
        Serial.println("[INFO] ADXL345 OK (0xE5)");
    } else {
        Serial.printf("[ERROR] ADXL345 DEVID=0x%02X (expected 0xE5)\n", devid);
    }
}

// ============================================================
// I2S (INMP441)
// ============================================================

void setup_i2s() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_BCLK,
        .ws_io_num    = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DIN
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    Serial.println("[INFO] I2S (INMP441) OK");
}

// ============================================================
// DSP TASK - Core 0
// Doc I2S + ADXL dong bo, dong goi, gui vao queue
// ============================================================

void dsp_task(void* pvParameters) {
    int32_t raw_i2s[CHUNK_SIZE];
    size_t  bytes_read = 0;
    int16_t tempX, tempY, tempZ;

    // Flush DMA buffer de tranh data cu
    for (int i = 0; i < 16; i++)
        i2s_read(I2S_NUM_0, raw_i2s, sizeof(raw_i2s), &bytes_read, pdMS_TO_TICKS(100));

    Serial.println("[INFO] DSP task ready, streaming...");

    while (true) {
        // Doc CHUNK_SIZE samples audio (~1ms tai 8kHz)
        esp_err_t err = i2s_read(I2S_NUM_0, raw_i2s, sizeof(raw_i2s),
                                  &bytes_read, pdMS_TO_TICKS(500));
        if (err != ESP_OK || bytes_read == 0) continue;

        // Doc ADXL dong bo voi moi chunk audio
        readADXLRaw(tempX, tempY, tempZ);

        // Dong goi
        packet.header = 0xBBAA;
        packet.ax     = tempX;
        packet.ay     = tempY;
        packet.az     = tempZ;

        for (int i = 0; i < CHUNK_SIZE; i++) {
            // INMP441: 24-bit data nam o bits [31:8] cua word 32-bit
            // Shift right 14 de ra int16 (bo 8 bit thap + lay 16 bit cao nhat)
            int16_t s     = (int16_t)(raw_i2s[i] >> 14);
            packet.audio[i] = highPassFilter(s);
        }

        // Gui vao queue (non-blocking, neu day thi bo qua)
        xQueueSend(dataQueue, &packet, 0);
    }
}

// ============================================================
// SENDER TASK - Core 1
// Lay packet tu queue, gui qua Serial
// ============================================================

void sender_task(void* pvParameters) {
    SyncPacket out;
    while (true) {
        if (xQueueReceive(dataQueue, &out, portMAX_DELAY)) {
            // Gui raw bytes qua Serial
            Serial.write((uint8_t*)&out, sizeof(SyncPacket));
        }
    }
}

// ============================================================
// SETUP & LOOP
// ============================================================

void setup() {
    Serial.begin(921600);
    delay(500);

    // Tat WiFi va Bluetooth de giam nhieu dien tu
    WiFi.mode(WIFI_OFF);
    btStop();

    Wire.begin(I2C_SDA, I2C_SCL, 400000);
    setup_adxl345();
    setup_i2s();

    // Queue du lon de xu ly burst
    dataQueue = xQueueCreate(64, sizeof(SyncPacket));
    if (!dataQueue) {
        Serial.println("[ERROR] Queue creation failed!");
        while(true) delay(1000);
    }

    // DSP tren Core 0, Sender tren Core 1
    xTaskCreatePinnedToCore(dsp_task,    "DSP",    8192, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(sender_task, "SENDER", 4096, NULL,  5, NULL, 1);
}

void loop() {
    vTaskDelete(NULL);
}
