// Firmware thu dữ liệu máy giặt cho XIAO ESP32-S3.
// Gửi các gói âm thanh INMP441 và rung ADXL345 đã đồng bộ qua Serial.

#include <Wire.h>
#include <WiFi.h>
#include "driver/i2s.h"
#include "esp_bt.h"

#define I2S_WS        6
#define I2S_BCLK      43
#define I2S_DIN       44
#define I2C_SDA       5
#define I2C_SCL       4
#define ADXL345_ADDR  0x53

#define AUDIO_SAMPLE_RATE 8000
#define SERIAL_BAUD       921600
#define AUDIO_CHUNK_SIZE  8

struct __attribute__((packed)) SensorPacket {
    uint16_t header;                 // 0xBBAA sẽ xuất hiện trên dây dưới dạng AA BB.
    int16_t ax, ay, az;              // Giá trị raw ADXL345; chia 256.0 để đổi sang g.
    int16_t audio[AUDIO_CHUNK_SIZE]; // Mẫu PCM INMP441 sau lọc thông cao.
};

static SensorPacket packet;
static QueueHandle_t packet_queue = nullptr;
static int32_t hpf_prev_x = 0;
static int32_t hpf_prev_y = 0;

// Loại bỏ lệch DC của microphone trước khi gửi mẫu về máy tính.
static int16_t high_pass_filter(int16_t x) {
    int32_t cx = (int32_t)x;
    int32_t cy = (243 * (hpf_prev_y + cx - hpf_prev_x)) >> 8;
    hpf_prev_x = cx;
    hpf_prev_y = cy;
    return (int16_t)cy;
}

// Đọc một mẫu gia tốc XYZ raw từ thanh ghi dữ liệu của ADXL345.
static void read_adxl_raw(int16_t &x, int16_t &y, int16_t &z) {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x32);
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

// Cấu hình ADXL345 ở chế độ đo với dải đo và tốc độ dữ liệu của đồ án.
static void setup_adxl345() {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2D);
    Wire.write(0x08);
    Wire.endTransmission();

    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x31);
    Wire.write(0x00); // Dải +/-2g, chế độ 10-bit.
    Wire.endTransmission();

    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x2C);
    Wire.write(0x0B); // ODR 200 Hz.
    Wire.endTransmission();

    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(0x00);
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL345_ADDR, 1);
    uint8_t devid = Wire.available() ? Wire.read() : 0;

    if (devid == 0xE5) {
        Serial.println("[ADXL] OK (0xE5)");
    } else {
        Serial.printf("[ADXL] WARN devid=0x%02X\n", devid);
    }
}

// Cấu hình I2S RX để nhận luồng mẫu từ microphone INMP441.
static void setup_i2s() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN,
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    Serial.println("[I2S] OK");
}

// Task thu dữ liệu: đọc ADXL345/INMP441 rồi đóng gói frame để gửi Serial.
static void capture_task(void*) {
    int32_t raw_i2s[AUDIO_CHUNK_SIZE];
    size_t bytes_read = 0;
    int16_t ax, ay, az;

    for (int i = 0; i < 16; i++) {
        i2s_read(I2S_NUM_0, raw_i2s, sizeof(raw_i2s), &bytes_read, pdMS_TO_TICKS(100));
    }

    Serial.println("[CAPTURE] Ready");

    while (true) {
        esp_err_t err = i2s_read(
            I2S_NUM_0,
            raw_i2s,
            sizeof(raw_i2s),
            &bytes_read,
            pdMS_TO_TICKS(500)
        );
        if (err != ESP_OK || bytes_read == 0) {
            continue;
        }

        read_adxl_raw(ax, ay, az);

        packet.header = 0xBBAA;
        packet.ax = ax;
        packet.ay = ay;
        packet.az = az;

        int samples = bytes_read / sizeof(int32_t);
        for (int i = 0; i < AUDIO_CHUNK_SIZE; i++) {
            int16_t sample = 0;
            if (i < samples) {
                sample = (int16_t)(raw_i2s[i] >> 14);
            }
            packet.audio[i] = high_pass_filter(sample);
        }

        xQueueSend(packet_queue, &packet, 0);
    }
}

// Task Serial: truyền gói nhị phân sang `final_data_collection.py`.
static void serial_task(void*) {
    SensorPacket out;
    while (true) {
        if (xQueueReceive(packet_queue, &out, portMAX_DELAY)) {
            Serial.write((uint8_t*)&out, sizeof(SensorPacket));
        }
    }
}

// Khởi tạo cảm biến, queue và hai task FreeRTOS phục vụ thu dữ liệu.
void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(500);

    WiFi.mode(WIFI_OFF);
    btStop();

    Wire.begin(I2C_SDA, I2C_SCL, 400000);
    setup_adxl345();
    setup_i2s();

    packet_queue = xQueueCreate(64, sizeof(SensorPacket));
    if (!packet_queue) {
        Serial.println("[ERROR] packet queue allocation failed");
        while (true) delay(1000);
    }

    xTaskCreatePinnedToCore(capture_task, "CAPTURE", 8192, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(serial_task, "SERIAL", 4096, NULL, 5, NULL, 1);
}

// Toàn bộ xử lý chạy trong task; vòng lặp Arduino chỉ nhường CPU.
void loop() {
    vTaskDelete(NULL);
}
