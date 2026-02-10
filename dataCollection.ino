/*
 * DATA COLLECTOR: INMP441 (I2S) + ADXL335 (Analog)
 * Board: ESP32 Dev Module
 * Upload Speed: 921600
 */

#include <driver/i2s.h>

// --- CẤU HÌNH MIC (INMP441) ---
#define I2S_WS 15
#define I2S_SD 32
#define I2S_SCK 14
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 8000 // 8kHz là đủ cho tiếng máy móc
#define BLOCK_SIZE 64    // Đọc mỗi lần 64 mẫu để buffer

// --- CẤU HÌNH ACCEL (ADXL335) ---
#define PIN_ACCEL_X 36
#define PIN_ACCEL_Y 39
#define PIN_ACCEL_Z 34

void setup() {
  // 1. Serial tốc độ siêu cao (Bắt buộc)
  Serial.begin(921600);
  
  // 2. Cấu hình ADC cho ADXL335
  analogReadResolution(12); // Đọc 12-bit (0-4095)
  // ADXL335 chạy 3.3V, dải đo từ 0-3.3V nên để Attenuation 11db là đẹp
  analogSetAttenuation(ADC_11db); 

  // 3. Cấu hình I2S cho Mic
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BLOCK_SIZE,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  
  delay(1000);
  // Serial.println("START_RECORDING"); 
  // (Không in dòng này để tránh làm bẩn file CSV, Python sẽ tự lo)
}

int16_t mic_buffer[BLOCK_SIZE];

void loop() {
  size_t bytesIn = 0;
  
  // 1. Đọc một cục dữ liệu âm thanh (Block reading)
  // Dùng blocking mode (portMAX_DELAY) để đảm bảo đồng bộ thời gian
  esp_err_t result = i2s_read(I2S_PORT, &mic_buffer, BLOCK_SIZE * sizeof(int16_t), &bytesIn, portMAX_DELAY);

  if (result == ESP_OK) {
    // 2. Đọc giá trị rung động (Chỉ cần đọc 1 lần cho cả Block âm thanh này)
    // Vì rung động thay đổi chậm hơn âm thanh nhiều
    int ax = analogRead(PIN_ACCEL_X);
    int ay = analogRead(PIN_ACCEL_Y);
    int az = analogRead(PIN_ACCEL_Z);

    // 3. Gửi dữ liệu: CSV Format -> Mic, X, Y, Z
    int samples = bytesIn / 2;
    for (int i = 0; i < samples; i++) {
      // Dùng printf để format nhanh gọn
      // Mic, Accel_X, Accel_Y, Accel_Z
      Serial.printf("%d,%d,%d,%d\n", mic_buffer[i], ax, ay, az);
    }
  }
}