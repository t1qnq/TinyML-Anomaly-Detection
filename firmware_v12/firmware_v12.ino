// ============================================================
//  TinyML Anomaly Detection - Xiao ESP32S3
//  Version: 10.0 — True Parallel Dual-Core
//
//  Kiến trúc mới (tách từ dsp_task đơn lẻ của v9):
//    Core 0 | audio_task  : I2S capture + FFT → audio_queue (float[MEL_DIM])
//    Core 1 | vib_task    : ADXL345 @ 1ms chính xác → vib_queue (float[VIB_DIM])
//    Core 1 | ai_task     : EventGroup AND-wait → gộp → inference
//
//  Hardware:
//    INMP441 microphone   : WS=6, BCLK=43, DIN=44
//    ADXL345 accelerometer: SDA=5, SCL=4
// ============================================================

#include <Wire.h>
#include <WiFi.h>
#include "driver/i2s.h"
#include "esp_dsp.h"
#include "freertos/event_groups.h"
#include "model_data.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ============================================================
// SECTION 1: CONSTANTS & ENUMS
// ============================================================

#define PIN_I2S_WS    6
#define PIN_I2S_BCLK  43
#define PIN_I2S_DIN   44
#define PIN_I2C_SDA   5
#define PIN_I2C_SCL   4
#define ADXL_ADDR     0x53

enum AiMode { MODE_GENTLE, MODE_STRONG, MODE_SPIN };

#define SAMPLE_RATE    8000
#define FFT_SIZE       512
#define HOP_SIZE       256
#define NUM_FRAMES     30
#define CHUNKS_PER_HOP (HOP_SIZE / 8)

#define VIB_SAMPLES    1000

#define MEL_BANDS      13
#define FEAT_DIM       19
#define MEL_DIM        13
#define VIB_DIM        6
#define VIB_IDX        13

#define SILENCE_THR    1e5f

#define WEIGHT_MEL     1.0f
#define WEIGHT_VIB     5.0f
#define WEIGHT_SUM     (MEL_DIM * WEIGHT_MEL + VIB_DIM * WEIGHT_VIB)

// ============================================================
// SECTION 2: MEL FILTERBANK
// ============================================================

struct MelBand { int lo, mid, hi; };

static const MelBand MEL_FB[MEL_BANDS] = {
    {  1, 11, 21 }, { 11, 21, 32 }, { 22, 32, 42 },
    { 33, 43, 53 }, { 43, 54, 64 }, { 54, 64, 76 },
    { 65, 76, 90 }, { 77, 91,107 }, { 91,108,128 },
    {108,128,152 }, {129,153,181 }, {153,181,215 },
    {182,215,255 },
};

static void mel_power(const float* fft, float* out) {
    for (int b = 0; b < MEL_BANDS; b++) {
        float p = 0.0f;
        const int lo = MEL_FB[b].lo, mid = MEL_FB[b].mid, hi = MEL_FB[b].hi;
        for (int k = lo; k <= mid && k <= FFT_SIZE/2; k++) {
            float w  = (mid > lo) ? (float)(k - lo) / (float)(mid - lo) : 1.0f;
            float re = fft[k*2], im = fft[k*2+1];
            float m2 = (re*re + im*im) * ((k > 0 && k < FFT_SIZE/2) ? 2.0f : 1.0f);
            p += w * m2;
        }
        for (int k = mid+1; k <= hi && k <= FFT_SIZE/2; k++) {
            float w  = (hi > mid) ? (float)(hi - k) / (float)(hi - mid) : 0.0f;
            float re = fft[k*2], im = fft[k*2+1];
            float m2 = (re*re + im*im) * ((k > 0 && k < FFT_SIZE/2) ? 2.0f : 1.0f);
            p += w * m2;
        }
        out[b] = p + 1e-10f;
    }
}

// ============================================================
// SECTION 3: GLOBALS, QUEUES & SYNC
// ============================================================

const char* ssid = "Coin";
const char* password = "01152718";
const char* mqtt_server = "broker.emqx.io";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

void reconnect_mqtt() {
    if (!mqttClient.connected()) {
        if (mqttClient.connect("ESP32_WashingMachine")) {
            Serial.println("[MQTT] Connected");
        }
    }
}

static int32_t hpf_x = 0, hpf_y = 0;

static inline int16_t hpf(int16_t in) {
    int32_t cx = (int32_t)in;
    int32_t cy = (243 * (hpf_y + cx - hpf_x)) >> 8;
    hpf_x = cx; hpf_y = cy;
    return (int16_t)cy;
}

struct FeatureVec {
    float raw[FEAT_DIM];
    float scaled[FEAT_DIM];
};

// --- Tách thành 2 queue riêng biệt thay vì 1 feat_queue gộp sẵn ---
static QueueHandle_t     audio_queue  = nullptr;  // float[MEL_DIM]
static QueueHandle_t     vib_queue    = nullptr;  // float[VIB_DIM]
static SemaphoreHandle_t model_mutex  = nullptr;

// EventGroup: ai_task đợi CẢ HAI bit trước khi chạy inference
static EventGroupHandle_t sync_events = nullptr;
#define AUDIO_READY_BIT  (1 << 0)
#define VIB_READY_BIT    (1 << 1)
#define BOTH_READY_BITS  (AUDIO_READY_BIT | VIB_READY_BIT)

#define TENSOR_ARENA_SIZE (64 * 1024)

static uint8_t* tensor_arena_gentle = nullptr;
static uint8_t* tensor_arena_strong = nullptr;
static uint8_t* tensor_arena_spin   = nullptr;

static tflite::MicroInterpreter* interp_gentle = nullptr;
static tflite::MicroInterpreter* interp_strong = nullptr;
static tflite::MicroInterpreter* interp_spin   = nullptr;

alignas(16) static uint8_t interp_buf_gentle[sizeof(tflite::MicroInterpreter)];
alignas(16) static uint8_t interp_buf_strong[sizeof(tflite::MicroInterpreter)];
alignas(16) static uint8_t interp_buf_spin  [sizeof(tflite::MicroInterpreter)];

static tflite::MicroErrorReporter    micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static tflite::MicroMutableOpResolver<10> resolver;
static bool resolver_ready = false;

// Stats
static uint32_t total_wins   = 0;
static uint32_t silent_wins  = 0;
static uint32_t alarm_wins   = 0;
static uint32_t gentle_wins  = 0;
static uint32_t strong_wins  = 0;
static uint32_t spin_wins    = 0;
static uint32_t consec_alarm = 0;

// ── Sliding Window Alarm ──
// Vào ALARM: 5/10 win gần nhất là HIGH
// Thoát ALARM: 9/10 win gần nhất là OK
#define ALARM_WINDOW     10
#define ALARM_ENTER_THR   5   // >= 5 HIGH → vào ALARM
#define ALARM_EXIT_THR    1   // <= 1 HIGH (tức 9 OK) → thoát ALARM

// ── MQTT Smart Publish ──
// Cấu trúc chia sẻ giữa ai_task → mqtt_task (lock-free via queue)
struct MqttPayload {
    float   mae;
    uint32_t win;
    uint32_t consec;
    bool    is_alarm;
    bool    over_thr;
    char    mode[8];    // "GENTLE", "STRONG", "SPIN"
};
static QueueHandle_t mqtt_queue = nullptr;  // MqttPayload

#define MQTT_HEARTBEAT_SEC  30  // Gửi heartbeat mỗi 30s khi bình thường

static const float FEAT_WEIGHTS[FEAT_DIM] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,   // mel
    5,5,5,5,5,5                   // vib
};

// ============================================================
// SECTION 4: ADXL345 & I2S
// ============================================================
static void adxl_init() {
    Wire.beginTransmission(ADXL_ADDR); Wire.write(0x2D); Wire.write(0x08); Wire.endTransmission();
    Wire.beginTransmission(ADXL_ADDR); Wire.write(0x31); Wire.write(0x00); Wire.endTransmission();
    Wire.beginTransmission(ADXL_ADDR); Wire.write(0x2C); Wire.write(0x0B); Wire.endTransmission(); // 200Hz ODR

    uint8_t id = 0;
    Wire.beginTransmission(ADXL_ADDR); Wire.write(0x00); Wire.endTransmission(false);
    Wire.requestFrom(ADXL_ADDR, 1);
    if (Wire.available()) id = Wire.read();
    if (id == 0xE5) Serial.println("[ADXL] OK (0xE5)");
    else            Serial.printf("[ADXL] WARN id=0x%02X\n", id);
}

static void adxl_read(int16_t& rx, int16_t& ry, int16_t& rz) {
    Wire.beginTransmission(ADXL_ADDR); Wire.write(0x32); Wire.endTransmission(false);
    Wire.requestFrom(ADXL_ADDR, 6);
    uint8_t buf[6];
    for (int i = 0; i < 6 && Wire.available(); i++) buf[i] = Wire.read();
    rx = (int16_t)((buf[1] << 8) | buf[0]);
    ry = (int16_t)((buf[3] << 8) | buf[2]);
    rz = (int16_t)((buf[5] << 8) | buf[4]);
}

static void i2s_init() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = PIN_I2S_BCLK, .ws_io_num = PIN_I2S_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = PIN_I2S_DIN,
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    Serial.println("[I2S] OK");
}

// ============================================================
// SECTION 5: SCALING
// ============================================================
static void apply_scale(const float* raw, float* out, AiMode mode) {
    for (int j = 0; j < MEL_DIM; j++) {
        float v = (raw[j] + 80.0f) / 80.0f;
        out[j]  = fmaxf(0.0f, fminf(1.0f, v));
    }
    float vib[VIB_DIM];
    for (int j = 0; j < VIB_DIM; j++) vib[j] = raw[VIB_IDX + j];
    vib[1] = fminf(vib[1], VIB_CLIP_VAR_X);
    vib[3] = fminf(vib[3], VIB_CLIP_VAR_Y);
    vib[5] = fminf(vib[5], VIB_CLIP_VAR_Z);

    const float* center;
    const float* scale;

    if (mode == MODE_SPIN) {
        center = VIB_CENTER_SPIN; scale = VIB_SCALE_SPIN;
    } else if (mode == MODE_STRONG) {
        center = VIB_CENTER_STRONG; scale = VIB_SCALE_STRONG;
    } else {
        center = VIB_CENTER_GENTLE; scale = VIB_SCALE_GENTLE;
    }

    for (int j = 0; j < VIB_DIM; j++) {
        float v = (vib[j] - center[j]) / scale[j];
        v = fmaxf(-3.0f, fminf(3.0f, v));
        out[VIB_IDX + j] = v / 6.0f + 0.5f;
    }
}

// ============================================================
// SECTION 6: TFLITE MODELS
// ============================================================
static void init_resolver() {
    if (resolver_ready) return;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddResizeNearestNeighbor();
    resolver.AddSlice();
    resolver.AddLogistic();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver_ready = true;
}

static bool load_one_model(const uint8_t* model_data, size_t model_len, uint8_t* arena, size_t arena_size, uint8_t* interp_buf, tflite::MicroInterpreter*& interp_out, const char* name) {
    if (interp_out) { interp_out->~MicroInterpreter(); interp_out = nullptr; }
    const tflite::Model* m = tflite::GetModel(model_data);
    if (m->version() != TFLITE_SCHEMA_VERSION) { Serial.printf("[%s] Schema mismatch\n", name); return false; }
    interp_out = new(interp_buf) tflite::MicroInterpreter(m, resolver, arena, arena_size, error_reporter);
    if (interp_out->AllocateTensors() != kTfLiteOk) { Serial.printf("[%s] AllocateTensors FAIL\n", name); return false; }
    return true;
}

static bool load_models() {
    bool ok = true;
    ok &= load_one_model(model_gentle_tflite, model_gentle_tflite_len, tensor_arena_gentle, TENSOR_ARENA_SIZE, interp_buf_gentle, interp_gentle, "GENTLE");
    ok &= load_one_model(model_strong_tflite, model_strong_tflite_len, tensor_arena_strong, TENSOR_ARENA_SIZE, interp_buf_strong, interp_strong, "STRONG");
    ok &= load_one_model(model_spin_tflite,   model_spin_tflite_len,   tensor_arena_spin,   TENSOR_ARENA_SIZE, interp_buf_spin,   interp_spin,   "SPIN");
    return ok;
}

static int8_t quant(float v, float scale, int32_t zp) { return (int8_t)constrain((int32_t)roundf(v / scale) + zp, -128, 127); }
static float dequant(int8_t v, float scale, int32_t zp) { return ((float)v - (float)zp) * scale; }

// ============================================================
// SECTION 7: CORE 0 — AUDIO TASK (I2S + FFT → audio_queue)
// Tách ra từ dsp_task. Chỉ lo audio, không đụng ADXL/I2C.
// ============================================================
void audio_task(void*) {
    static float hann[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++) hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));

    static float overlap[FFT_SIZE];
    static float fft_buf[FFT_SIZE * 2];
    static float mel_frames[NUM_FRAMES][MEL_BANDS];

    memset(overlap, 0, sizeof(overlap));
    Serial.println("[AUDIO] Ready");

    while (true) {
        float mel_max = 0.0f;
        bool  ok      = true;

        memset(overlap, 0, sizeof(overlap));

        // Thu ~1 giây audio (NUM_FRAMES hop)
        for (int fr = 0; fr < NUM_FRAMES && ok; fr++) {
            int new_idx = HOP_SIZE, chunks = 0;
            while (chunks < CHUNKS_PER_HOP) {
                int32_t raw[8]; size_t br;
                if (i2s_read(I2S_NUM_0, raw, sizeof(raw), &br, pdMS_TO_TICKS(500)) != ESP_OK || br == 0) { ok = false; break; }
                int n = (int)(br / sizeof(int32_t));
                for (int i = 0; i < n && new_idx < FFT_SIZE; i++, new_idx++) {
                    overlap[new_idx] = (float)hpf((int16_t)(raw[i] >> 14));
                }
                chunks++;
            }
            if (!ok) break;

            for (int i = 0; i < HOP_SIZE; i++)        { fft_buf[i*2] = overlap[i] * hann[i]; fft_buf[i*2+1] = 0.0f; }
            for (int i = HOP_SIZE; i < FFT_SIZE; i++) { fft_buf[i*2] = overlap[i] * hann[i]; fft_buf[i*2+1] = 0.0f; }
            memmove(overlap, overlap + HOP_SIZE, HOP_SIZE * sizeof(float));
            memset(overlap + HOP_SIZE, 0, HOP_SIZE * sizeof(float));

            dsps_fft2r_fc32(fft_buf, FFT_SIZE);
            dsps_bit_rev_fc32(fft_buf, FFT_SIZE);

            mel_power(fft_buf, mel_frames[fr]);
            for (int b = 0; b < MEL_BANDS; b++) if (mel_frames[fr][b] > mel_max) mel_max = mel_frames[fr][b];
        }

        if (!ok) { continue; }

        // Tính log-mel trung bình (feat = 0 nếu silence, ai_task tự bỏ qua)
        float feat[MEL_DIM] = {0};
        if (mel_max >= SILENCE_THR) {
            if (mel_max < 1e-10f) mel_max = 1e-10f;
            for (int b = 0; b < MEL_BANDS; b++) {
                float sum = 0.0f;
                for (int fr = 0; fr < NUM_FRAMES; fr++) sum += fmaxf(10.0f * log10f(mel_frames[fr][b] / mel_max), -80.0f);
                feat[b] = sum / NUM_FRAMES;
            }
        } else {
            silent_wins++;
        }

        // Ghi đè frame cũ nếu ai_task chưa kịp tiêu thụ
        if (xQueueSend(audio_queue, feat, 0) != pdTRUE) {
            float dummy[MEL_DIM];
            xQueueReceive(audio_queue, dummy, 0);
            xQueueSend(audio_queue, feat, 0);
        }
        xEventGroupSetBits(sync_events, AUDIO_READY_BIT);
    }
}

// ============================================================
// SECTION 8: CORE 1 — VIB TASK (ADXL345 @ 1ms → vib_queue)
// Priority cao nhất trên Core 1.
// Dùng vTaskDelayUntil thay vTaskDelay để timing không bị trôi.
// ============================================================
void vib_task(void*) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1);
    Serial.println("[VIB] Ready");

    while (true) {
        float sum[3] = {0,0,0}, sq_sum[3] = {0,0,0};
        int16_t rx, ry, rz;

        // Lấy đúng 1000 mẫu, mỗi mẫu cách nhau đúng 1ms (không tích lũy sai số)
        for (int i = 0; i < VIB_SAMPLES; i++) {
            adxl_read(rx, ry, rz);
            float ax = rx / 256.0f, ay = ry / 256.0f, az = rz / 256.0f;
            sum[0]+=ax; sq_sum[0]+=ax*ax;
            sum[1]+=ay; sq_sum[1]+=ay*ay;
            sum[2]+=az; sq_sum[2]+=az*az;
            vTaskDelayUntil(&xLastWakeTime, xFrequency);
        }

        // Tính RMS và Variance cho 3 trục
        float feat[VIB_DIM]; float n = (float)VIB_SAMPLES;
        for (int a = 0; a < 3; a++) {
            float mean = sum[a] / n;
            feat[a*2]   = sqrtf(sq_sum[a] / n);          // RMS
            feat[a*2+1] = (sq_sum[a] / n) - mean * mean;  // Variance
        }

        // Ghi đè nếu ai_task chưa kịp tiêu thụ
        if (xQueueSend(vib_queue, feat, 0) != pdTRUE) {
            float dummy[VIB_DIM];
            xQueueReceive(vib_queue, dummy, 0);
            xQueueSend(vib_queue, feat, 0);
        }
        xEventGroupSetBits(sync_events, VIB_READY_BIT);
    }
}

// ============================================================
// SECTION 9: CORE 1 — AI TASK (Đợi cả 2 → gộp → inference)
// Priority thấp hơn vib_task → không tranh timing ADXL.
// ============================================================
void ai_task(void*) {
    while (!audio_queue || !vib_queue || !model_mutex || !sync_events) vTaskDelay(10);
    Serial.println("[AI] Ready, waiting for both queues...");

    float mel_in[MEL_DIM], vib_in[VIB_DIM];
    FeatureVec fv;

    // Giữ nguyên ngưỡng vật lý từ v9
    const float VAR_Z_THR1 = 0.105845f;
    const float VAR_Z_THR2 = 0.386260f;

    uint32_t grace_period_wins = 0;
    const uint32_t GRACE_WINDOWS = 0;
    bool was_idle = true;

    bool last_alarm_state = false;

    while (true) {
        // ── Đợi TÍN HIỆU cả 2 nguồn đồng thời (AND logic) ──
        // Timeout 2500ms để phát hiện nếu một task bị treo
        EventBits_t bits = xEventGroupWaitBits(
            sync_events,
            BOTH_READY_BITS,
            pdTRUE,          // Tự clear bits sau khi nhận
            pdTRUE,          // AND: phải có CẢ HAI bit
            pdMS_TO_TICKS(2500)
        );

        if ((bits & BOTH_READY_BITS) != BOTH_READY_BITS) {
            Serial.printf("[AI] WARN: sync timeout! audio=%d vib=%d\n",
                (bits & AUDIO_READY_BIT) != 0, (bits & VIB_READY_BIT) != 0);
            continue;
        }

        // Kéo data (non-blocking, biết chắc có sẵn)
        if (xQueueReceive(audio_queue, mel_in, 0) != pdTRUE ||
            xQueueReceive(vib_queue,   vib_in, 0) != pdTRUE) {
            Serial.println("[AI] WARN: queue empty after event, skipping");
            continue;
        }

        // Gộp thành FeatureVec
        memcpy(fv.raw,           mel_in, sizeof(mel_in));
        memcpy(fv.raw + MEL_DIM, vib_in, sizeof(vib_in));
        total_wins++;

        // Log RAW (giống v9)
        Serial.printf("[RAW] mel: %.1f %.1f ... %.1f | rms: x=%.4f y=%.4f z=%.4f | var: x=%.6f y=%.6f z=%.6f\n",
            fv.raw[0], fv.raw[1], fv.raw[MEL_DIM-1],
            fv.raw[13], fv.raw[15], fv.raw[17],
            fv.raw[14], fv.raw[16], fv.raw[18]);

        // --- 1. NHẬN DIỆN KHỞI ĐỘNG (Smart Grace Period) ---
        bool currently_idle = (fv.raw[16] < 0.000035f);
        if (currently_idle) {
            was_idle = true;
        } else if (was_idle) {
            grace_period_wins = GRACE_WINDOWS;
            was_idle = false;
        }

        // --- 2. PHÂN LUỒNG TRI-STATE ---
        AiMode ai_mode;
        tflite::MicroInterpreter* interp = nullptr;
        float thr = 0.0f;
        const char* mode_str = "";

        if (fv.raw[18] >= VAR_Z_THR2) {
            ai_mode = MODE_SPIN;   interp = interp_spin;   thr = THRESHOLD_SPIN;   mode_str = "SPIN";   spin_wins++;
        } else if (fv.raw[18] >= VAR_Z_THR1) {
            ai_mode = MODE_STRONG; interp = interp_strong; thr = THRESHOLD_STRONG; mode_str = "STRONG"; strong_wins++;
        } else {
            ai_mode = MODE_GENTLE; interp = interp_gentle; thr = THRESHOLD_GENTLE; mode_str = "GENTLE"; gentle_wins++;
        }

        apply_scale(fv.raw, fv.scaled, ai_mode);
        if (!interp) continue;

        xSemaphoreTake(model_mutex, portMAX_DELAY);
        TfLiteTensor* inp = interp->input(0);
        TfLiteTensor* out = interp->output(0);

        int8_t xin[FEAT_DIM];
        for (int i = 0; i < FEAT_DIM; i++) {
            int8_t q = quant(fv.scaled[i], inp->params.scale, inp->params.zero_point);
            inp->data.int8[i] = q; xin[i] = q;
        }

        int64_t t0 = esp_timer_get_time();
        TfLiteStatus s = interp->Invoke();
        uint32_t us = (uint32_t)(esp_timer_get_time() - t0);

        if (s != kTfLiteOk) {
            Serial.printf("[AI] Invoke fail mode=%s\n", mode_str);
            xSemaphoreGive(model_mutex);
            continue;
        }

        int8_t yout[FEAT_DIM];
        for (int i = 0; i < FEAT_DIM; i++) yout[i] = out->data.int8[i];

        float mae_float = 0.0f, mae_int8 = 0.0f;
        for (int i = 0; i < FEAT_DIM; i++) {
            float pred = fmaxf(0.0f, fminf(1.0f, dequant(yout[i], out->params.scale, out->params.zero_point)));
            mae_float += fabsf(fv.scaled[i] - pred) * FEAT_WEIGHTS[i];
            float xf = fmaxf(0.0f, fminf(1.0f, dequant(xin[i], inp->params.scale, inp->params.zero_point)));
            mae_int8  += fabsf(xf - pred) * FEAT_WEIGHTS[i];
        }
        mae_float /= WEIGHT_SUM;
        mae_int8  /= WEIGHT_SUM;

        bool over_thr = mae_int8 > thr;

        // Grace period
        if (grace_period_wins > 0) {
            grace_period_wins--;
            over_thr = false;
            Serial.printf("[GRACE  ][%s] MAE:%.4f THR:%.4f (Ignored) | wins:%lu\n", mode_str, mae_int8, thr, total_wins);
            xSemaphoreGive(model_mutex);
            continue;
        }

        // Blackbox logger
        if (over_thr) {
            Serial.printf("[FEATURES] IN: ");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf("%.2f ", fv.scaled[i]);
            Serial.printf("| OUT: ");
            for (int i = 0; i < FEAT_DIM; i++) {
                float pred = fmaxf(0.0f, fminf(1.0f, dequant(yout[i], out->params.scale, out->params.zero_point)));
                Serial.printf("%.2f ", pred);
            }
            Serial.println();
        }

        // ── Sliding Window Alarm Logic ──
        // Đẩy kết quả vào ring buffer
        static bool win_ring[ALARM_WINDOW] = {false};
        static int  win_idx = 0;
        static bool in_alarm_state = false;

        win_ring[win_idx] = over_thr;
        win_idx = (win_idx + 1) % ALARM_WINDOW;

        // Đếm số HIGH trong 10 win gần nhất
        int high_count = 0;
        for (int i = 0; i < ALARM_WINDOW; i++) {
            if (win_ring[i]) high_count++;
        }

        // Hysteresis: bất đối xứng vào/ra
        if (!in_alarm_state && high_count >= ALARM_ENTER_THR) {
            in_alarm_state = true;   // Vào ALARM
        } else if (in_alarm_state && high_count <= (ALARM_WINDOW - 9)) {
            in_alarm_state = false;  // Thoát ALARM (9/10 OK)
        }

        bool alarm = in_alarm_state;
        if (over_thr) consec_alarm++; else consec_alarm = 0;
        if (over_thr && consec_alarm == 1) alarm_wins++;

        // --- SMART MQTT: Đẩy payload vào queue (non-blocking, ~0µs) ---
        // Quyết định có gửi hay không do mqtt_task xử lý
        MqttPayload payload;
        payload.mae      = mae_int8;
        payload.win      = total_wins;
        payload.consec   = high_count;  // Số HIGH trong 10 win gần nhất
        payload.is_alarm = alarm;
        payload.over_thr = over_thr;
        strncpy(payload.mode, mode_str, sizeof(payload.mode));
        
        // Ghi đè nếu mqtt_task chưa kịp tiêu thụ (không bao giờ block)
        if (xQueueSend(mqtt_queue, &payload, 0) != pdTRUE) {
            MqttPayload dummy;
            xQueueReceive(mqtt_queue, &dummy, 0);
            xQueueSend(mqtt_queue, &payload, 0);
        }

        if (alarm != last_alarm_state) {
            Serial.printf("[MQTT] >>> TRẠNG THÁI THAY ĐỔI: %s <<<\n", alarm ? "ALARM" : "OK");
            last_alarm_state = alarm;
        }

        Serial.printf("[%s][%s] MAE:%.4f (f:%.4f) THR:%.4f consec:%lu t:%uus | wins:%lu g:%lu st:%lu sp:%lu\n",
            alarm ? "ALARM" : (over_thr ? "HIGH " : "OK   "),
            mode_str, mae_int8, mae_float, thr, consec_alarm, us,
            total_wins, gentle_wins, strong_wins, spin_wins);

        xSemaphoreGive(model_mutex);
    }
}

// ============================================================
// SECTION 10: MQTT TASK — Chạy trên loop(), tách khỏi AI
// Chiến lược gửi:
//   1. Heartbeat mỗi 30s khi OK (giữ App không LOST)
//   2. HIGH: gửi 1 gói ĐẦU TIÊN khi vào HIGH, 1 gói khi thoát HIGH
//   3. ALARM: gửi LIÊN TỤC (~1Hz) suốt quá trình alarm
//   4. Mọi chuyển trạng thái (OK↔HIGH↔ALARM): gửi NGAY LẬP TỨC
//   5. KHÔNG BAO GIỜ block luồng suy luận
// ============================================================
// 3 trạng thái: 0=OK, 1=HIGH, 2=ALARM
static uint8_t  mqtt_last_state = 0;
static uint32_t mqtt_last_send  = 0;

void mqtt_smart_publish() {
    MqttPayload p;
    if (xQueueReceive(mqtt_queue, &p, 0) != pdTRUE) return;
    if (!mqttClient.connected()) return;

    // Phân loại trạng thái hiện tại
    uint32_t now       = millis();
    uint8_t  cur_state = p.is_alarm ? 2 : (p.over_thr ? 1 : 0);
    bool     changed   = (cur_state != mqtt_last_state);
    bool     should_send = false;

    if (changed || (now - mqtt_last_send >= 1000UL)) {
        should_send = true;
        if (changed) {
            const char* names[] = {"OK", "HIGH", "ALARM"};
            Serial.printf("[MQTT] Edge: %s → %s\n", names[mqtt_last_state], names[cur_state]);
        }
    }
    // HIGH không thay đổi → KHÔNG gửi (chỉ gửi gói đầu tiên ở trên)

    if (!should_send) return;

    StaticJsonDocument<256> doc;
    doc["state"]    = p.is_alarm ? "ALARM" : (p.over_thr ? "HIGH" : "OK");
    doc["mae"]      = p.mae;
    doc["is_alarm"] = p.is_alarm;
    doc["win"]      = p.win;
    doc["consec"]   = p.consec;
    doc["mode"]     = p.mode;

    char buf[256];
    serializeJson(doc, buf);
    mqttClient.publish("tinyml/quang_wm_2026/status", buf);

    mqtt_last_state = cur_state;
    mqtt_last_send  = now;
}

// ============================================================
// SECTION 11: SETUP & LOOP
// ============================================================
void setup() {
    Serial.begin(921600);
    delay(500);
    Serial.println("=== BOOT v10.0 (True Parallel Dual-Core) ===");
    Serial.println("  Core 0: AUDIO (I2S + FFT)");
    Serial.println("  Core 1: VIB (ADXL@1ms, prio=15) + AI (prio=5)");

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("[WIFI] Connecting...");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println("\n[WIFI] Connected");
    
    mqttClient.setServer(mqtt_server, 1883);

    tensor_arena_gentle = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    tensor_arena_strong = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    tensor_arena_spin   = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);

    if (!tensor_arena_gentle || !tensor_arena_strong || !tensor_arena_spin) {
        Serial.println("[ERROR] PSRAM alloc fail"); while (1) vTaskDelay(1000);
    }
    memset(tensor_arena_gentle, 0, TENSOR_ARENA_SIZE);
    memset(tensor_arena_strong, 0, TENSOR_ARENA_SIZE);
    memset(tensor_arena_spin,   0, TENSOR_ARENA_SIZE);

    if (dsps_fft2r_init_fc32(NULL, FFT_SIZE) != ESP_OK) {
        Serial.println("[ERROR] FFT init fail"); while (1) vTaskDelay(1000);
    }

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    adxl_init();
    i2s_init();

    tflite::InitializeTarget();
    init_resolver();
    if (!load_models()) { Serial.println("[ERROR] model load fail"); while (1) vTaskDelay(1000); }

    // Khởi tạo RTOS objects
    model_mutex = xSemaphoreCreateMutex();
    sync_events = xEventGroupCreate();
    audio_queue = xQueueCreate(2, sizeof(float) * MEL_DIM);
    vib_queue   = xQueueCreate(2, sizeof(float) * VIB_DIM);
    mqtt_queue  = xQueueCreate(2, sizeof(MqttPayload));

    if (!model_mutex || !sync_events || !audio_queue || !vib_queue || !mqtt_queue) {
        Serial.println("[ERROR] RTOS object create fail"); while (1) vTaskDelay(1000);
    }

    // Core 0: audio_task — I2S+FFT, không đụng I2C
    xTaskCreatePinnedToCore(audio_task, "AUDIO", 40960, NULL, 10, NULL, 0);

    // Core 1: vib_task — priority CAO (15), giữ timing 1ms tuyệt đối
    xTaskCreatePinnedToCore(vib_task,   "VIB",   8192,  NULL, 15, NULL, 1);

    // Core 1: ai_task — priority THẤP (5), chạy khi vib_task idle
    xTaskCreatePinnedToCore(ai_task,    "AI",    24576, NULL,  5, NULL, 1);

    Serial.println("=== READY v10.0 ===");
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        reconnect_mqtt();
        mqttClient.loop();
    }
    // Smart publish: xử lý MQTT tách biệt khỏi AI
    mqtt_smart_publish();
    vTaskDelay(pdMS_TO_TICKS(50));
}