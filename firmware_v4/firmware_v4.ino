// ============================================================
//  TinyML Anomaly Detection - Xiao ESP32S3
//  Version: 7.0 - Single Autoencoder
//
//  Hardware:
//    INMP441 microphone  : WS=6, BCLK=43, DIN=44
//    ADXL345 accelerometer: SDA=5, SCL=4
//
//  Feature vector (19-dim):
//    [0..12]  mel dB bands (norm=None, center=False)
//    [13][14] rms_x, var_x  (X = doc = trong luc ~1g)
//    [15][16] rms_y, var_y
//    [17][18] rms_z, var_z
//
//  Pipeline:
//    I2S -> HPF -> mel(30 frames, 512pt FFT) -> 13 mel features
//    ADXL 1000 reads @ 1ms -> rms + var -> 6 vib features
//    MEL: global MinMaxScaler -> clip[0,1]
//    VIB: global RobustScaler -> clip(-3,3)/6+0.5
//    1 AE model -> MAE vs THRESHOLD -> ALARM/OK
//
//  model_data.h chua:
//    THRESHOLD                  - 1 gia tri duy nhat (tu INT8 MAE)
//    MEL_MIN[13], MEL_SCALE[13] - global mel scaler
//    VIB_CENTER[6], VIB_SCALE[6]- global vib scaler
//    model_single_ae_tflite[]   - INT8 TFLite model
// ============================================================

#include <Wire.h>
#include <WiFi.h>
#include "driver/i2s.h"
#include "esp_dsp.h"
#include "model_data.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================
// CRC32 (model fingerprint)
// ============================================================
static uint32_t crc32_ieee(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

// ============================================================
// SECTION 1: CONSTANTS
// ============================================================

#define PIN_I2S_WS    6
#define PIN_I2S_BCLK  43
#define PIN_I2S_DIN   44
#define PIN_I2C_SDA   5
#define PIN_I2C_SCL   4
#define ADXL_ADDR     0x53

#define SAMPLE_RATE    8000
#define FFT_SIZE       512
#define HOP_SIZE       256
#define NUM_FRAMES     30
#define CHUNKS_PER_HOP (HOP_SIZE / 8)

#define MEL_BANDS      13
#define VIB_SAMPLES    1000
#define FEAT_DIM       19
#define MEL_DIM        13
#define VIB_DIM        6
#define VIB_IDX        13

#define TENSOR_ARENA   (60 * 1024)
#define SILENCE_THR    1e5f

#define WEIGHT_MEL     1.0f
#define WEIGHT_VIB     5.0f
#define WEIGHT_SUM     (MEL_DIM * WEIGHT_MEL + VIB_DIM * WEIGHT_VIB)  // 43.0

// ============================================================
// SECTION 2: MEL FILTERBANK
// ============================================================

struct MelBand { int lo, mid, hi; };

static const MelBand MEL_FB[MEL_BANDS] = {
    {  1, 11, 21 },
    { 11, 21, 32 },
    { 22, 32, 42 },
    { 33, 43, 53 },
    { 43, 54, 64 },
    { 54, 64, 76 },
    { 65, 76, 90 },
    { 77, 91,107 },
    { 91,108,128 },
    {108,128,152 },
    {129,153,181 },
    {153,181,215 },
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
// SECTION 3: GLOBALS
// ============================================================

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

static QueueHandle_t     feat_queue  = nullptr;
static SemaphoreHandle_t model_mutex = nullptr;

// TFLite - 1 model duy nhat
static uint8_t*                  tensor_arena = nullptr;
static tflite::MicroInterpreter* interpreter  = nullptr;
alignas(16) static uint8_t interp_buf[sizeof(tflite::MicroInterpreter)];
static tflite::MicroErrorReporter    micro_error_reporter;
static tflite::ErrorReporter*        error_reporter = &micro_error_reporter;
static tflite::MicroMutableOpResolver<4> resolver;
static bool resolver_ready = false;

// Stats
static uint32_t total_wins   = 0;
static uint32_t dropped      = 0;
static uint32_t silent_wins  = 0;
static uint32_t alarm_wins   = 0;
static uint32_t idle_wins    = 0;
static uint32_t consec_alarm = 0;  // so window ALARM lien tiep (debounce)
#define ALARM_DEBOUNCE  3          // phai co >= 3 window lien tiep moi bao loi

// Debug: print full IN/OUT vectors occasionally / on HIGH
#define DEBUG_PRINT_IO  1
#define DEBUG_IO_EVERY_N_WINS  20   // print mỗi N windows (khi không idle)
// Print raw int8 tensors so you can reproduce exactly on Kaggle
#define DEBUG_PRINT_TENSORS_INT8  1

// Weighted MAE weights
static const float FEAT_WEIGHTS[FEAT_DIM] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,   // mel
    5,5,5,5,5,5                   // vib
};

// ============================================================
// IDLE GATING (skip inference when vibration is very low)
// IMPORTANT: thresholds are tuned from your logs to separate idle vs running:
// - idle example:    rms_y~0.014, var_y~0.000153, var_z~0.000469
// - running example: rms_y~0.019, var_y~0.000337, var_z~0.001626
// ============================================================

static inline bool is_idle_vib(const float* raw) {
    const float rms_y = raw[15];
    const float var_y = raw[16];
    const float var_z = raw[18];

    // Conservative: require BOTH variances low to avoid skipping real running.
    const bool low_rms_y = (rms_y < 0.030f);
    const bool low_var_y = (var_y < 0.00025f);
    const bool low_var_z = (var_z < 0.00080f);
    return low_rms_y && low_var_y && low_var_z;
}

// ============================================================
// SECTION 4: ADXL345
// ============================================================

static void adxl_init() {
    Wire.beginTransmission(ADXL_ADDR);
    Wire.write(0x2D); Wire.write(0x08);
    Wire.endTransmission();
    Wire.beginTransmission(ADXL_ADDR);
    Wire.write(0x31); Wire.write(0x00);
    Wire.endTransmission();
    Wire.beginTransmission(ADXL_ADDR);
    Wire.write(0x2C); Wire.write(0x0B);
    Wire.endTransmission();
    uint8_t id = 0;
    Wire.beginTransmission(ADXL_ADDR);
    Wire.write(0x00);
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL_ADDR, 1);
    if (Wire.available()) id = Wire.read();
    if (id == 0xE5) Serial.println("[ADXL] OK (0xE5)");
    else            Serial.printf("[ADXL] WARN id=0x%02X\n", id);
}

static void adxl_read(int16_t& rx, int16_t& ry, int16_t& rz) {
    Wire.beginTransmission(ADXL_ADDR);
    Wire.write(0x32);
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL_ADDR, 6);
    uint8_t buf[6];
    for (int i = 0; i < 6 && Wire.available(); i++) buf[i] = Wire.read();
    rx = (int16_t)((buf[1] << 8) | buf[0]);
    ry = (int16_t)((buf[3] << 8) | buf[2]);
    rz = (int16_t)((buf[5] << 8) | buf[4]);
}

// ============================================================
// SECTION 5: I2S
// ============================================================

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
        .bck_io_num   = PIN_I2S_BCLK,
        .ws_io_num    = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = PIN_I2S_DIN,
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pins));
    Serial.println("[I2S] OK");
}

// ============================================================
// SECTION 6: SCALING (Global, không per-cluster)
// ============================================================

static void apply_scale(const float* raw, float* out) {
    // MEL: fixed scaling [-80,0] dB -> [0,1] (robust to environment)
    for (int j = 0; j < MEL_DIM; j++) {
        float v = (raw[j] + 80.0f) / 80.0f;
        out[j]  = fmaxf(0.0f, fminf(1.0f, v));
    }
    // VIB: clip outlier truoc khi scale (khop featureExtraction.py + training.py)
    // rms_z (vib[4]): cap 0.30g - pha vat manh tao spike khong phai anomaly
    // var_x (vib[1]), var_y (vib[3]), var_z (vib[5]): cap tai p99 training data
    float vib[VIB_DIM];
    for (int j = 0; j < VIB_DIM; j++) vib[j] = raw[VIB_IDX + j];
    vib[1] = fminf(vib[1], VIB_CLIP_VAR_X);   // var_x
    vib[3] = fminf(vib[3], VIB_CLIP_VAR_Y);   // var_y
    vib[4] = fminf(vib[4], VIB_CLIP_RMS_Z);   // rms_z
    vib[5] = fminf(vib[5], VIB_CLIP_VAR_Z);   // var_z
    // RobustScaler -> clip(-3,3)/6+0.5
    for (int j = 0; j < VIB_DIM; j++) {
        float v = (vib[j] - VIB_CENTER[j]) / VIB_SCALE[j];
        v = fmaxf(-3.0f, fminf(3.0f, v));
        out[VIB_IDX + j] = v / 6.0f + 0.5f;
    }
}

// ============================================================
// SECTION 7: TFLITE MODEL (1 model duy nhat)
// ============================================================

static void init_resolver() {
    if (resolver_ready) return;
    resolver.AddFullyConnected();
    resolver.AddLogistic();     // chi cho sigmoid output
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver_ready = true;
}

static bool load_model() {
    if (interpreter) {
        interpreter->~MicroInterpreter();
        interpreter = nullptr;
    }

    const tflite::Model* m = tflite::GetModel(model_single_ae_tflite);
    if (m->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[MODEL] Schema mismatch");
        return false;
    }

    // Model fingerprint (quick sanity check you flashed the right model_data.h)
    uint32_t crc = crc32_ieee((const uint8_t*)model_single_ae_tflite, (size_t)model_single_ae_tflite_len);
    Serial.printf("[MODEL] bytes=%u | head=%02x %02x %02x %02x | tail=%02x %02x %02x %02x\n",
        (unsigned)model_single_ae_tflite_len,
        (unsigned)model_single_ae_tflite[0], (unsigned)model_single_ae_tflite[1],
        (unsigned)model_single_ae_tflite[2], (unsigned)model_single_ae_tflite[3],
        (unsigned)model_single_ae_tflite[model_single_ae_tflite_len - 4],
        (unsigned)model_single_ae_tflite[model_single_ae_tflite_len - 3],
        (unsigned)model_single_ae_tflite[model_single_ae_tflite_len - 2],
        (unsigned)model_single_ae_tflite[model_single_ae_tflite_len - 1]);
    Serial.printf("[MODEL] crc32=%08lx\n", (unsigned long)crc);

    Serial.println("[MODEL] Creating interpreter...");
    interpreter = new(interp_buf)
        tflite::MicroInterpreter(m, resolver, tensor_arena, TENSOR_ARENA,
                                 error_reporter);
    Serial.println("[MODEL] Interpreter created");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[MODEL] AllocateTensors FAIL");
        interpreter->~MicroInterpreter();
        interpreter = nullptr;
        return false;
    }

    TfLiteTensor* inp = interpreter->input(0);
    TfLiteTensor* out = interpreter->output(0);
    Serial.printf("[MODEL] Loaded | in s=%.5f zp=%d | out s=%.5f zp=%d\n",
        inp->params.scale, inp->params.zero_point,
        out->params.scale, out->params.zero_point);
    Serial.printf("[MODEL] inp bytes=%d  out bytes=%d  (expected %d)\n",
        (int)inp->bytes, (int)out->bytes, FEAT_DIM);

    // Tensor shapes
    Serial.printf("[MODEL] inp dims:");
    for (int i = 0; i < inp->dims->size; i++) Serial.printf(" %d", inp->dims->data[i]);
    Serial.println();
    Serial.printf("[MODEL] out dims:");
    for (int i = 0; i < out->dims->size; i++) Serial.printf(" %d", out->dims->data[i]);
    Serial.println();
    Serial.flush();
    return true;
}

// ============================================================
// SECTION 8: QUANTIZATION HELPERS
// ============================================================

static int8_t quant(float v, float scale, int32_t zp) {
    return (int8_t)constrain((int32_t)roundf(v / scale) + zp, -128, 127);
}

static float dequant(int8_t v, float scale, int32_t zp) {
    return ((float)v - (float)zp) * scale;
}

// ============================================================
// SECTION 9: DSP TASK (Core 0)
// ============================================================

void dsp_task(void*) {
    static float hann[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));

    static float    overlap[FFT_SIZE];
    static float    fft_buf[FFT_SIZE * 2];
    static float    mel_frames[NUM_FRAMES][MEL_BANDS];
    memset(overlap, 0, sizeof(overlap));

    Serial.println("[DSP] Ready");

    while (true) {
        FeatureVec fv;
        float mel_max = 0.0f;
        bool  ok      = true;

        // ── AUDIO ──────────────────────────────────────────
        for (int fr = 0; fr < NUM_FRAMES && ok; fr++) {
            int new_idx = HOP_SIZE, chunks = 0;

            while (chunks < CHUNKS_PER_HOP) {
                int32_t raw[8]; size_t br;
                if (i2s_read(I2S_NUM_0, raw, sizeof(raw),
                             &br, pdMS_TO_TICKS(500)) != ESP_OK || br == 0) {
                    ok = false; break;
                }
                int n = (int)(br / sizeof(int32_t));
                for (int i = 0; i < n && new_idx < FFT_SIZE; i++, new_idx++) {
                    int16_t s = hpf((int16_t)(raw[i] >> 14));
                    overlap[new_idx]   = (float)s;
                }
                chunks++;
            }
            if (!ok) break;

            for (int i = 0; i < HOP_SIZE; i++) {
                fft_buf[i*2]     = overlap[i] * hann[i];
                fft_buf[i*2 + 1] = 0.0f;
            }
            for (int i = HOP_SIZE; i < FFT_SIZE; i++) {
                fft_buf[i*2]     = overlap[i] * hann[i];
                fft_buf[i*2 + 1] = 0.0f;
            }
            memmove(overlap, overlap + HOP_SIZE, HOP_SIZE * sizeof(float));
            memset(overlap + HOP_SIZE, 0, HOP_SIZE * sizeof(float));

            dsps_fft2r_fc32(fft_buf, FFT_SIZE);
            dsps_bit_rev_fc32(fft_buf, FFT_SIZE);
            mel_power(fft_buf, mel_frames[fr]);

            for (int b = 0; b < MEL_BANDS; b++)
                if (mel_frames[fr][b] > mel_max) mel_max = mel_frames[fr][b];
        }

        if (!ok) { memset(overlap, 0, sizeof(overlap)); continue; }

        // Silence detection
        if (mel_max < SILENCE_THR) {
            silent_wins++;
            Serial.printf("[SILENT] power=%.2e skip #%lu\n", mel_max, silent_wins);
            continue;
        }

        // power_to_db
        if (mel_max < 1e-10f) mel_max = 1e-10f;
        for (int b = 0; b < MEL_BANDS; b++) {
            float sum = 0.0f;
            for (int fr = 0; fr < NUM_FRAMES; fr++)
                sum += fmaxf(10.0f * log10f(mel_frames[fr][b] / mel_max), -80.0f);
            fv.raw[b] = sum / NUM_FRAMES;
        }

        // ── VIB ────────────────────────────────────────────
        {
            float sum[3]    = {0, 0, 0};
            float sq_sum[3] = {0, 0, 0};
            int16_t rx, ry, rz;
            for (int i = 0; i < VIB_SAMPLES; i++) {
                adxl_read(rx, ry, rz);
                float ax = rx / 256.0f, ay = ry / 256.0f, az = rz / 256.0f;
                sum[0] += ax;  sq_sum[0] += ax*ax;
                sum[1] += ay;  sq_sum[1] += ay*ay;
                sum[2] += az;  sq_sum[2] += az*az;
                vTaskDelay(1);
            }
            float n = (float)VIB_SAMPLES;
            for (int a = 0; a < 3; a++) {
                float mean = sum[a] / n;
                fv.raw[VIB_IDX + a*2]     = sqrtf(sq_sum[a] / n);
                fv.raw[VIB_IDX + a*2 + 1] = (sq_sum[a] / n) - mean*mean;
            }
        }

        // Log raw
        Serial.printf("[RAW] mel: %.1f %.1f %.1f ... %.1f | "
                      "rms: x=%.4f y=%.4f z=%.4f | "
                      "var: x=%.6f y=%.6f z=%.6f\n",
            fv.raw[0], fv.raw[1], fv.raw[2], fv.raw[MEL_DIM-1],
            fv.raw[13], fv.raw[15], fv.raw[17],
            fv.raw[14], fv.raw[16], fv.raw[18]);

        total_wins++;

        if (xQueueSend(feat_queue, &fv, 0) != pdTRUE) {
            dropped++;
            Serial.printf("[WARN] Queue full: %lu/%lu dropped\n", dropped, total_wins);
        }
    }
}

// ============================================================
// SECTION 10: AI TASK (Core 1)
// ============================================================

void ai_task(void*) {
    while (!feat_queue || !model_mutex) vTaskDelay(10);

    FeatureVec fv;

    while (true) {
        if (xQueueReceive(feat_queue, &fv, portMAX_DELAY) != pdTRUE) continue;

        // Idle gating: if vibration is very low, always OK and skip inference.
        if (is_idle_vib(fv.raw)) {
            idle_wins++;
            consec_alarm = 0;
            Serial.printf("[OK_IDLE] rms_y=%.4f var_y=%.6f var_z=%.6f | idle:%lu wins:%lu\n",
                fv.raw[15], fv.raw[16], fv.raw[18], idle_wins, total_wins);
            continue;
        }

        // Scale (global, không cần cluster assignment)
        apply_scale(fv.raw, fv.scaled);

        // Log scaled
        int clips = 0;
        for (int i = 0; i < FEAT_DIM; i++)
            if (fv.scaled[i] <= 0.001f || fv.scaled[i] >= 0.999f) clips++;
        if (clips > 2)
            Serial.printf("[WARN] Scale clip %d/%d\n", clips, FEAT_DIM);
        Serial.printf("[SCALED] mel:%.3f %.3f %.3f | vib:%.3f %.3f %.3f | clip=%d\n",
            fv.scaled[0], fv.scaled[1], fv.scaled[2],
            fv.scaled[13], fv.scaled[15], fv.scaled[17], clips);

        if (!interpreter) continue;

        // Quantize + Invoke
        // Dense model: input tensor shape (1, 19) - flat, khong phai (1, 19, 1)
        // Ghi thang vao inp->data.int8[0..18], copy sang xin[] truoc Invoke()
        xSemaphoreTake(model_mutex, portMAX_DELAY);
        TfLiteTensor* inp = interpreter->input(0);
        TfLiteTensor* out = interpreter->output(0);

        int8_t xin[FEAT_DIM];
        for (int i = 0; i < FEAT_DIM; i++) {
            int8_t q = quant(fv.scaled[i], inp->params.scale, inp->params.zero_point);
            inp->data.int8[i] = q;
            xin[i] = q;
        }

        int64_t t0 = esp_timer_get_time();
        TfLiteStatus s = interpreter->Invoke();
        uint32_t us = (uint32_t)(esp_timer_get_time() - t0);

        if (s != kTfLiteOk) {
            Serial.println("[AI] Invoke fail");
            xSemaphoreGive(model_mutex);
            continue;
        }

        // Copy output - Dense model output shape (1, 19), doc tu out->data.int8[0..18]
        int8_t yout[FEAT_DIM];
        for (int i = 0; i < FEAT_DIM; i++) yout[i] = out->data.int8[i];

        // Dequantize + MAE
        float mae_float = 0.0f;   // compare fv.scaled vs pred (float-domain)
        float mae_int8  = 0.0f;   // compare dequant(input_int8) vs pred (INT8-domain) - MUST match Kaggle verify
        float pred_f[FEAT_DIM];
        float xf_f[FEAT_DIM];
        for (int i = 0; i < FEAT_DIM; i++) {
            float pred = fmaxf(0.0f, fminf(1.0f,
                dequant(yout[i],
                        out->params.scale,
                        out->params.zero_point)));
            pred_f[i] = pred;

            // Float-domain MAE (useful for debugging only)
            mae_float += fabsf(fv.scaled[i] - pred) * FEAT_WEIGHTS[i];

            // INT8-domain MAE (this is what THRESHOLD is trained on)
            float xf = fmaxf(0.0f, fminf(1.0f,
                dequant(xin[i],
                        inp->params.scale,
                        inp->params.zero_point)));
            xf_f[i] = xf;
            mae_int8 += fabsf(xf - pred) * FEAT_WEIGHTS[i];
        }
        mae_float /= WEIGHT_SUM;
        mae_int8  /= WEIGHT_SUM;

        // Use INT8-domain MAE to compare with exported THRESHOLD
        bool over_thr = mae_int8 > THRESHOLD;

#if DEBUG_PRINT_IO
        // Print IN/OUT on first few wins, on HIGH/ALARM, and every N wins.
        if (total_wins <= 5 || over_thr || (total_wins % DEBUG_IO_EVERY_N_WINS == 0)) {
            Serial.printf("[OUT] pred: ");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf("%.3f ", pred_f[i]);
            Serial.println();
            Serial.printf("[IN]  inp:  ");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf("%.3f ", fv.scaled[i]);
            Serial.println();
            Serial.printf("[XF]  xint8:");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf("%.3f ", xf_f[i]);
            Serial.println();
#if DEBUG_PRINT_TENSORS_INT8
            Serial.printf("[XIN] int8:");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf(" %d", (int)xin[i]);
            Serial.println();
            Serial.printf("[YOUT]int8:");
            for (int i = 0; i < FEAT_DIM; i++) Serial.printf(" %d", (int)yout[i]);
            Serial.println();
#endif
            Serial.printf("[DBG] d_in_xf mel0..2: %.6f %.6f %.6f | vib(rms_x,rms_y,rms_z): %.6f %.6f %.6f\n",
                (double)(fv.scaled[0] - xf_f[0]),
                (double)(fv.scaled[1] - xf_f[1]),
                (double)(fv.scaled[2] - xf_f[2]),
                (double)(fv.scaled[13] - xf_f[13]),
                (double)(fv.scaled[15] - xf_f[15]),
                (double)(fv.scaled[17] - xf_f[17]));
        }
#endif

        // Debounce: chi bao ALARM khi co >= ALARM_DEBOUNCE windows lien tiep
        if (over_thr) {
            consec_alarm++;
        } else {
            consec_alarm = 0;
        }
        bool alarm = (consec_alarm >= ALARM_DEBOUNCE);
        if (over_thr && consec_alarm == 1) alarm_wins++;  // dem lan dau vuot thr

        // Print mae_int8 as MAE so it matches Kaggle's thr_int8.
        // Also show mae_float for debugging drift between float vs int8 input.
        Serial.printf("[%s] MAE:%.4f (f:%.4f) THR:%.4f consec:%lu t:%uus | wins:%lu\n",
            alarm ? "ALARM" : (over_thr ? "HIGH " : "OK   "),
            mae_int8, mae_float, THRESHOLD, consec_alarm, us, total_wins);

        // Release mutex after all reads/prints are done.
        xSemaphoreGive(model_mutex);
    }
}

// ============================================================
// SECTION 11: SETUP & LOOP
// ============================================================

void setup() {
    Serial.begin(921600);
    delay(500);
    Serial.println("=== BOOT v7.0 (Single AE) ===");

    WiFi.mode(WIFI_OFF);
    btStop();

    // Tensor arena
    tensor_arena = (uint8_t*)heap_caps_aligned_alloc(
        16, TENSOR_ARENA, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        Serial.println("[MEM] SPIRAM fail, trying internal...");
        tensor_arena = (uint8_t*)heap_caps_aligned_alloc(
            16, TENSOR_ARENA, MALLOC_CAP_INTERNAL);
    }
    if (!tensor_arena) {
        Serial.println("[ERROR] tensor arena alloc fail");
        while (1) vTaskDelay(1000);
    }
    memset(tensor_arena, 0, TENSOR_ARENA);
    Serial.printf("[MEM] Arena %dKB @ %p\n", TENSOR_ARENA/1024, tensor_arena);

    // FFT
    if (dsps_fft2r_init_fc32(NULL, FFT_SIZE) != ESP_OK) {
        Serial.println("[ERROR] FFT init fail");
        while (1) vTaskDelay(1000);
    }

    // Hardware
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    adxl_init();
    i2s_init();

    // TFLite
    tflite::InitializeTarget();
    init_resolver();
    if (!load_model()) {
        Serial.println("[ERROR] model load fail");
        while (1) vTaskDelay(1000);
    }

    // RTOS
    model_mutex = xSemaphoreCreateMutex();
    feat_queue  = xQueueCreate(5, sizeof(FeatureVec));
    if (!model_mutex || !feat_queue) {
        Serial.println("[ERROR] RTOS init fail");
        while (1) vTaskDelay(1000);
    }

    xTaskCreatePinnedToCore(dsp_task, "DSP", 40960, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(ai_task,  "AI",  16384, NULL,  5, NULL, 1);

    Serial.println("=== READY v7.0 ===");
}

void loop() {
    vTaskDelete(NULL);
}
