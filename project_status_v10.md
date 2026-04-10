# 📋 Trạng Thái Hiện Tại — Dự Án TinyML Anomaly Detection (v10.0)

> **Ngày cập nhật:** 10/04/2026  
> **Phiên bản Firmware:** 10.0 — True Parallel Dual-Core  
> **Trạng thái:** ✅ Hoạt động ổn định, sẵn sàng nâng cấp lên v11

---

## 1. Tổng Quan Dự Án

**Tên đề tài:** Hệ thống Nhúng TinyML với Autoencoder Không Giám sát cho Phát hiện Bất thường và Bảo trì Dự đoán Máy Giặt

**Tác giả:** Quách Ngọc Quang — Sinh viên Kỹ thuật Máy tính, UET-VNU  
**Người hướng dẫn:** TS. Nguyễn Kiêm Hùng

### 1.1 Mục tiêu
- Triển khai AI suy luận **trực tiếp trên vi điều khiển** (Edge AI / TinyML), không phụ thuộc cloud.
- Phát hiện bất thường dựa trên **lỗi tái tạo (MAE)** của mô hình Autoencoder lượng tử hóa INT8.
- Cảnh báo thời gian thực qua **MQTT** đến ứng dụng di động **Flutter**.

---

## 2. Kiến Trúc Phần Cứng

| Thành phần | Linh kiện | Giao tiếp | Thông số |
|---|---|---|---|
| **MCU** | Seeed XIAO ESP32-S3 | — | Dual-Core 240MHz, 8MB PSRAM, Wi-Fi |
| **Microphone** | INMP441 | I2S | WS=GPIO6, BCLK=GPIO43, DIN=GPIO44 |
| **Gia tốc kế** | ADXL345 | I2C | SDA=GPIO5, SCL=GPIO4, Addr=0x53, 200Hz ODR |

### 2.1 Sơ đồ kết nối

```
XIAO ESP32S3
┌─────────────────────┐
│  GPIO6  ──── WS     │── INMP441
│  GPIO43 ──── BCLK   │   (I2S Microphone)
│  GPIO44 ──── DIN    │
│                     │
│  GPIO5  ──── SDA    │── ADXL345
│  GPIO4  ──── SCL    │   (I2C Accelerometer)
└─────────────────────┘
```

---

## 3. Kiến Trúc Phần Mềm Firmware (v10.0)

### 3.1 Mô hình xử lý song song (True Parallel Dual-Core)

```
┌─────────────────────────────────────────────────────────────┐
│                    XIAO ESP32-S3                            │
│                                                             │
│  ┌───────────────────┐    ┌───────────────────────────────┐ │
│  │     CORE 0        │    │          CORE 1               │ │
│  │                   │    │                               │ │
│  │  audio_task       │    │  vib_task (priority=15)       │ │
│  │  (priority=10)    │    │  ADXL345 @ 1ms/sample         │ │
│  │                   │    │  1000 samples → RMS + Var     │ │
│  │  I2S Capture      │    │  ► vib_queue (float[6])       │ │
│  │  512-pt FFT       │    │                               │ │
│  │  13-band Mel      │    │  ai_task (priority=5)         │ │
│  │  30 frames        │    │  EventGroup AND-wait          │ │
│  │  ► audio_queue    │    │  Tri-State routing            │ │
│  │    (float[13])    │    │  TFLite Micro Inference       │ │
│  │                   │    │  MQTT Edge-triggered publish  │ │
│  └───────────────────┘    └───────────────────────────────┘ │
│                                                             │
│  Đồng bộ: EventGroup (AUDIO_READY_BIT | VIB_READY_BIT)     │
│  Bảo vệ mô hình: model_mutex (Semaphore)                   │
└─────────────────────────────────────────────────────────────┘
```

| Task | Core | Priority | Stack | Chức năng |
|---|---|---|---|---|
| `audio_task` | 0 | 10 | 40KB | I2S capture → 512-pt FFT → 13-band log-Mel → trung bình 30 frames |
| `vib_task` | 1 | 15 | 8KB | ADXL345 @ 1ms (vTaskDelayUntil) → RMS + Variance trên 3 trục |
| `ai_task` | 1 | 5 | 24KB | Đợi cả 2 nguồn (AND logic) → gộp 19-dim → scale → inference → MQTT |

### 3.2 Phân luồng Tri-State

Hệ thống **tự động nhận diện 3 chế độ vận hành** của máy giặt dựa trên mức phương sai trục Z:

```
Var_Z < 0.105845     →  MODE_GENTLE  (Giặt nhẹ / Thấm)
0.105845 ≤ Var_Z < 0.386260  →  MODE_STRONG  (Giặt chính)
Var_Z ≥ 0.386260     →  MODE_SPIN    (Vắt cao tốc)
```

Mỗi chế độ **sử dụng một mô hình Autoencoder riêng** được huấn luyện chuyên biệt, giúp giảm triệt để cảnh báo giả (False Positive).

### 3.3 Vector Đặc trưng (Feature Vector) — 19 chiều

| Index | Tên | Mô tả |
|---|---|---|
| 0–12 | `mel_0` … `mel_12` | 13 dải log-Mel Spectrogram (dB, trung bình 30 frames) |
| 13 | `rms_x` | RMS gia tốc trục X (trục trọng lực ~1g) |
| 14 | `var_x` | Phương sai trục X |
| 15 | `rms_y` | RMS gia tốc trục Y |
| 16 | `var_y` | Phương sai trục Y |
| 17 | `rms_z` | RMS gia tốc trục Z |
| 18 | `var_z` | Phương sai trục Z **(dùng để phân luồng Tri-State)** |

**Trọng số MAE:** Mel = 1.0, Vib = 5.0 → `WEIGHT_SUM = 13×1 + 6×5 = 43`

### 3.4 Tiền xử lý & Scaling

| Đặc trưng | Phương pháp | Chi tiết |
|---|---|---|
| Mel (0–12) | MinMax Scaling | `(raw + 80) / 80` → clip [0, 1] |
| Vib (13–18) | RobustScaler | `(raw - center) / scale` → clip [-3, 3] → `v/6 + 0.5` |

**Các tham số Scaler hiện tại (từ `model_data.h`):**

| Tham số | GENTLE | STRONG | SPIN |
|---|---|---|---|
| VIB_CENTER | [1.033, 0.000105, 0.0105, 0.000109, 0.235, 0.000935] | [1.036, 0.00110, 0.0410, 0.000978, 0.515, 0.213] | [1.037, 0.00110, 0.0454, 0.000978, 0.795, 0.580] |
| VIB_SCALE | [0.00319, 0.002, 0.0159, 0.002, 0.0636, 0.0262] | [0.00646, 0.002, 0.0692, 1.0, 0.101, 0.0972] | [0.00711, 1.0, 0.0378, 1.0, 0.152, 0.237] |
| VIB_CLIP (var) | var_x ≤ 0.00110 | var_y ≤ 0.000978 | var_z ≤ 0.851 |

---

## 4. Mô Hình AI

### 4.1 Kiến trúc Autoencoder

```
Input (19) → Dense(128, sigmoid) → Dense(64, sigmoid) → Dense(32, sigmoid)
           → Dense(64, sigmoid) → Dense(128, sigmoid) → Dense(19, sigmoid)
```

- **Loại:** Autoencoder Đối xứng (Symmetric)
- **Activation:** Sigmoid toàn bộ (phù hợp dữ liệu đã scale về [0, 1])
- **Bottleneck:** 32 neurons
- **Phương pháp huấn luyện:** Float32 Pre-training (200 epochs) → QAT INT8 Fine-tuning (80 epochs)
- **Loss function:** Weighted MAE (trọng số Mel=1, Vib=5)

### 4.2 Kích thước mô hình sau lượng tử hóa

| Mô hình | Kích thước (bytes) | Dung lượng Flash |
|---|---|---|
| `model_gentle_tflite` | 32,376 | ~31.6 KB |
| `model_strong_tflite` | 32,416 | ~31.7 KB |
| `model_spin_tflite` | 32,416 | ~31.7 KB |
| **Tổng cộng** | **97,208** | **~95 KB** |

### 4.3 Ngưỡng phát hiện bất thường (Mean + 3σ)

| Chế độ | Threshold (MAE) | Ghi chú |
|---|---|---|
| GENTLE | 0.0510 | Nhẹ nhất, nhạy nhất |
| STRONG | 0.0948 | Ngưỡng cao hơn do chung mode rung mạnh |
| SPIN | 0.0826 | Vắt cao tốc |

### 4.4 Cơ chế cảnh báo

- **Debounce:** `ALARM_DEBOUNCE = 6` — cần **6 cửa sổ liên tiếp** vượt ngưỡng mới kích hoạt ALARM.
- **Edge-triggered MQTT:** Chỉ gửi MQTT tại thời điểm **chuyển trạng thái** (OK → ALARM hoặc ALARM → OK), không gửi liên tục.
- **Grace Period:** `GRACE_WINDOWS = 0` (hiện tại tắt).

---

## 5. Bộ Dữ Liệu

### 5.1 Thống kê

| Thuộc tính | Giá trị |
|---|---|
| **Thư mục** | `dataset_v6/normal/` |
| **Số file mẫu** | 2,800 cặp (`.wav` + `.csv`) |
| **Mỗi file** | 5 giây → 5 cửa sổ (1s/window) |
| **Tổng cửa sổ huấn luyện** | 14,000 windows (trong `train_features_v6.csv`) |
| **Kích thước CSV** | 2.97 MB (14,001 dòng, bao gồm header) |
| **Sample Rate** | 8000 Hz (audio) / 1000 Hz (vibration) |

### 5.2 Pipeline thu thập dữ liệu

```
ESP32 (dataCollection firmware)
  │
  │  Binary packet (24 bytes): [0xAA][0xBB] + [3×int16 vib] + [8×int16 audio]
  │  Serial @ 921600 baud
  ▼
dataCollection.py (Python 3)
  │  Nhận packet, kiểm tra gravity, lưu .wav + .csv
  │  5 giây/file, kiểm tra trọng lực trục X (~1g)
  ▼
dataset_v6/normal/
  │  sample_0001.wav + sample_0001.csv
  ▼
featureExtraction.py
  │  WAV → FFT → 13-band Mel Spectrogram (match firmware FFT params)
  │  CSV → RMS + Variance trên 3 trục
  │  19-dim feature vector × 5 windows/file
  ▼
train_features_v6.csv (14,000 vectors)
```

---

## 6. Giao Thức Truyền Thông (MQTT)

### 6.1 Cấu hình hiện tại

| Thuộc tính | Giá trị |
|---|---|
| **Broker** | `broker.emqx.io` (public) |
| **Port** | 1883 (không mã hóa) |
| **Topic** | `tinyml/quang_wm_2026/status` |
| **Client ID (ESP32)** | `ESP32_WashingMachine` |
| **Protocol** | MQTT v3.1.1 |
| **QoS** | 0 (At Most Once) |

### 6.2 Định dạng Payload hiện tại (JSON)

```json
{
  "state": "ALARM",
  "mae": 0.1234,
  "is_alarm": true,
  "win": 156,
  "consec": 7
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `state` | string | `"ALARM"` hoặc `"OK"` |
| `mae` | float | Giá trị MAE tại thời điểm chuyển trạng thái |
| `is_alarm` | bool | Trạng thái cảnh báo |
| `win` | int | Số cửa sổ tổng cộng đã xử lý |
| `consec` | int | Số cửa sổ liên tiếp vượt ngưỡng |

> ⚠️ **Hạn chế:** Payload JSON hiện tại ~120 bytes. Không gửi `mode` (GENTLE/STRONG/SPIN) trong payload.

---

## 7. Ứng Dụng Di Động (Flutter)

### 7.1 Thông tin chung

| Thuộc tính | Giá trị |
|---|---|
| **Framework** | Flutter (Dart) |
| **File chính** | `tinyml_app/lib/main.dart` (1,262 dòng, 46 KB) |
| **Giao diện** | Dark theme (#0A0C12), Material Design 3 |
| **Tabs** | 3 tab: Giám sát / Lịch sử / Thống kê |

### 7.2 Dependencies (`pubspec.yaml`)

```yaml
dependencies:
  mqtt_client: ^10.0.0          # Kết nối MQTT
  intl: ^0.19.0                 # Định dạng ngày giờ
  firebase_core: ^3.0.0         # Firebase Core
  cloud_firestore: ^5.0.0       # Lưu lịch sử cảnh báo
  flutter_local_notifications: ^18.0.0  # Push notification
  fl_chart: ^1.2.0              # Biểu đồ MAE (mới thêm)
```

### 7.3 Chức năng chính

| Chức năng | Trạng thái | Chi tiết |
|---|---|---|
| Kết nối MQTT | ✅ Hoạt động | Auto-reconnect, heartbeat watchdog (60s) |
| Hiển thị trạng thái | ✅ Hoạt động | OK / ALARM / Waiting / Device Lost |
| Push Notification | ✅ Hoạt động | Thông báo khi phát hiện ALARM |
| Biểu đồ MAE | ✅ Hoạt động | `fl_chart`, 40 điểm gần nhất |
| Lịch sử sự kiện | ✅ Hoạt động | 30 sự kiện gần nhất |
| Lưu Firestore | ✅ Hoạt động | Lưu vào Cloud Firestore khi có sự kiện |
| Thống kê pha giặt | ✅ Hoạt động | Đếm GENTLE/STRONG/SPIN |
| Animation cảnh báo | ✅ Hoạt động | Glow animation khi ALARM |

### 7.4 Parse MQTT Payload (hiện tại)

```dart
final data = jsonDecode(payload) as Map<String, dynamic>;
final isAlarm = data['is_alarm'] as bool? ?? false;
final mae     = (data['mae'] ?? 0.0).toDouble();
final win     = data['win'] as int? ?? 0;
final consec  = data['consec'] as int? ?? 0;
final mode    = parseModeFromString(data['mode'] as String?);
```

> ⚠️ **Lưu ý:** Flutter đang parse trường `mode` nhưng firmware **không gửi** trường này → `mode` luôn là `unknown` phía Flutter.

---

## 8. Pipeline Huấn Luyện (`training.py`)

### 8.1 Quy trình

```
train_features_v6.csv
  │
  ▼
1. Phân chia Tri-State theo Var_Z
  │  GENTLE: var_z < 0.105845
  │  STRONG: 0.105845 ≤ var_z < 0.386260
  │  SPIN:   var_z ≥ 0.386260
  │
  ▼
2. Clip outlier (P99 cho var_x, var_y; max cho var_z)
  │
  ▼
3. RobustScaler (quantile_range=[10, 90]) cho Vib
  │  MinMax [-80, 0] → [0, 1] cho Mel
  │
  ▼
4. Data Augmentation (nếu < 10,000 mẫu → tile + noise)
  │
  ▼
5. Float32 Pre-training (200 epochs, Adam 1e-3, EarlyStopping=20)
  │
  ▼
6. QAT INT8 Fine-tuning (80 epochs, Adam 1e-4, EarlyStopping=15)
  │
  ▼
7. TFLite INT8 Export + Threshold tính toán (Mean + 3σ hoặc P99.5)
  │
  ▼
8. Tạo model_data.h (C-array + scaler params + thresholds)
```

### 8.2 Đầu ra

File `model_data.h` (~601 KB) chứa:
- 3 mô hình TFLite INT8 dạng C-array (`model_gentle_tflite[]`, `model_strong_tflite[]`, `model_spin_tflite[]`)
- 3 ngưỡng phát hiện (`THRESHOLD_GENTLE`, `THRESHOLD_STRONG`, `THRESHOLD_SPIN`)
- Tham số scaler cho từng chế độ (`VIB_CENTER_*`, `VIB_SCALE_*`)
- Giá trị clip phương sai (`VIB_CLIP_VAR_*`)

---

## 9. Cấu Trúc Thư Mục Dự Án

```
TinyML-Anomaly-Detection/
├── README.md                    # Mô tả dự án
├── project_status_v10.md        # Tài liệu trạng thái hiện tại (FILE NÀY)
│
├── dataCollection.py            # Script thu thập dữ liệu (Python)
│                                  Binary packet 921600 baud, lưu .wav + .csv
│
├── featureExtraction.py         # Script trích xuất đặc trưng (Python)
│                                  WAV → Mel, CSV → RMS/Var → 19-dim vector
│
├── training.py                  # Script huấn luyện mô hình (Python/Kaggle)
│                                  Tri-State AE, QAT INT8, xuất model_data.h
│
├── train_features_v6.csv        # Bộ đặc trưng huấn luyện (2.97 MB, 14,000 vectors)
│
├── dataset_v6/
│   └── normal/                  # 2,800 cặp file (.wav + .csv), mỗi file 5 giây
│       ├── sample_001.wav       # Audio PCM 16-bit, 8000Hz
│       ├── sample_001.csv       # Vibration 3-axis, 1000Hz
│       └── ...
│
├── firmware_v10/
│   ├── firmware_v10.ino         # Firmware chính (640 dòng, 25.6 KB)
│   └── model_data.h             # Header mô hình (8,139 dòng, 601 KB)
│
├── tinyml_app/                  # Ứng dụng Flutter
│   ├── pubspec.yaml             # Dependencies
│   └── lib/
│       ├── main.dart            # Mã nguồn chính (1,262 dòng, 46 KB)
│       └── firebase_options.dart
│
├── dataCollection/              # (Firmware thu thập dữ liệu - Arduino)
└── Slide/                       # Slide thuyết trình
```

---

## 10. Sử Dụng Tài Nguyên (ESP32-S3)

### 10.1 Bộ nhớ

| Loại | Sử dụng | Chi tiết |
|---|---|---|
| **Flash** | ~601 KB | `model_data.h` (3 mô hình hardcoded) |
| **PSRAM** | 120 KB | 3 × tensor_arena (40KB mỗi cái), aligned 16 bytes |
| **SRAM (Stack)** | ~73 KB | audio_task(40K) + vib_task(8K) + ai_task(24K) |
| **SRAM (Heap)** | ~2 KB | Queues (2×float[13] + 2×float[6]) + EventGroup + Mutex |

### 10.2 Hiệu năng

| Thông số | Giá trị |
|---|---|
| **Tốc độ suy luận** | ~200–500 µs/inference (INT8 trên PSRAM) |
| **Chu kỳ cửa sổ** | ~1 giây (30 frames × 256-sample hop @ 8kHz) |
| **Serial baud** | 921,600 bps |
| **Wi-Fi** | STA mode, kết nối 1 AP |

---

## 11. Hạn Chế Của Phiên Bản Hiện Tại (v10)

| # | Hạn chế | Mức độ | Mô tả |
|---|---|---|---|
| 1 | **Mô hình hardcoded** | 🔴 Nghiêm trọng | Model nhúng trong `model_data.h` → phải flash lại toàn bộ firmware khi cập nhật model |
| 2 | **MQTT payload verbose** | 🟡 Trung bình | JSON ~120 bytes, không gửi trường `mode` |
| 3 | **Không có Risk Score** | 🟡 Trung bình | Chỉ có binary OK/ALARM, không có mức độ nguy hiểm liên tục |
| 4 | **Không có OTA** | 🔴 Nghiêm trọng | Không thể cập nhật mô hình từ xa qua Wi-Fi |
| 5 | **Hardcoded WiFi** | 🟡 Trung bình | SSID/password cố định trong source code |
| 6 | **Public MQTT broker** | 🟡 Trung bình | Sử dụng broker.emqx.io không bảo mật |

---

## 12. Kế Hoạch Nâng Cấp v11 (Tham khảo `implementation_plan.md`)

### 12.1 Phase 1: MQTT Compact Protocol
- Thay JSON bằng pipe-delimited: `v1|ALARM|0.1234|SPIN|156|7` (~20 bytes)
- Flutter hỗ trợ cả 2 format (backward compatible)
- **Tiết kiệm ~83% bandwidth**

### 12.2 Phase 2: LittleFS + OTA Model Update
- Custom partition table với phân vùng LittleFS 960KB
- Lưu `.tflite` files riêng biệt trên filesystem
- Web Server (port 80) để upload model mới qua Wi-Fi
- Dual-slot rollback (`.tflite.bak`) chống bricking

### 12.3 Phase 3: Sigmoid Risk Score + Color-coded UI
- Chuyển từ binary (OK/ALARM) sang Risk Score 0–100%
- Sigmoid function: `risk = 100 / (1 + exp(-k * (mae/threshold - 1)))`, k=15
- UI Flutter: Gradient progress ring (Green → Yellow → Red)

---

*Tài liệu này là snapshot trạng thái dự án tại thời điểm trước khi nâng cấp lên v11. Phục vụ mục đích viết báo cáo đồ án tốt nghiệp.*
