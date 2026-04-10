# 📋 Kế Hoạch Nâng Cấp TinyML Anomaly Detection (v10 → v11)

Đây là danh sách chi tiết các công việc cần thực hiện để nâng cấp dự án. Bạn có thể đánh dấu `[x]` vào các ô tương ứng khi hoàn thành.

---

## 🚀 Giai đoạn 1: Tối ưu hoá Protocol MQTT (Phase 1)
Tiết kiệm ~83% bandwidth bằng cách đổi định dạng MQTT từ JSON sang Pipe-delimited.

- [ ] **1.1. Cập nhật Firmware (`firmware_v10.ino`)**
  - [ ] Xoá thư viện `ArduinoJson` (nếu không còn dùng ở đâu khác).
  - [ ] Thay thế lệnh build JSON bằng `sprintf` hoặc `#String`, định dạng: `v1|<state>|<mae>|<mode>|<win>|<consec>`.
  - [ ] Chú ý ánh xạ enum `AiMode` sang String (`GENTLE`, `STRONG`, `SPIN`).
- [ ] **1.2. Cập nhật Ứng dụng Flutter (`tinyml_app/lib/main.dart`)**
  - [ ] Sửa hàm `_parseData` để hỗ trợ cả 2 định dạng (JSON cũ và chuỗi `v1|` mới).
  - [ ] Parse chuỗi bằng `split('|')` và ép kiểu dữ liệu tương ứng.
  - [ ] Kiểm thử kết nối MQTT nhận dữ liệu thành công.

---

## 🧠 Giai đoạn 2: Cập nhật Training Pipeline (`training.py`)
Tách rời mô hình AI khỏi source code firmware.

- [ ] **2.1. Xuất file mô hình (`.tflite`)**
  - [ ] Cập nhật hàm `export_int8_and_verify` để lưu trực tiếp ra các file: `gentle.tflite`, `strong.tflite`, `spin.tflite`.
  - [ ] Loại bỏ script sinh file C-array khổng lồ (`model_data.h`).
- [ ] **2.2. Xuất file cấu hình (`config.json`)**
  - [ ] Viết đoạn code export các tham số: Threshold (MAE), Scaler Center/Scale, Clip Var sang một file `config.json` hoặc lưu tạm dưới header gọn nhẹ hơn (tùy vào cách load tham số v11).

---

## 💾 Giai đoạn 3: LittleFS, OTA & Quản lý Bộ Nhớ (Phase 2)
Cho phép cập nhật mô hình từ xa không cần flash lại firmware.

- [ ] **3.1. Phân vùng bộ nhớ (Partition Table)**
  - [ ] Tạo file `partitions.csv` tuỳ chỉnh.
  - [ ] Dành 960KB (hoặc ~1MB) cho phân vùng `littlefs`.
- [ ] **3.2. Tích hợp LittleFS trong Firmware**
  - [ ] Thêm thư viện `LITTLEFS`.
  - [ ] Khởi tạo `LittleFS.begin(true)` trong `setup()`.
- [ ] **3.3. Tích hợp Web Server để Upload OTA**
  - [ ] Cài đặt thư viện `WebServer`.
  - [ ] Viết hàm POST (port 80) để nhận file `.tflite` từ Web/Client.
  - [ ] Xử lý lưu file mới và tạo backup (`.tflite.bak`) để phòng ngừa sự cố mất điện lúc ghi.
- [ ] **3.4. Load mô hình vào PSRAM (`heap_caps_aligned_alloc`)**
  - [ ] Đọc file từ LittleFS thay vì mảng C.
  - [ ] Copy dữ liệu mô hình vào PSRAM (Memory-mapped hoặc load thẳng vào array cấp phát trong PSRAM).

---

## 🎨 Giai đoạn 4: UI/UX Mới trên Flutter (Phase 3)
Thể hiện mức độ rủi ro trực quan hơn thay vì chỉ báo Đỏ/Xanh cứng nhắc.

- [ ] **4.1. Thuật toán Risk Score (Hàm Sigmoid)**
  - [ ] Triển khai hàm tính % rủi ro trong Flutter: `Risk = 100 / (1 + e^(-k * (MAE/Threshold - 1)))`.
  - [ ] Chỉnh tham số `k = 15` để đường cong trơn tru.
- [ ] **4.2. Cập nhật Giao diện (Progress Ring / Gauge)**
  - [ ] Tạo custom painter hoặc dùng package vẽ vòng cung gauge.
  - [ ] Chuyển màu mượt mà theo Risk Score: **Xanh lá (0-30%) → Vàng (31-70%) → Đỏ (71-100%)**.
  - [ ] Hiển thị thông số `win`, con số Risk % ngay giữa vòng tròn.

---

## 🧪 Giai đoạn 5: Testing & Nghiệm thu
- [ ] **5.1. Test Unit & Integration**
  - [ ] Thử upload một mô hình rác coi ESP32 có rớt không (bắt lỗi mismatch schema).
  - [ ] Thử ngắt điện giữa chừng lúc đang OTA → Khởi động lên phải lấy lại được bản `.bak`.
- [ ] **5.2. Test Performance**
  - [ ] Test độ trễ suy luận AI sau khi load từ LittleFS (đảm bảo không bị nghẽn RAM/Flash cache).
- [ ] **5.3. Final Code Cleanup**
  - [ ] Xóa các biến/hàm obsolete từ bản `v10.0`.
  - [ ] Cập nhật lại số hiệu phiên bản thành `v11.0` trên Serial log.
