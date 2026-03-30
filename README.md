# TinyML Embedded System with Unsupervised Autoencoder for Anomaly Detection and Predictive Maintenance of Washing Machines

![ESP32](https://img.shields.io/badge/Hardware-ESP32_S3-blue?style=flat-square&logo=espressif)
![TensorFlow Lite Micro](https://img.shields.io/badge/AI-TFLite_Micro-orange?style=flat-square&logo=tensorflow)
![Flutter](https://img.shields.io/badge/App-Flutter-02569B?style=flat-square&logo=flutter)
![Python](https://img.shields.io/badge/Script-Python_3.8+-3776AB?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/Firmware-C++-00599C?style=flat-square&logo=c%2B%2B)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Hệ thống Edge AI (TinyML) giám sát tình trạng và phát hiện bất thường cho máy giặt theo thời gian thực. Dự án kết hợp xử lý song song trên kiến trúc Dual-Core của vi điều khiển ESP32-S3, ứng dụng mô hình Autoencoder không giám sát được lượng tử hóa (QAT INT8) và giao diện giám sát di động đa nền tảng.

## 📑 Mục lục
- [Tổng quan dự án](#tổng-quan-dự-án)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn triển khai](#hướng-dẫn-triển-khai)
- [Tác giả](#tác-giả)
- [Giấy phép](#giấy-phép)

---

## 💡 Tổng quan dự án

Bảo trì dự đoán (Predictive Maintenance) giúp chẩn đoán hỏng hóc trước khi chúng xảy ra. Thay vì dựa vào cloud tốn kém và có độ trễ cao, dự án này đẩy trí tuệ nhân tạo xuống sát vùng biên (Edge AI):
- **Phân luồng Tri-State:** Thuật toán tự động nhận diện 3 chế độ vật lý của máy giặt (Nhẹ, Mạnh, Vắt) dựa trên phương sai trục Z (Z-variance) để gọi đúng mô hình AI tương ứng, giúp loại bỏ triệt để cảnh báo giả (False Positives).
- **True Parallel Dual-Core:** Tối ưu hóa RTOS. Lõi 0 chuyên trách xử lý tín hiệu âm thanh I2S và biến đổi FFT. Lõi 1 ưu tiên định tuyến I2C (1ms/sample) và chạy suy luận mạng nơ-ron.
- **Tối ưu hóa tài nguyên:** Ứng dụng Quantization-Aware Training (QAT INT8) giảm dung lượng mô hình, tối ưu hóa băng thông bộ nhớ PSRAM trên ESP32.

---

## ⚙️ Kiến trúc hệ thống

Dòng chảy dữ liệu (Data Pipeline) được thực hiện khép kín:
1. **Thu nhận tín hiệu:** Âm thanh (INMP441 - I2S) & Gia tốc 3 trục (ADXL345 - I2C).
2. **Tiền xử lý (DSP):** Trích xuất 13 dải Mel-spectrograms và 6 đặc trưng RMS/Variance.
3. **Suy luận (Inference):** TFLite Micro tính toán Mean Absolute Error (MAE) trực tiếp trên ESP32-S3.
4. **Cảnh báo (Telemetry):** Gửi gói tin qua giao thức MQTT đến ứng dụng Flutter để hiển thị đồ thị và thông báo Push.

---

## 📂 Cấu trúc thư mục

```text
📦 Washing-Machine-TinyML
 ┣ 📂 firmware
 ┃ ┣ 📜 dataCollection.ino       # Thu thập dữ liệu thô (Raw stream) qua Serial
 ┃ ┗ 📜 firmware_v10.ino         # Firmware Edge AI chính thức (Deploy)
 ┣ 📂 scripts
 ┃ ┣ 📜 dataCollection.py        # Kịch bản thu nhận dữ liệu và lưu chuẩn .wav, .csv
 ┃ ┣ 📜 featureExtraction.py     # Trích xuất đặc trưng không gian và thời gian
 ┃ ┗ 📜 training.py              # Huấn luyện Autoencoder, lượng tử hóa và xuất C-array
 ┗ 📂 mobile_app
   ┗ 📜 main.dart                # Mã nguồn ứng dụng giám sát Flutter
