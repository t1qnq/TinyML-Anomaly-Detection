# TinyML Embedded System with Unsupervised Autoencoder for Anomaly Detection and Predictive Maintenance of Washing Machines

![Hardware](https://img.shields.io/badge/Hardware-XIAO_ESP32S3-blue?style=flat-square&logo=espressif)
![AI](https://img.shields.io/badge/AI-TFLite_Micro-orange?style=flat-square&logo=tensorflow)
![App](https://img.shields.io/badge/App-Flutter-02569B?style=flat-square&logo=flutter)
![Python](https://img.shields.io/badge/Script-Python_3.8+-3776AB?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/Firmware-C++-00599C?style=flat-square&logo=c%2B%2B)

Hệ thống Edge AI (TinyML) giám sát tình trạng và phát hiện bất thường cho máy giặt theo thời gian thực. Dự án kết hợp xử lý song song trên kiến trúc Dual-Core của vi điều khiển XIAO ESP32S3, ứng dụng mô hình Autoencoder không giám sát được lượng tử hóa (QAT INT8) và giao diện giám sát di động đa nền tảng.

## 📑 Mục lục
- [Tổng quan dự án](#tổng-quan-dự-án)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn triển khai](#hướng-dẫn-triển-khai)
- [Tác giả & Lời cảm ơn](#tác-giả--lời-cảm-ơn)

---

## 💡 Tổng quan dự án

Bảo trì dự đoán (Predictive Maintenance) giúp chẩn đoán hỏng hóc trước khi chúng xảy ra. Thay vì dựa vào nền tảng cloud tốn kém và có độ trễ cao, dự án này đẩy trí tuệ nhân tạo xuống sát vùng biên (Edge AI):
- **Phân luồng Tri-State:** Thuật toán tự động nhận diện 3 chế độ vật lý của máy giặt (Nhẹ, Mạnh, Vắt) dựa trên phương sai trục Z (Z-variance) để gọi đúng mô hình AI tương ứng, giúp loại bỏ triệt để cảnh báo giả (False Positives).
- **True Parallel Dual-Core:** Tối ưu hóa RTOS trên kiến trúc lõi kép. Lõi 0 chuyên trách xử lý tín hiệu âm thanh I2S và biến đổi FFT. Lõi 1 ưu tiên định tuyến I2C (1ms/sample) và chạy suy luận mạng nơ-ron mà không gây nghẽn phần cứng.
- **Tối ưu hóa tài nguyên:** Ứng dụng Quantization-Aware Training (QAT INT8) giảm dung lượng mô hình, tối ưu hóa băng thông bộ nhớ PSRAM trên vi điều khiển.

---

## ⚙️ Kiến trúc hệ thống

Dòng chảy dữ liệu (Data Pipeline) được thiết kế khép kín và tối ưu độ trễ:
1. **Thu nhận tín hiệu:** Âm thanh qua microphone đa hướng INMP441 (I2S) và gia tốc 3 trục qua cảm biến ADXL345 (I2C).
2. **Tiền xử lý (DSP):** Trích xuất 13 dải Mel-spectrograms và 6 đặc trưng RMS/Variance cho mỗi cửa sổ trượt.
3. **Suy luận (Inference):** TFLite Micro tính toán Mean Absolute Error (MAE) trực tiếp trên XIAO ESP32S3.
4. **Cảnh báo (Telemetry):** Gửi các gói tin JSON chứa trạng thái qua giao thức MQTT đến ứng dụng Flutter để hiển thị đồ thị, phân tích thống kê và thông báo Push.

---

## 📂 Cấu trúc thư mục

```text
📦 Washing-Machine-TinyML
 ┣ 📂 firmware
 ┃ ┣ 📜 dataCollection.ino       # C++: Firmware thu thập dữ liệu thô (Raw stream) qua Serial
 ┃ ┗ 📜 firmware_v10.ino         # C++: Firmware Edge AI chính thức (Dual-Core, TFLite)
 ┣ 📂 scripts
 ┃ ┣ 📜 dataCollection.py        # Python: Kịch bản thu nhận dữ liệu và lưu định dạng .wav, .csv
 ┃ ┣ 📜 featureExtraction.py     # Python: Trích xuất đặc trưng không gian và thời gian
 ┃ ┗ 📜 training.py              # Python: Huấn luyện Autoencoder, lượng tử hóa và xuất C-array
 ┗ 📂 mobile_app
   ┗ 📜 main.dart                # Dart/Flutter: Mã nguồn ứng dụng giám sát thời gian thực
```

---

## 🛠 Yêu cầu hệ thống

### Phần cứng (Hardware)
- **MCU:** Seeed Studio XIAO ESP32S3
- **Audio Sensor:** INMP441 (Giao tiếp I2S: WS=6, BCLK=43, DIN=44)
- **Vibration Sensor:** ADXL345 (Giao tiếp I2C: SDA=5, SCL=4)

### Phần mềm (Software/Dependencies)
- **Edge Environment:** Arduino IDE (với gói hỗ trợ thư viện `TensorFlowLite_ESP32`, `dsps_fft`).
- **AI Pipeline:** Python 3.8+ (Cài đặt các gói: `tensorflow`, `numpy`, `pandas`, `scikit-learn`).
- **Mobile Dashboard:** Flutter SDK, tích hợp Firebase Core / Cloud Firestore.

---

## 🚀 Hướng dẫn triển khai

### Bước 1: Thu thập bộ dữ liệu (Data Collection)
1. Nạp `dataCollection.ino` vào XIAO ESP32S3.
2. Gắn cố định thiết bị vào thành máy giặt (lưu ý căn chỉnh trục X hướng xuống đất để đo trọng lực chuẩn ~1g).
3. Mở terminal trên máy tính và chạy script thu thập liên tục:
   ```bash
   python scripts/dataCollection.py --port COM_PORT
   ```

### Bước 2: Huấn luyện mô hình (Training Pipeline)
1. Trích xuất đặc trưng (Feature Extraction) từ các tệp `.wav` và `.csv`:
   ```bash
   python scripts/featureExtraction.py --data_dir dataset_v6/normal --out train_features_v6.csv
   ```
2. Tiến hành huấn luyện và lượng tử hóa mô hình (khuyên dùng môi trường Kaggle/Colab):
   ```bash
   python scripts/training.py
   ```
   *Kết quả đầu ra sẽ tự động sinh ra tệp `model_data.h` chứa trọng số mô hình INT8 và các ngưỡng an toàn.*

### Bước 3: Triển khai lên thiết bị (Edge Deployment)
1. Di chuyển tệp `model_data.h` vào chung thư mục với `firmware_v10.ino`.
2. Thay đổi cấu hình mạng WLAN (`ssid`, `password`) và thông tin địa chỉ MQTT broker trong mã nguồn C++.
3. Build và nạp firmware hoàn chỉnh vào XIAO ESP32S3.

### Bước 4: Khởi chạy ứng dụng giám sát (Mobile Dashboard)
1. Di chuyển vào thư mục ứng dụng Flutter và cài đặt các thư viện phụ thuộc:
   ```bash
   flutter pub get
   ```
2. Chạy ứng dụng trên thiết bị di động (Android/iOS) hoặc máy ảo:
   ```bash
   flutter run
   ```

---

## 👨‍💻 Tác giả & Lời cảm ơn

**Tác giả:** Quách Ngọc Quang
*Sinh viên năm 4 chuyên ngành Kỹ thuật Máy tính*
*Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (UET - VNU)*

**Lời cảm ơn:** Đồ án tốt nghiệp này được thực hiện và hoàn thiện dưới sự hướng dẫn tận tình của TS. Nguyễn Kiêm Hùng. Cảm ơn thầy đã hỗ trợ và định hướng trong suốt quá trình nghiên cứu và phát triển dự án.
