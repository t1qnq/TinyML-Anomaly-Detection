# Báo cáo Tổng hợp: Nâng cấp Toàn diện Hệ thống TinyML Anomaly Detection (Version 11)

**Ngày hoàn thiện:** 19/04/2026
**Phiên bản:** v11.3 (Final Production-Ready)
**Mục tiêu:** Chuyển đổi từ một dự án thử nghiệm sang giải pháp giám sát máy giặt công nghiệp tập trung vào bảo mật, độ tin cậy và khả năng quan sát sâu.

---

## 1. Trí tuệ nhân tạo tập trung vào dữ liệu (Data-Centric TinyML)

Thay vì chỉ tối ưu cấu trúc mạng, v11 tập trung vào việc tạo ra một tập dữ liệu huấn luyện chất lượng cao và các ngưỡng cảnh báo có cơ sở thực chứng.

### 1.1. Kiến trúc Mô hình (AE Architecture)
*   **Loại model:** Symmetric Autoencoder (Kiến trúc đối xứng).
*   **Cấu trúc lớp:** 19 (Input) → 128 → 64 → 32 (Bottleneck) → 64 → 128 → 19 (Output).
*   **Bottleneck 32 chiều:** Ép mô hình học các đặc trưng tinh túy nhất của máy giặt, loại bỏ nhiễu rung động ngẫu nhiên.
*   **Lượng tử hóa:** INT8 QAT (Quantization-Aware Training) giúp nén mô hình nhưng vẫn giữ được độ chính xác gần như tương đương Float32.

### 1.2. Pipeline Tăng cường & Giả lập lỗi (Augmentation & Synthesis)
Để khắc phục tình trạng thiếu dữ liệu lỗi, v11 sử dụng cơ chế **Anomaly Synthesis**:
*   **Vibration Scaling (0.8x - 1.2x):** Mô phỏng máy chạy với các khối lượng quần áo khác nhau.
*   **Audio Volume Shifting (±dB):** Mô phỏng nhiễu môi trường và tiếng ồn motor theo thời gian.
*   **Hard-Test Synthesis:** Tự động sinh ra các đoạn dữ liệu "văng lồng" (Z-axis 4x) và "lệch tâm" (X/Y bias) để kiểm chứng độ nhạy của Threshold.

### 1.3. Chỉ số Hiệu năng (Metrics)
| Chế độ | Threshold (MAE) | **F1-Score** | **AUC (Separation)** |
| :--- | :--- | :--- | :--- |
| **GENTLE** | 0.0557 | **0.9898** | **0.9989** |
| **STRONG** | 0.0940 | **0.9892** | **0.9837** |
| **SPIN** | 0.0901 | **0.9967** | **0.9993** |

---

## 2. Hạ tầng IoT & Bảo mật Secure Cloud

Chuyển đổi hoàn toàn phương thức giao tiếp để đáp ứng tiêu chuẩn triển khai thực tế.

### 2.1. Bảo mật đa lớp (Security Layers)
*   **Private Broker:** Di chuyển từ broker công cộng sang **HiveMQ Cloud**, cô lập luồng dữ liệu lỗi/trạng thái.
*   **Xác thực Authentication:** Triển khai Username/Password cho cả thiết bị và ứng dụng, loại bỏ rủi ro bị "nghe lén" dữ liệu.
*   **Encryption TLS/SSL:** Toàn bộ dữ liệu được mã hóa qua Port 8883, đảm bảo tính toàn vẹn của tín hiệu cảnh báo.

### 2.2. Chuẩn hóa Tần suất (1Hz Telemetry)
*   **Vấn đề cũ:** Thiết bị gửi dữ liệu không đều dẫn đến "Sampling Bias" (App luôn đỏ lòm dù máy đang ổn).
*   **Giải pháp:** Cưỡng bức gửi gói tin định kỳ chuẩn **1 giây/lần (1Hz)**.
*   **Lợi ích:** App hiển thị chính xác dòng thời gian thực tế, cho thấy rõ ranh giới giữa các đoạn máy chạy bình thường và bất thường.

---

## 3. Logic Cảnh báo & Quan sát (App Implementation)

Nâng cấp bộ não của ứng dụng Flutter để đưa ra quyết định thông minh hơn.

### 3.1. Thuật toán Cửa sổ trượt (Sliding Window Alarm)
Để triệt tiêu các cảnh báo giả do nhiễu nhất thời, chúng tôi triển khai cơ chế lọc cửa sổ:
*   **Kích hoạt ALARM:** Khi phát hiện **5/10 gói tin** liên tiếp ở mức HIGH.
*   **Phục hồi OK:** Khi nhận được **9/10 gói tin** liên tiếp ở mức OK.
*   **Kết quả:** Hệ thống chỉ báo động khi có dấu hiệu lỗi duy trì bền bỉ, mang lại sự tin tưởng cho người vận hành.

### 3.2. Mở rộng khả năng quan sát (Observability)
*   **Timeline Expansion:** Tăng dung lượng hiển thị từ 100 lên **500 phân đoạn lịch sử**.
*   **Grid Layout 10x50:** Thiết kế lưới mới giúp theo dõi chu kỳ 500 giây (gần 10 phút) trên một màn hình duy nhất một cách trực quan.
*   **Heartbeat Monitor:** Tự động báo trạng thái "DEVICE LOST" nếu không nhận được tín hiệu sau **60 giây**, giúp phát hiện ngay lập tức tình trạng mất mạng hoặc hỏng thiết bị.
*   **Firestore & Notifications:** Tự động lưu vết vĩnh viễn các sự kiện Alarm vào Firebase và phát thông báo đẩy (Local Notifications) ngay cả khi ứng dụng đang chạy nền.

---

## 4. Đặc tả thông số kỹ thuật (Technical Reference for Slides)

Dành cho trình bày slide thuyết trình đồ án:

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **Kiến trúc** | Symmetric Autoencoder | Nén 19 chiều đầu vào |
| **Tham số** | **25,779 parameters** | Siêu nhẹ cho MCU |
| **Inference Time** | **<500 µs** | Đáp ứng thời gian thực cực nhanh |
| **Dung lượng Flash** | **~95 KB** | Chiếm 2% Flash ESP32-S3 |
| **Tần suất truyền** | **1.0 Hz** | Chuẩn hóa dòng dữ liệu |
| **Bảo mật** | TLS/SSL (Port 8883) | Mật mã hóa đầu cuối |
| **Uptime History** | **500 segments** | Theo dõi chu kỳ ~10 phút |

---

## Kết luận
Version 11 đánh dấu sự hoàn thiện về cả **Chiều sâu (Mô hình AI)** và **Chiều rộng (Hạ tầng Security)**. Hệ thống hiện tại không chỉ là một dự án TinyML đơn thuần mà đã tiệm cận một sản phẩm IoT thực tế, có khả năng vận hành ổn định trong môi trường công nghiệp có nhiều nhiễu.
