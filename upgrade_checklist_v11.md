# 📋 Kế Hoạch Nâng Cấp TinyML Anomaly Detection (v11 - The Data-Centric Update)

Dựa trên định hướng mới: **Tập trung vào Dữ liệu thay vì Hệ thống**, version 11 sẽ nhắm trực tiếp vào việc nâng cao chất lượng mô hình AI bằng cách tái sử dụng và tăng cường (Augment) dữ liệu tĩnh hiện có (từ `train_features_v6.csv`), cả cho việc Huấn luyện lẫn việc Test.

---

## 🚀 Giai đoạn 1: Nâng cấp Data Augmentation (Trong `training.py`)
Mở rộng phần Data Augmentation (hiện tại mới chỉ có cộng thêm nhiễu Gaussian tĩnh) thành một pipeline sinh dữ liệu động và đa dạng.

- [x] **1.1. Magnitude & Volume Scaling (Giả lập tải trọng & Tiếng ồn)**
  - [x] **Vibration (Rung lắc):** Nhân các biến RMS, VAR với tỷ lệ ngẫu nhiên (0.8x tới 1.2x) mô phỏng máy chạy lúc ít đồ và lúc nhiều đồ nặng.
  - [x] **Audio (Âm thanh):** Tịnh tiến (cộng/trừ) một lượng ngẫu nhiên (VD: ±2 đến ±5 dB) vào 13 dải Mel (`mfe_`) để giả lập mức âm lượng motor khác nhau hoặc do môi trường phòng giặt tĩnh/ồn hơn.
- [x] **1.2. Feature Jittering Đa Trục**
  - [x] Thêm nhiễu biên độ khác nhau cho từng trục X, Y, Z thay vì một mức nhiễu chung (noise_level). Đặc biệt là trục Z (phản ánh độ nảy của lồng giặt).
- [x] **1.3. Áp dụng Augmentation ngay trước khi Scaler**
  - [x] Đảm bảo việc sinh dữ liệu giả lập được áp dụng vào dữ liệu RAW trước khi đi qua hàm `RobustScaler`, để Scaler học được cả độ lệch của dữ liệu mới.

---

## 🧪 Giai đoạn 2: Xây Dựng Tập "Hard-Test" Bằng Dữ Liệu Cũ
Biến đổi dữ liệu bình thường (Normal) thành dữ liệu bất thường (Anomaly) để làm bài test cực khó cho AI.

- [x] **2.1. Tách Data chuẩn**
  - [x] Trích xuất riêng biệt 15-20% dữ liệu gốc để làm tập Kiểm thử (Hold-out Test Set) và giữ sạch, không cho qua vòng Train Augmentation.
- [x] **2.2. Sinh Anomaly giả lập (Synthesize Anomalies)**
  - [x] **Giả lập văng lồng:** Lấy một vài mẫu bình thường, cố tình nhân giá trị trục Z (`var_z`, `rms_z`) lên x3 lần.
  - [x] **Giả lập chân đế kênh:** Cộng thêm một giá trị hằng số cố định vào `rms_x` / `rms_y` để giả lập việc trọng tâm máy bị lệch.
  - [x] **Giả lập ồn môi trường:** Nâng cường độ `mfe_` (Mel Features) lên sát mức 0 dB để mô phỏng tiếng máy bơm nước hỏng hoặc có vật cọ xát.
- [x] **2.3. Hỗn hợp nhãn dán (Label Mixing)**
  - [x] Gắn nhãn `Normal (0)` cho các data gốc, và `Anomaly (1)` cho các data vừa bị "phá" ở bước 2.2.

---

## 📊 Giai đoạn 3: Tinh Chỉnh Threshold Dựa Trên Hard-Test
Thay đổi tư duy thiết lập Ngưỡng cảnh báo rủi ro (Risk Threshold).

- [x] **3.1. Chạy Evaluation trên Tập Hard-Test**
  - [x] Truyền tập dữ liệu Gốc và dữ liệu Giả lập (Anomaly) đi qua mô hình TFlite (INT8). Tính toán và thu thập các điểm số MAE tương ứng cho hai nhóm.
- [x] **3.2. Không Cảm Tính - Dùng Thuật Toán Quyết Định**
  - [x] Vẽ biểu đồ đè (Overlap Histogram) giữa phân phối lỗi MAE của hàng Ngon (Normal) và MAE hàng Lỗi (Anomaly).
  - [x] Tìm chính xác giao điểm (Intersection Point) để set Threshold, hoặc tính ROC/AUC để tìm ngưỡng tối ưu (Best F1-Score) thay vì thuật toán tĩnh `Mean + 3 Sigma` hiện tại.
  - [x] In thông số `Threshold` mới vào `model_data.h`.


---

- [x] **4.1. Dọn dẹp mã nguồn**
  - [x] Xóa bỏ các logic dư thừa trong `training.py`, gộp các cell export.
- [x] **4.2. Ghi chú hệ số Augmentation**
  - [x] Ghi chú chi tiết lại các tỷ lệ Scaling và hệ số Augment trong `training.py` và báo cáo.
- [x] **4.3. Tích hợp Firmware**
  - [x] Đồng bộ hóa file `model_data.h` sang thư mục `firmware_v10/`.
  - [x] Kiểm tra tên biến `THRESHOLD_GENTLE` ... khớp với logic của file `.ino`.

