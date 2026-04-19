# %% [markdown]
# # TinyML Tri-State AE QAT - Ultimate Visualization Edition
# **Máy giặt - Edge AI Anomaly Detection (ESP32S3)**
# Kiến trúc: Gộp IDLE, Phân luồng Tri-State (0.001 & 0.010), Data Augmentation & QAT INT8.

# %%
# ==========================================
# CELL 1: IMPORTS & CONFIG
# ==========================================
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
# import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Style biểu đồ Kaggle
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

CSV_PATH = 'train_features_v6.csv'
MEL_DIM, VIB_DIM, FEAT_DIM = 13, 6, 19
WEIGHTS = np.array([1]*MEL_DIM + [5]*VIB_DIM, dtype=np.float32)
W_SUM = float(WEIGHTS.sum())

# NGƯỠNG CHIA VÙNG VẬT LÝ (Thực chứng - Đã sửa lại chuẩn 0.001 và 0.010)
VAR_Z_THR1 = 0.105845   # Gentle | Strong
VAR_Z_THR2 = 0.386260   # Strong | Spin

MEL_DB_MIN, MEL_DB_MAX = -80.0, 0.0
MEL_DB_RANGE = MEL_DB_MAX - MEL_DB_MIN

# %%
# ==========================================
# CELL 2: LOAD DATA & TRỰC QUAN HÓA TRI-STATE
# ==========================================
print("--- Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.values.astype(np.float32)

VIB_CLIP = {
    14: float(np.percentile(X_raw[:, 14], 99)),
    16: float(np.percentile(X_raw[:, 16], 99)),
    18: float(np.max(X_raw[:, 18])) # Thả rông trục Z (Spin)
}
for col, cap in VIB_CLIP.items():
    X_raw[:, col] = np.clip(X_raw[:, col], 0, cap)

mask_gentle = X_raw[:, 18] < VAR_Z_THR1
mask_strong = (X_raw[:, 18] >= VAR_Z_THR1) & (X_raw[:, 18] < VAR_Z_THR2)
mask_spin   = X_raw[:, 18] >= VAR_Z_THR2

X_raw_gentle = X_raw[mask_gentle]
X_raw_strong = X_raw[mask_strong]
X_raw_spin   = X_raw[mask_spin]

counts = [mask_gentle.sum(), mask_strong.sum(), mask_spin.sum()]
labels = ['GENTLE\n(<0.001)', 'STRONG\n(0.001-0.010)', 'SPIN\n(>=0.010)']
colors = ['#3498DB', '#2ECC71', '#E67E22']

# --- VIZ 1: SỰ MẤT CÂN BẰNG DỮ LIỆU ---
plt.figure(figsize=(10, 5))
bars = plt.bar(labels, counts, color=colors, alpha=0.8)
plt.title('Phân phối Dữ liệu 3 Trạng thái (Tri-State Data Imbalance)', fontweight='bold')
plt.ylabel('Số lượng mẫu (Windows)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
plt.savefig('data_imbalance.png')
    # plt.show()

# %%
# ==========================================
# CELL 3: SCALER
# ==========================================
def safe_robust_fit(data):
    scaler = RobustScaler(quantile_range=(10, 90)).fit(data)
    # Ép scale tối thiểu để chống lỗi chia cho 0 khi dữ liệu quá tĩnh
    scaler.scale_ = np.maximum(scaler.scale_, 0.002)
    return scaler

vib_scaler_gentle = safe_robust_fit(X_raw_gentle[:, MEL_DIM:])
vib_scaler_strong = safe_robust_fit(X_raw_strong[:, MEL_DIM:])
vib_scaler_spin   = safe_robust_fit(X_raw_spin[:, MEL_DIM:])

def scale_subset(raw, vib_scaler):
    mel = np.clip((raw[:, :MEL_DIM] - MEL_DB_MIN) / MEL_DB_RANGE, 0, 1)
    vib = np.clip(vib_scaler.transform(raw[:, MEL_DIM:]), -3, 3) / 6.0 + 0.5
    return np.concatenate([mel, vib], axis=1).astype(np.float32)

X_scaled_gentle = scale_subset(X_raw_gentle, vib_scaler_gentle)
X_scaled_strong = scale_subset(X_raw_strong, vib_scaler_strong)
X_scaled_spin   = scale_subset(X_raw_spin, vib_scaler_spin)

# %%
# ==========================================
# CELL 4: HÀM AUGMENT DỮ LIỆU & TRAIN
# ==========================================
def augment_raw_data(X_raw, target_size=10000):
    """
    Tăng cường dữ liệu trong không gian vật lý (Raw Space).
    Vib: Magnitude Scaling (nhân hệ số). Audio: Volume Shifting (cộng/trừ dB).
    """
    n_samples = len(X_raw)
    repeat_factor = int(np.ceil(target_size / n_samples))
    X_aug = np.tile(X_raw, (repeat_factor, 1))[:target_size]
    
    # 1. Magnitude Scaling cho Vibration (Indices 13-18)
    # Tỷ lệ ngẫu nhiên từ 0.8x đến 1.2x
    vib_scales = np.random.uniform(0.8, 1.2, size=(target_size, 1))
    # RMS (index 13, 15, 17) tỷ lệ thuận với biên độ
    X_aug[:, [13, 15, 17]] *= vib_scales
    # VAR (index 14, 16, 18) tỷ lệ thuận với bình phương biên độ
    X_aug[:, [14, 16, 18]] *= (vib_scales**2)
    
    # 2. Volume Shifting cho Audio (Indices 0-12: Mel bands tính theo dB)
    # Thay đổi âm lượng ngẫu nhiên -5.0dB đến +2.0dB
    audio_shifts = np.random.uniform(-5.0, 2.0, size=(target_size, 1))
    X_aug[:, 0:13] += audio_shifts
    X_aug[:, 0:13] = np.clip(X_aug[:, 0:13], -80.0, 0.0) # Ép cứng [-80, 0] theo firmware
    
    # 3. Jittering (Nhiễu nền cực nhỏ)
    X_aug[:, 0:13] += np.random.normal(0, 0.3, size=(target_size, 13)) # Nhiễu dB
    X_aug[:, 13:19] += np.random.normal(0, 0.0005, size=(target_size, 6)) # Nhiễu rung lắc
    
    return X_aug.astype(np.float32)

def weighted_mae(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred) * tf.constant(WEIGHTS), axis=-1) / W_SUM

def build_model():
    return models.Sequential([
        layers.InputLayer(input_shape=(FEAT_DIM,)),
        layers.Reshape((FEAT_DIM, 1)),
        
        # ENCODER
        layers.Conv1D(16, kernel_size=3, padding='same', activation='relu'), 
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.Conv1D(8, kernel_size=3, padding='same', activation='relu'),  
        layers.MaxPooling1D(pool_size=2, padding='same'),
        
        # BOTTLENECK
        layers.Flatten(),
        layers.Dense(8, activation='sigmoid'),

        # DECODER
        layers.Dense(40, activation='relu'),
        layers.Reshape((5, 8)),
        layers.UpSampling1D(size=2),                                     
        layers.Conv1D(16, kernel_size=3, padding='same', activation='relu'),
        layers.UpSampling1D(size=2),                                     
        layers.Conv1D(1, kernel_size=3, padding='same', activation='sigmoid'), 
        
        layers.Cropping1D(cropping=(0, 1)),
        layers.Flatten()
    ])

def plot_training_history(hist_float, name, color):
    plt.figure(figsize=(8, 4))
    plt.plot(hist_float.history['loss'], label='Train Loss', color=color)
    plt.plot(hist_float.history['val_loss'], label='Val Loss', color='gray', linestyle='--')
    plt.title(f'Lịch sử Huấn luyện - {name} (Float32)', fontweight='bold')
    plt.xlabel('Epochs'); plt.ylabel('Weighted MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'history_{name}.png')
    # plt.show()

def train_pipeline(X_raw_subset, vib_scaler, name, color):
    print(f"\n>>> Training: {name} (Original: {len(X_raw_subset)} samples)")
    
    # 1. Thực hiện Data Augmentation trên RAW data
    target = 20000 if name == "GENTLE" else 10000
    print(f"--- Augmenting to {target} samples...")
    X_aug_raw = augment_raw_data(X_raw_subset, target_size=target)
    
    # 2. Scale dữ liệu đã Augment
    print("--- Scaling data...")
    X_clean_aug = scale_subset(X_aug_raw, vib_scaler)
    
    # 3. Tạo dữ liệu đầu vào có nhiễu
    print("--- Adding noise...")
    noise = np.random.normal(0, 0.02, X_clean_aug.shape).astype(np.float32)
    X_noisy_aug = np.clip(X_clean_aug + noise, 0, 1)

    # 4. Training Phase (Float32)
    bs = 64
    print("--- Building model...")
    m = build_model()
    print("--- Compiling model...")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_mae)
    print("--- Starting Training (Float32)...")
    # Tăng epoch lên 300 cho PTQ ổn định hơn
    hist_float = m.fit(X_noisy_aug, X_clean_aug, epochs=300, batch_size=bs, validation_split=0.15, verbose=1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
    
    plot_training_history(hist_float, name, color)
    return m

model_gentle = train_pipeline(X_raw_gentle, vib_scaler_gentle, "GENTLE", colors[0])
model_strong = train_pipeline(X_raw_strong, vib_scaler_strong, "STRONG", colors[1])
model_spin   = train_pipeline(X_raw_spin,   vib_scaler_spin,   "SPIN",   colors[2])

# %%
# ==========================================
# CELL 5: XUẤT TFLITE & TÍNH THRESHOLD DỰA TRÊN HARD-TEST
# ==========================================
# Ghi chú cho Giai đoạn 4:
# Các hệ số Augmentation tối ưu được chọn sau nhiều lần thử nghiệm:
# - Vibration Scaling: 0.8x - 1.2x (Giả lập biến thiên tải trọng thực tế)
# - Audio Shifting: -5dB to +2dB (Cân bằng giữa nhiễu môi trường và cường độ âm motor)
# - Anomaly Z-Factor: 2.5x - 4.5x RMS (Văng lồng rõ rệt nhưng không phi vật lý)

def synthesize_anomalies(X_raw, name=""):
    """
    Tạo dữ liệu bất thường giả lập (Anomaly) từ dữ liệu sạch.
    Điều chỉnh tham số theo từng mode. GENTLE cần lỗi mạnh hơn để phân biệt.
    """
    X_anom = X_raw.copy()
    n = len(X_anom)
    
    # Hệ số khuếch đại cho GENTLE
    sh = 1.5 if name == "GENTLE" else 1.0

    # 1. Giả lập văng lồng: Tăng mạnh trục Z (rms_z, var_z)
    X_anom[:, 17] *= np.random.uniform(3.0 * sh, 6.0 * sh, size=n)
    X_anom[:, 18] *= np.random.uniform(7.0 * sh, 15.0 * sh, size=n)
    # 2. Giả lập lệch trọng tâm: Cộng offset vào X/Y (tăng biên độ lỗi)
    X_anom[:, [13, 15]] += np.random.uniform(0.2 * sh, 0.6 * sh, size=(n, 2))
    # 3. Giả lập ồn môi trường: Tăng dB Mel (lỗi rõ hơn)
    X_anom[:, 0:13] += np.random.uniform(8 * sh, 15 * sh, size=(n, 13))
    X_anom[:, 0:13] = np.clip(X_anom[:, 0:13], -80.0, 0.0)
    return X_anom.astype(np.float32)

def calculate_optimal_threshold(maes_normal, maes_anomaly, name=""):
    """
    Tìm ngưỡng tối ưu dựa trên việc cân bằng giữa Normal và Anomaly (Giao điểm hoặc F1).
    """
    all_maes = np.concatenate([maes_normal, maes_anomaly])
    labels = np.concatenate([np.zeros(len(maes_normal)), np.ones(len(maes_anomaly))])
    
    thresholds = np.linspace(np.min(maes_normal), np.max(maes_anomaly), 200)
    best_f1 = -1
    best_thr = -1
    
    for thr in thresholds:
        preds = (all_maes > thr).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    # GENTLE và STRONG cho phép nới lỏng ngưỡng để tăng độ nhạy (Recall)
    p_level = 95 if name == "GENTLE" else 98
    safe_min = np.percentile(maes_normal, p_level)
    final_thr = max(best_thr, safe_min)
    
    # Tính toán bộ metrics cuối cùng tại final_thr
    final_preds = (all_maes > final_thr).astype(int)
    results = {
        'threshold': final_thr,
        'f1': f1_score(labels, final_preds),
        'precision': precision_score(labels, final_preds),
        'recall': recall_score(labels, final_preds),
        'auc': roc_auc_score(labels, all_maes)
    }
    return results

def export_int8_and_verify(float_m, X_raw_subset, vib_scaler, name, color):
    # 1. Chuẩn bị dữ liệu Scaling
    X_scaled_normal = scale_subset(X_raw_subset, vib_scaler)
    X_raw_anom = synthesize_anomalies(X_raw_subset, name=name)
    X_scaled_anom = scale_subset(X_raw_anom, vib_scaler)
    
    # 2. Post-Training Quantization (PTQ) - Full Integer Quantization
    def rep_gen():
        # Dùng 100 mẫu đại diện để tính dải động cho INT8
        for i in range(min(100, len(X_scaled_normal))):
            yield [X_scaled_normal[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(float_m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    
    # Ép kiểu toàn bộ tensor thành INT8
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    
    print(f"--- Converting {name} to TFLite INT8 (PTQ)...")
    tfl = conv.convert()
    
    # 3. Đánh giá MAE trên mô hình INT8
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    i_sc, i_zp = inp_d['quantization']; o_sc, o_zp = out_d['quantization']
    
    def get_maes(data):
        m_list = []
        for x in data:
            xi = np.clip(np.round(x/i_sc)+i_zp, -128, 127).astype(np.int8).reshape(1, FEAT_DIM)
            interp.set_tensor(inp_d['index'], xi); interp.invoke()
            yo = interp.get_tensor(out_d['index']).reshape(FEAT_DIM).astype(np.float32)
            yf = np.clip((yo - o_zp) * o_sc, 0, 1)
            xf = np.clip((xi.reshape(FEAT_DIM).astype(np.float32) - i_zp) * i_sc, 0, 1)
            m_list.append(np.sum(np.abs(xf - yf) * WEIGHTS) / W_SUM)
        return np.array(m_list)

    maes_n = get_maes(X_scaled_normal)
    maes_a = get_maes(X_scaled_anom)
    
    # 4. Tìm Threshold tối ưu và Metrics
    results = calculate_optimal_threshold(maes_n, maes_a, name=name)
    thr = results['threshold']
    
    print(f"[{name}] Optimal Threshold: {thr:.4f}")
    print(f"[{name}] Metrics - F1: {results['f1']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, AUC: {results['auc']:.4f}")
    
    # --- VIZ 5: SO SÁNH PHÂN PHỐI NORMAL VS ANOMALY ---
    plt.figure(figsize=(10, 5))
    plt.hist(maes_n, bins=40, color=color, alpha=0.6, label='Normal Data (Gốc)')
    plt.hist(maes_a, bins=40, color='red', alpha=0.4, label='Synthesized Anomaly (Giả lập)')
    plt.axvline(thr, color='black', linestyle='--', linewidth=3, label=f'Threshold: {thr:.4f} (F1: {results["f1"]:.2f})')
    plt.title(f'Phân tích Phân tách Lỗi (MAE Separation) - {name}', fontweight='bold')
    plt.xlabel('MAE'); plt.ylabel('Số lượng mẫu')
    plt.legend()
    plt.savefig(f'mae_sep_{name}.png')
    # plt.show()
    
    return tfl, thr

tfl_g, thr_g = export_int8_and_verify(model_gentle, X_raw_gentle, vib_scaler_gentle, "GENTLE", colors[0])
tfl_s, thr_s = export_int8_and_verify(model_strong, X_raw_strong, vib_scaler_strong, "STRONG", colors[1])
tfl_sp, thr_sp = export_int8_and_verify(model_spin, X_raw_spin,   vib_scaler_spin,   "SPIN",   colors[2])

# %%
# ==========================================
# CELL 6: TẠO FILE model_data.h CHO C++
# ==========================================
def arr_to_c(data, name):
    lines = ['  ' + ', '.join(f'0x{b:02x}' for b in data[i:i+12]) for i in range(0, len(data), 12)]
    return f"const unsigned char {name}[] __attribute__((aligned(16))) = {{\n" + ",\n".join(lines) + f"\n}};\nconst unsigned int {name}_len = {len(data)};\n\n"
def floats(arr): return ', '.join(f'{v:.10f}f' for v in arr)

h  = "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n// v8.7 Ultimate\n\n"
h += f"const float THRESHOLD_GENTLE = {thr_g:.10f}f;\nconst float THRESHOLD_STRONG = {thr_s:.10f}f;\nconst float THRESHOLD_SPIN   = {thr_sp:.10f}f;\n\n"
h += f"const float MEL_MIN[{MEL_DIM}]   = {{{floats(np.full((MEL_DIM,), MEL_DB_MIN, dtype=np.float32))}}};\n"
h += f"const float MEL_SCALE[{MEL_DIM}] = {{{floats(np.full((MEL_DIM,), MEL_DB_RANGE, dtype=np.float32))}}};\n\n"
h += f"const float VIB_CENTER_GENTLE[{VIB_DIM}] = {{{floats(vib_scaler_gentle.center_)}}};\nconst float VIB_SCALE_GENTLE[{VIB_DIM}]  = {{{floats(vib_scaler_gentle.scale_)}}};\n"
h += f"const float VIB_CENTER_STRONG[{VIB_DIM}] = {{{floats(vib_scaler_strong.center_)}}};\nconst float VIB_SCALE_STRONG[{VIB_DIM}]  = {{{floats(vib_scaler_strong.scale_)}}};\n"
h += f"const float VIB_CENTER_SPIN[{VIB_DIM}]   = {{{floats(vib_scaler_spin.center_)}}};\nconst float VIB_SCALE_SPIN[{VIB_DIM}]    = {{{floats(vib_scaler_spin.scale_)}}};\n\n"
h += f"const float VIB_CLIP_VAR_X = {VIB_CLIP[14]:.10f}f;\nconst float VIB_CLIP_VAR_Y = {VIB_CLIP[16]:.10f}f;\nconst float VIB_CLIP_VAR_Z = {VIB_CLIP[18]:.10f}f;\n\n"
h += arr_to_c(tfl_g, "model_gentle_tflite") + arr_to_c(tfl_s, "model_strong_tflite") + arr_to_c(tfl_sp, "model_spin_tflite") + "#endif\n"

open('firmware_v12/model_data.h', 'w').write(h)

print("[OK] Finished! Exported model_data.h for C++ Firmware!")