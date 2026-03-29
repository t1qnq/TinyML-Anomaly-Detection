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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Style biểu đồ Kaggle
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

CSV_PATH = '/kaggle/input/datasets/quchngcquang/dataset2603/train_features_v6.csv'
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
print("⏳ Đang tải dữ liệu...")
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
plt.show()

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
# CELL 4: HÀM TRAIN & TRỰC QUAN HÓA LOSS
# ==========================================
def weighted_mae(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred) * tf.constant(WEIGHTS), axis=-1) / W_SUM

def build_model():
    return models.Sequential([
        layers.InputLayer(input_shape=(FEAT_DIM,)),
        layers.Dense(128, activation='sigmoid'), layers.Dense(64,  activation='sigmoid'),
        layers.Dense(32,  activation='sigmoid'), layers.Dense(64,  activation='sigmoid'),
        layers.Dense(128, activation='sigmoid'), layers.Dense(FEAT_DIM, activation='sigmoid')
    ])

def plot_training_history(hist_float, hist_qat, name, color):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'Lịch sử Huấn luyện - Mô hình {name}', fontweight='bold', fontsize=14)
    
    axes[0].plot(hist_float.history['loss'], label='Train Loss', color=color)
    axes[0].plot(hist_float.history['val_loss'], label='Val Loss', color='gray', linestyle='--')
    axes[0].set_title('Phase 1: Float32 Pre-training')
    axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Weighted MAE')
    axes[0].legend()
    
    axes[1].plot(hist_qat.history['loss'], label='Train Loss (QAT)', color=color)
    axes[1].plot(hist_qat.history['val_loss'], label='Val Loss (QAT)', color='gray', linestyle='--')
    axes[1].set_title('Phase 2: INT8 Quantization-Aware Training')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    plt.tight_layout(); plt.show()

def train_pipeline(X_scaled, name, color):
    print(f"\n🚀 Đang huấn luyện: {name} (Gốc: {len(X_scaled)} mẫu)")
    
    # DATA AUGMENTATION
    TARGET_SIZE = 10000
    if len(X_scaled) < TARGET_SIZE:
        repeat_factor = int(np.ceil(TARGET_SIZE / len(X_scaled)))
        print(f"   ⚠️ Ép xung dữ liệu! Nhân bản tập {name} lên {repeat_factor} lần...")
        X_clean_aug = np.tile(X_scaled, (repeat_factor, 1))
    else:
        X_clean_aug = X_scaled.copy()

    noise_level = 0.04 if len(X_scaled) < 100 else 0.02
    noise = np.random.normal(0, noise_level, X_clean_aug.shape).astype(np.float32)
    X_noisy_aug = np.clip(X_clean_aug + noise, 0, 1)
    bs = 16 if len(X_scaled) < 200 else 64

    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_mae)
    hist_float = m.fit(X_noisy_aug, X_clean_aug, epochs=200, batch_size=bs, validation_split=0.15, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    
    qat_m = tfmot.quantization.keras.quantize_model(m)
    qat_m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=weighted_mae)
    hist_qat = qat_m.fit(X_noisy_aug, X_clean_aug, epochs=80, batch_size=bs, validation_split=0.15, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    
    plot_training_history(hist_float, hist_qat, name, color)
    return qat_m

qat_gentle = train_pipeline(X_scaled_gentle, "GENTLE", colors[0])
qat_strong = train_pipeline(X_scaled_strong, "STRONG", colors[1])
qat_spin   = train_pipeline(X_scaled_spin,   "SPIN",   colors[2])

# %%
# ==========================================
# CELL 5: XUẤT TFLITE & TRỰC QUAN HÓA THRESHOLD
# ==========================================
def export_int8_and_verify(qat_m, X_scaled, name, color):
    def rep_gen():
        for i in range(min(100, len(X_scaled))): yield [X_scaled[i:i+1].astype(np.float32)]
    conv = tf.lite.TFLiteConverter.from_keras_model(qat_m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv.representative_dataset = rep_gen
    tfl = conv.convert()
    
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    i_sc, i_zp = inp_d['quantization']; o_sc, o_zp = out_d['quantization']
    
    maes = []
    # Đánh giá MAE trên tập gốc (không nhiễu) để tính Threshold sát thực tế nhất
    for x in X_scaled:
        xi = np.clip(np.round(x/i_sc)+i_zp, -128, 127).astype(np.int8).reshape(1, FEAT_DIM)
        interp.set_tensor(inp_d['index'], xi); interp.invoke()
        yo = interp.get_tensor(out_d['index']).reshape(FEAT_DIM).astype(np.float32)
        yf = np.clip((yo - o_zp) * o_sc, 0, 1)
        xf = np.clip((xi.reshape(FEAT_DIM).astype(np.float32) - i_zp) * i_sc, 0, 1)
        maes.append(np.sum(np.abs(xf - yf) * WEIGHTS) / W_SUM)
        
    maes = np.array(maes)
    mean_mae, std_mae = np.mean(maes), np.std(maes)
    
    # Tính threshold: Mean + 3 Sigma
    thr = max(mean_mae + 3 * std_mae, np.percentile(maes, 99.5))
    
    # --- VIZ 3: PHÂN PHỐI MAE VÀ NGƯỠNG AN TOÀN ---
    plt.figure(figsize=(8, 4))
    plt.hist(maes, bins=40, color=color, alpha=0.7)
    plt.axvline(mean_mae, color='black', linestyle=':', linewidth=2, label=f'Mean MAE: {mean_mae:.4f}')
    plt.axvline(thr, color='red', linestyle='--', linewidth=2, label=f'Threshold (Mean+3σ): {thr:.4f}')
    plt.title(f'Phân phối Lỗi tái tạo (MAE) & Ngưỡng An toàn - {name}', fontweight='bold')
    plt.xlabel('MAE (Lỗi càng nhỏ càng tốt)'); plt.ylabel('Số lượng mẫu')
    plt.legend(); plt.show()
    
    return tfl, thr

tfl_g, thr_g = export_int8_and_verify(qat_gentle, X_scaled_gentle, "GENTLE", colors[0])
tfl_s, thr_s = export_int8_and_verify(qat_strong, X_scaled_strong, "STRONG", colors[1])
tfl_sp, thr_sp = export_int8_and_verify(qat_spin, X_scaled_spin, "SPIN", colors[2])

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

open('model_data.h', 'w').write(h)
print("✅ Hoàn tất! Đã xuất file model_data.h cho Firmware C++!")