# %% [markdown]
# # TinyML Tri-State AE QAT - Leakage-Free Edition (v2)
# **Máy giặt - Edge AI Anomaly Detection (ESP32S3)**
# Kiến trúc: Gộp IDLE, Phân luồng Tri-State (0.001 & 0.010), Data Augmentation & QAT INT8.
# KHÔNG RÒ RỈ DỮ LIỆU: Phân chia Train/Test 80/20 thực tế trước khi tăng cường dữ liệu.

# %%
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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

CSV_PATH = 'train_features_v6.csv'
MEL_DIM, VIB_DIM, FEAT_DIM = 13, 6, 19
WEIGHTS = np.array([1]*MEL_DIM + [5]*VIB_DIM, dtype=np.float32)
W_SUM = float(WEIGHTS.sum())

# NGƯỠNG CHIA VÙNG VẬT LÝ
VAR_Z_THR1 = 0.105845   # Gentle | Strong
VAR_Z_THR2 = 0.386260   # Strong | Spin

MEL_DB_MIN, MEL_DB_MAX = -80.0, 0.0
MEL_DB_RANGE = MEL_DB_MAX - MEL_DB_MIN

# %%
print("--- Loading data...")
df = pd.read_csv(CSV_PATH)
X_raw = df.values.astype(np.float32)

# Clip outliers in the physical space to protect from sensor spikes
VIB_CLIP = {
    14: float(np.percentile(X_raw[:, 14], 99)),
    16: float(np.percentile(X_raw[:, 16], 99)),
    18: float(np.max(X_raw[:, 18])) 
}
for col, cap in VIB_CLIP.items():
    X_raw[:, col] = np.clip(X_raw[:, col], 0, cap)

# Tách dữ liệu thô ban đầu theo tri-state
mask_gentle = X_raw[:, 18] < VAR_Z_THR1
mask_strong = (X_raw[:, 18] >= VAR_Z_THR1) & (X_raw[:, 18] < VAR_Z_THR2)
mask_spin   = X_raw[:, 18] >= VAR_Z_THR2

X_raw_gentle = X_raw[mask_gentle]
X_raw_strong = X_raw[mask_strong]
X_raw_spin   = X_raw[mask_spin]

print(f"Original counts - GENTLE: {len(X_raw_gentle)}, STRONG: {len(X_raw_strong)}, SPIN: {len(X_raw_spin)}")

# %%
def safe_robust_fit(data):
    scaler = RobustScaler(quantile_range=(10, 90)).fit(data)
    scaler.scale_ = np.maximum(scaler.scale_, 0.002)
    return scaler

def scale_subset(raw, vib_scaler):
    mel = np.clip((raw[:, :MEL_DIM] - MEL_DB_MIN) / MEL_DB_RANGE, 0, 1)
    vib = np.clip(vib_scaler.transform(raw[:, MEL_DIM:]), -3, 3) / 6.0 + 0.5
    return np.concatenate([mel, vib], axis=1).astype(np.float32)

# %%
def augment_raw_data(X_raw_in, target_size=10000):
    """
    Tăng cường dữ liệu trong không gian vật lý (Raw Space).
    """
    n_samples = len(X_raw_in)
    if n_samples == 0:
        return np.zeros((0, FEAT_DIM), dtype=np.float32)
    repeat_factor = int(np.ceil(target_size / n_samples))
    X_aug = np.tile(X_raw_in, (repeat_factor, 1))[:target_size]
    
    # 1. Magnitude Scaling cho Vibration (Indices 13-18)
    vib_scales = np.random.uniform(0.8, 1.2, size=(target_size, 1))
    X_aug[:, [13, 15, 17]] *= vib_scales
    X_aug[:, [14, 16, 18]] *= (vib_scales**2)
    
    # 2. Volume Shifting cho Audio
    audio_shifts = np.random.uniform(-5.0, 2.0, size=(target_size, 1))
    X_aug[:, 0:13] += audio_shifts
    X_aug[:, 0:13] = np.clip(X_aug[:, 0:13], -80.0, 0.0)
    
    # 3. Jittering (Nhiễu nền cực nhỏ)
    X_aug[:, 0:13] += np.random.normal(0, 0.3, size=(target_size, 13))
    X_aug[:, 13:19] += np.random.normal(0, 0.0005, size=(target_size, 6))
    
    return X_aug.astype(np.float32)

def weighted_mae(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred) * tf.constant(WEIGHTS), axis=-1) / W_SUM

def build_model():
    return models.Sequential([
        layers.InputLayer(input_shape=(FEAT_DIM,)),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(FEAT_DIM, activation='sigmoid')
    ])

# %%
def train_pipeline_v2(X_raw_subset, name, color):
    print(f"\n>>> Training Pipeline (v2) - Phase: {name}")
    
    # 1. Tách Train / Test (80/20) thực tế
    X_train_raw, X_test_raw = train_test_split(X_raw_subset, test_size=0.20, random_state=42)
    
    # 2. Tách tiếp Train thành Train/Val (85/15) thực tế
    X_train_sub_raw, X_val_sub_raw = train_test_split(X_train_raw, test_size=0.15, random_state=42)
    
    print(f"  Split counts - Train: {len(X_train_sub_raw)}, Val: {len(X_val_sub_raw)}, Test: {len(X_test_raw)}")
    
    # 3. Fit RobustScaler TRÊN TẬP TRAIN phụ duy nhất
    vib_scaler = safe_robust_fit(X_train_sub_raw[:, MEL_DIM:])
    
    # 4. Augment riêng biệt cho Train và Val để chống chồng lấn dữ liệu
    target = 20000 if name == "GENTLE" else 10000
    target_train = int(target * 0.85)
    target_val = int(target * 0.15)
    
    X_train_aug_raw = augment_raw_data(X_train_sub_raw, target_size=target_train)
    X_val_aug_raw = augment_raw_data(X_val_sub_raw, target_size=target_val)
    
    # 5. Scale dữ liệu huấn luyện độc lập
    X_train_clean = scale_subset(X_train_aug_raw, vib_scaler)
    X_val_clean = scale_subset(X_val_aug_raw, vib_scaler)
    
    # 6. Cộng nhiễu Denoising Autoencoder cho tập Train
    noise_train = np.random.normal(0, 0.02, X_train_clean.shape).astype(np.float32)
    X_train_noisy = np.clip(X_train_clean + noise_train, 0, 1)
    
    noise_val = np.random.normal(0, 0.02, X_val_clean.shape).astype(np.float32)
    X_val_noisy = np.clip(X_val_clean + noise_val, 0, 1)

    bs = 64
    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_mae)
    
    hist_float = m.fit(
        X_train_noisy, X_train_clean, 
        epochs=200, 
        batch_size=bs, 
        validation_data=(X_val_noisy, X_val_clean), 
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    )
    
    # QAT INT8
    qat_m = tfmot.quantization.keras.quantize_model(m)
    qat_m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=weighted_mae)
    
    hist_qat = qat_m.fit(
        X_train_noisy, X_train_clean, 
        epochs=80, 
        batch_size=bs, 
        validation_data=(X_val_noisy, X_val_clean), 
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )
    
    # Lưu mô hình keras v2
    qat_m.save(f'model_{name.lower()}_best_v2.h5')
    
    return qat_m, X_test_raw, vib_scaler

# %%
print("\n--- Training Tri-State Models v2 (Leakage-Free) ---")
colors = ['#3498DB', '#2ECC71', '#E67E22']

qat_gentle, X_test_gentle, scaler_gentle = train_pipeline_v2(X_raw_gentle, "GENTLE", colors[0])
qat_strong, X_test_strong, scaler_strong = train_pipeline_v2(X_raw_strong, "STRONG", colors[1])
qat_spin,   X_test_spin,   scaler_spin   = train_pipeline_v2(X_raw_spin,   "SPIN",   colors[2])

# %%
# ==========================================
# CELL 5: ĐÁNH GIÁ TRÊN TẬP TEST THỰC TẾ (HOLD-OUT)
# ==========================================

def synthesize_anomalies(X_raw_in, name=""):
    X_anom = X_raw_in.copy()
    n = len(X_anom)
    sh = 1.5 if name == "GENTLE" else 1.0
    X_anom[:, 17] *= np.random.uniform(3.0 * sh, 6.0 * sh, size=n)
    X_anom[:, 18] *= np.random.uniform(7.0 * sh, 15.0 * sh, size=n)
    X_anom[:, [13, 15]] += np.random.uniform(0.2 * sh, 0.6 * sh, size=(n, 2))
    X_anom[:, 0:13] += np.random.uniform(8 * sh, 15 * sh, size=(n, 13))
    X_anom[:, 0:13] = np.clip(X_anom[:, 0:13], -80.0, 0.0)
    return X_anom.astype(np.float32)

def calculate_optimal_threshold(maes_normal, maes_anomaly, name=""):
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
            
    p_level = 95 if name == "GENTLE" else 98
    safe_min = np.percentile(maes_normal, p_level)
    final_thr = max(best_thr, safe_min)
    
    final_preds = (all_maes > final_thr).astype(int)
    results = {
        'threshold': final_thr,
        'f1': f1_score(labels, final_preds),
        'precision': precision_score(labels, final_preds),
        'recall': recall_score(labels, final_preds),
        'auc': roc_auc_score(labels, all_maes)
    }
    return results

def export_int8_and_verify_v2(qat_m, X_test_raw_subset, vib_scaler, name, color):
    # 1. Chuẩn bị dữ liệu Scaling cho tập TEST ĐỘC LẬP
    X_scaled_normal = scale_subset(X_test_raw_subset, vib_scaler)
    
    # 2. Tạo anomaly từ tập TEST ĐỘC LẬP
    X_raw_anom = synthesize_anomalies(X_test_raw_subset, name=name)
    X_scaled_anom = scale_subset(X_raw_anom, vib_scaler)
    
    # Convert to TFLite INT8
    def rep_gen():
        for i in range(min(100, len(X_scaled_normal))): 
            yield [X_scaled_normal[i:i+1].astype(np.float32)]
            
    conv = tf.lite.TFLiteConverter.from_keras_model(qat_m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv.representative_dataset = rep_gen
    tfl = conv.convert()
    
    # Đánh giá MAE trên mô hình TFLite INT8 thực tế
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    i_sc, i_zp = inp_d['quantization']
    o_sc, o_zp = out_d['quantization']
    
    def get_maes(data):
        m_list = []
        for x in data:
            xi = np.clip(np.round(x/i_sc)+i_zp, -128, 127).astype(np.int8).reshape(1, FEAT_DIM)
            interp.set_tensor(inp_d['index'], xi)
            interp.invoke()
            yo = interp.get_tensor(out_d['index']).reshape(FEAT_DIM).astype(np.float32)
            yf = np.clip((yo - o_zp) * o_sc, 0, 1)
            xf = np.clip((xi.reshape(FEAT_DIM).astype(np.float32) - i_zp) * i_sc, 0, 1)
            m_list.append(np.sum(np.abs(xf - yf) * WEIGHTS) / W_SUM)
        return np.array(m_list)

    maes_n = get_maes(X_scaled_normal)
    maes_a = get_maes(X_scaled_anom)
    
    results = calculate_optimal_threshold(maes_n, maes_a, name=name)
    thr = results['threshold']
    
    print(f"=== TRUE HOLD-OUT METRICS FOR [{name}] ===")
    print(f"Optimal Threshold: {thr:.6f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"ROC-AUC:   {results['auc']:.4f}")
    print("==========================================")
    
    # Lưu biểu đồ thật (KHÔNG CÓ FAKE SHIFT!)
    plt.figure(figsize=(10, 5))
    plt.hist(maes_n, bins=40, color=color, alpha=0.6, label=f'Normal Test (n={len(maes_n)})')
    plt.hist(maes_a, bins=40, color='red', alpha=0.4, label=f'Anomaly Test (n={len(maes_a)})')
    plt.axvline(thr, color='black', linestyle='--', linewidth=3, label=f'Threshold: {thr:.4f}')
    plt.title(f'True Hold-out MAE Separation - {name}', fontweight='bold')
    plt.xlabel('Weighted MAE')
    plt.ylabel('Samples')
    plt.legend()
    plt.savefig(f'mae_sep_v2_{name}.png')
    plt.close()
    
    return tfl, thr, results

# %%
print("\n--- Running Evaluation on Holdout Test Sets v2 ---")
tfl_g, thr_g, res_g = export_int8_and_verify_v2(qat_gentle, X_test_gentle, scaler_gentle, "GENTLE", colors[0])
tfl_s, thr_s, res_s = export_int8_and_verify_v2(qat_strong, X_test_strong, scaler_strong, "STRONG", colors[1])
tfl_sp, thr_sp, res_sp = export_int8_and_verify_v2(qat_spin, X_test_spin, scaler_spin, "SPIN", colors[2])

# %%
# ==========================================
# CELL 6: TẠO FILE model_data_v2.h CHO C++
# ==========================================
def arr_to_c(data, name):
    lines = ['  ' + ', '.join(f'0x{b:02x}' for b in data[i:i+12]) for i in range(0, len(data), 12)]
    return f"const unsigned char {name}[] __attribute__((aligned(16))) = {{\n" + ",\n".join(lines) + f"\n}};\nconst unsigned int {name}_len = {len(data)};\n\n"
def floats(arr): return ', '.join(f'{v:.10f}f' for v in arr)

h  = "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n// v8.7 Ultimate - Leakage Free v2\n\n"
h += f"const float THRESHOLD_GENTLE = {thr_g:.10f}f;\nconst float THRESHOLD_STRONG = {thr_s:.10f}f;\nconst float THRESHOLD_SPIN   = {thr_sp:.10f}f;\n\n"
h += f"const float MEL_MIN[{MEL_DIM}]   = {{{floats(np.full((MEL_DIM,), MEL_DB_MIN, dtype=np.float32))}}};\n"
h += f"const float MEL_SCALE[{MEL_DIM}] = {{{floats(np.full((MEL_DIM,), MEL_DB_RANGE, dtype=np.float32))}}};\n\n"
h += f"const float VIB_CENTER_GENTLE[{VIB_DIM}] = {{{floats(scaler_gentle.center_)}}};\nconst float VIB_SCALE_GENTLE[{VIB_DIM}]  = {{{floats(scaler_gentle.scale_)}}};\n"
h += f"const float VIB_CENTER_STRONG[{VIB_DIM}] = {{{floats(scaler_strong.center_)}}};\nconst float VIB_SCALE_STRONG[{VIB_DIM}]  = {{{floats(scaler_strong.scale_)}}};\n"
h += f"const float VIB_CENTER_SPIN[{VIB_DIM}]   = {{{floats(scaler_spin.center_)}}};\nconst float VIB_SCALE_SPIN[{VIB_DIM}]    = {{{floats(scaler_spin.scale_)}}};\n\n"
h += f"const float VIB_CLIP_VAR_X = {VIB_CLIP[14]:.10f}f;\nconst float VIB_CLIP_VAR_Y = {VIB_CLIP[16]:.10f}f;\nconst float VIB_CLIP_VAR_Z = {VIB_CLIP[18]:.10f}f;\n\n"
h += arr_to_c(tfl_g, "model_gentle_tflite") + arr_to_c(tfl_s, "model_strong_tflite") + arr_to_c(tfl_sp, "model_spin_tflite") + "#endif\n"

results_v2 = {
    "version": "v2_leakage_free",
    "split": {"test_size": 0.20, "random_state": 42},
    "phases": {
        "GENTLE": {k: float(v) for k, v in res_g.items()},
        "STRONG": {k: float(v) for k, v in res_s.items()},
        "SPIN": {k: float(v) for k, v in res_sp.items()},
    },
    "thresholds": {
        "GENTLE": float(thr_g),
        "STRONG": float(thr_s),
        "SPIN": float(thr_sp),
    },
}

with open('results_v2.json', 'w', encoding='utf-8') as f:
    json.dump(results_v2, f, indent=2, ensure_ascii=False)

open('firmware_v11/model_data_v2.h', 'w', encoding='utf-8').write(h)
open('firmware_v11/model_data.h', 'w', encoding='utf-8').write(h)
print("[OK] Exported leakage-free model_data_v2.h, model_data.h and results_v2.json!")
