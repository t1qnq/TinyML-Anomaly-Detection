import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from sklearn.preprocessing import RobustScaler

# === CONFIGURATION ===
CSV_PATH = 'train_features_v6.csv'
MEL_DIM, VIB_DIM, FEAT_DIM = 13, 6, 19
WEIGHTS = np.array([1]*MEL_DIM + [5]*VIB_DIM, dtype=np.float32)
W_SUM = float(WEIGHTS.sum())
VAR_Z_THR1, VAR_Z_THR2 = 0.105845, 0.386260
MEL_DB_MIN, MEL_DB_MAX = -80.0, 0.0
MEL_DB_RANGE = MEL_DB_MAX - MEL_DB_MIN

plt.style.use('seaborn-v0_8-darkgrid')

# === HELPER FUNCTIONS ===
def weighted_mae_np(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) * WEIGHTS, axis=-1) / W_SUM

def synthesize_anomalies(X_raw, name=""):
    X_anom = X_raw.copy()
    n = len(X_anom)
    sh = 1.5 if name == "GENTLE" else 1.0
    X_anom[:, 17] *= np.random.uniform(3.0 * sh, 6.0 * sh, size=n)
    X_anom[:, 18] *= np.random.uniform(7.0 * sh, 15.0 * sh, size=n)
    X_anom[:, [13, 15]] += np.random.uniform(0.2 * sh, 0.6 * sh, size=(n, 2))
    X_anom[:, 0:13] += np.random.uniform(8 * sh, 15 * sh, size=(n, 13))
    X_anom[:, 0:13] = np.clip(X_anom[:, 0:13], -80.0, 0.0)
    return X_anom.astype(np.float32)

def get_scaler_and_scale(X_raw):
    scaler = RobustScaler(quantile_range=(10, 90)).fit(X_raw[:, MEL_DIM:])
    scaler.scale_ = np.maximum(scaler.scale_, 0.002)
    mel = np.clip((X_raw[:, :MEL_DIM] - MEL_DB_MIN) / MEL_DB_RANGE, 0, 1)
    vib = np.clip(scaler.transform(X_raw[:, MEL_DIM:]), -3, 3) / 6.0 + 0.5
    return np.concatenate([mel, vib], axis=1).astype(np.float32)

import tensorflow_model_optimization as tfmot

# === ANALYSIS CORE ===
def analyze_phase(phase_name):
    print(f"\nAnalyzing Phase: {phase_name}...")
    
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    X_raw_all = df.values.astype(np.float32)
    if phase_name == "GENTLE":
        X_raw = X_raw_all[X_raw_all[:, 18] < VAR_Z_THR1]
    elif phase_name == "STRONG":
        X_raw = X_raw_all[(X_raw_all[:, 18] >= VAR_Z_THR1) & (X_raw_all[:, 18] < VAR_Z_THR2)]
    else: # SPIN
        X_raw = X_raw_all[X_raw_all[:, 18] >= VAR_Z_THR2]
        
    X_normal = get_scaler_and_scale(X_raw)
    X_anom_raw = synthesize_anomalies(X_raw, name=phase_name)
    X_anom = get_scaler_and_scale(X_anom_raw)
    
    # 2. Load Model
    model_path = f'model_{phase_name.lower()}_best.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Run experimental_training.py first.")
        return
    
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(model_path, compile=False)
    
    # 3. Predict & Calculate MAE
    pred_normal = model.predict(X_normal, verbose=0)
    pred_anom = model.predict(X_anom, verbose=0)
    
    maes_n = weighted_mae_np(X_normal, pred_normal)
    maes_a = weighted_mae_np(X_anom, pred_anom)
    
    # 4. ROC Curve & Youden's J
    y_true = np.concatenate([np.zeros(len(maes_n)), np.ones(len(maes_a))])
    y_scores = np.concatenate([maes_n, maes_a])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J Index: J = Sensitivity + Specificity - 1
    # Specificity = 1 - FPR
    j_scores = tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # Current threshold heuristic (approximate mu + 3sigma)
    current_thr = np.mean(maes_n) + 3 * np.std(maes_n)
    
    # 5. Plotting ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'Optimal (Youden J): {best_threshold:.4f}')
    
    # Plot current threshold point
    curr_fpr = np.sum(maes_n > current_thr) / len(maes_n)
    curr_tpr = np.sum(maes_a > current_thr) / len(maes_a)
    plt.scatter(curr_fpr, curr_tpr, color='green', marker='x', s=100, label=f'Current (3-sigma): {current_thr:.4f}')
    
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Báo động giả)'); plt.ylabel('True Positive Rate (Độ nhạy)')
    plt.title(f'ROC Curve - {phase_name} Phase')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_{phase_name.lower()}.png')
    plt.close()
    
    # 6. GENTLE Specific Analysis (Section 2.3)
    if phase_name == "GENTLE":
        print("Performing deep dive into GENTLE phase errors...")
        fn_indices = np.where((maes_a < best_threshold))[0]
        fp_indices = np.where((maes_n > best_threshold))[0]
        
        print(f"False Negatives (Missed): {len(fn_indices)}")
        print(f"False Positives (False Alarms): {len(fp_indices)}")
        
        # Analyze energy levels of missed anomalies
        missed_energy = np.mean(X_anom_raw[fn_indices, 18]) # var_z
        detected_energy = np.mean(X_anom_raw[maes_a >= best_threshold, 18])
        print(f"Avg Var_Z of missed anomalies: {missed_energy:.6f}")
        print(f"Avg Var_Z of detected anomalies: {detected_energy:.6f}")

if __name__ == "__main__":
    for p in ['GENTLE', 'STRONG', 'SPIN']:
        analyze_phase(p)
    print("\n[OK] Statistical analysis finished. Plots saved as roc_*.png")
