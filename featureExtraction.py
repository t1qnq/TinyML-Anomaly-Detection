import librosa
import numpy as np
import pandas as pd
import glob
import os

# Cấu hình
DATA_DIR = "dataset/normal"
N_MELS = 13 # Số đặc trưng âm thanh

def get_features():
    all_features = []
    wav_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    
    for wav_path in wav_files:
        csv_path = wav_path.replace(".wav", ".csv")
        if not os.path.exists(csv_path): continue
        
        # Đọc dữ liệu
        y, sr = librosa.load(wav_path, sr=8000)
        df_vib = pd.read_csv(csv_path)
        
        # Chia thành 5 cửa sổ, mỗi cửa sổ 1 giây
        for i in range(5):
            # 1. Đặc trưng Âm thanh (MFE)
            start_a, end_a = i*8000, (i+1)*8000
            mels = librosa.feature.melspectrogram(y=y[start_a:end_a], sr=sr, n_mels=N_MELS)
            mels_db = librosa.power_to_db(mels, ref=np.max)
            audio_feat = np.mean(mels_db, axis=1)
            
            # 2. Đặc trưng Rung động (RMS & Variance)
            start_v, end_v = i*1000, (i+1)*1000
            v_win = df_vib.iloc[start_v:end_v]
            vib_feat = [
                np.sqrt(np.mean(v_win['X']**2)), np.var(v_win['X']),
                np.sqrt(np.mean(v_win['Y']**2)), np.var(v_win['Y']),
                np.sqrt(np.mean(v_win['Z']**2)), np.var(v_win['Z'])
            ]
            
            all_features.append(list(audio_feat) + vib_feat)
            
    # Lưu vào CSV
    cols = [f"mfe_{i}" for i in range(N_MELS)] + ["rms_x", "var_x", "rms_y", "var_y", "rms_z", "var_z"]
    pd.DataFrame(all_features, columns=cols).to_csv("train_features.csv", index=False)
    print("✅ Đã tạo xong train_features.csv")

get_features()