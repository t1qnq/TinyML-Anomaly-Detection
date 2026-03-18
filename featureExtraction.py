# ============================================================
#  Feature Extraction
#  WAV + CSV -> feature vectors (19-dim)
#
#  Pipeline:
#    Audio: librosa.melspectrogram(center=False, hop=256, n_fft=512)
#           -> 30 frames -> power_to_db(ref=max) -> mean -> 13 features
#    Vib:   1000 samples/window -> rms + var per axis -> 6 features
#
#  Feature vector layout:
#    [0..12]  mel dB (13 bands)
#    [13][14] rms_x, var_x  (ADXL X = doc = trong luc ~1g)
#    [15][16] rms_y, var_y
#    [17][18] rms_z, var_z
#
#  Chay:
#    python feature_extraction.py
#    python feature_extraction.py --data_dir dataset/normal --out train.csv
# ============================================================

import os
import glob
import argparse
import numpy as np
import pandas as pd
import wave

# ============================================================
# CONFIG
# ============================================================

DATA_DIR  = "dataset_v6/normal"
OUT_CSV   = "train_features_v6.csv"
SR        = 8000
N_MELS    = 13
N_FFT     = 512
HOP_LEN   = 256    # khop firmware: HOP_SIZE=256, center=False -> 30 frames
NUM_FRAMES = 30    # khop firmware: NUM_FRAMES=30
WIN_SEC   = 1      # giay moi window
WINS_PER_FILE = 5  # so windows cat ra tu moi file 5s

AUDIO_PER_WIN = SR * WIN_SEC          # 8000 samples
VIB_PER_WIN   = 1000 * WIN_SEC        # 1000 rows (1000Hz x 1s)

# ============================================================
# WAV READER (tránh librosa/numba, match dữ liệu int16)
# ============================================================

def read_wav_int16_mono(path: str, expected_sr: int = SR) -> np.ndarray:
    """
    Đọc WAV mono 16-bit PCM bằng `wave` (nhanh, không cần librosa).
    Trả về ndarray float32 (giá trị theo thang int16, KHÔNG normalize về [-1,1]).
    """
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n  = wf.getnframes()
        if ch != 1:
            raise ValueError(f"WAV must be mono (channels={ch})")
        if sw != 2:
            raise ValueError(f"WAV must be 16-bit PCM (sampwidth={sw})")
        if sr != expected_sr:
            raise ValueError(f"WAV sr={sr} != expected {expected_sr}")
        raw = wf.readframes(n)
    y = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return y

# ============================================================
# MEL FEATURES
# ============================================================

_MEL_FB = (
    (1, 11, 21),
    (11, 21, 32),
    (22, 32, 42),
    (33, 43, 53),
    (43, 54, 64),
    (54, 64, 76),
    (65, 76, 90),
    (77, 91, 107),
    (91, 108, 128),
    (108, 128, 152),
    (129, 153, 181),
    (153, 181, 215),
    (182, 215, 255),
)


def _hann(n: int) -> np.ndarray:
    # Match firmware: hann[i] = 0.5*(1-cos(2*pi*i/(N-1)))
    i = np.arange(n, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (n - 1)))


_HANN_512 = _hann(N_FFT)


def _frame_30(seg: np.ndarray) -> np.ndarray:
    """
    Match firmware framing for 1s window:
    - 30 frames
    - Each frame uses 512 samples: [prev256, new256]
    - prev256 init = zeros (per window in this extractor)
    """
    seg = seg.astype(np.float32, copy=False)
    prev = np.zeros(HOP_LEN, dtype=np.float32)
    frames = np.empty((NUM_FRAMES, N_FFT), dtype=np.float32)
    for fr in range(NUM_FRAMES):
        start = fr * HOP_LEN
        new = seg[start : start + HOP_LEN]
        if len(new) < HOP_LEN:
            new = np.pad(new, (0, HOP_LEN - len(new)), mode="constant")
        frames[fr, :HOP_LEN] = prev
        frames[fr, HOP_LEN:] = new
        prev = new
    return frames


def _mel_power_from_rfft(rfft: np.ndarray) -> np.ndarray:
    """
    rfft: complex spectrum length (N_FFT/2 + 1)
    Return 13 band powers, matching firmware `mel_power()`.
    """
    # Power with energy doubling except DC/Nyquist, match firmware:
    # m2 = (re^2+im^2) * (2 for 0<k<N/2 else 1)
    re = rfft.real.astype(np.float32, copy=False)
    im = rfft.imag.astype(np.float32, copy=False)
    p = re * re + im * im
    if len(p) > 2:
        p[1:-1] *= 2.0

    out = np.empty(N_MELS, dtype=np.float32)
    for b, (lo, mid, hi) in enumerate(_MEL_FB):
        acc = 0.0
        # Rising slope: lo..mid (inclusive)
        if mid >= lo:
            denom = float(mid - lo) if mid > lo else 1.0
            for k in range(lo, min(mid, N_FFT // 2) + 1):
                w = float(k - lo) / denom if denom != 0.0 else 1.0
                acc += w * float(p[k])
        # Falling slope: mid+1..hi (inclusive)
        if hi >= mid + 1:
            denom = float(hi - mid) if hi > mid else 1.0
            for k in range(mid + 1, min(hi, N_FFT // 2) + 1):
                w = float(hi - k) / denom if denom != 0.0 else 0.0
                acc += w * float(p[k])
        out[b] = acc + 1e-10
    return out


def mel_features(seg: np.ndarray) -> np.ndarray:
    """
    13 mel features từ 1 giây audio (8000 samples).

    IMPORTANT: phiên bản này cố tình **match firmware_v7** (không dùng mel filterbank chuẩn của librosa).
    - FFT=512, hop=256, 30 frames
    - Hann window như firmware
    - Filterbank tam giác theo bảng bin `MEL_FB` trong firmware
    - power_to_db: 10*log10(power/mel_max), clamp >= -80dB, rồi mean theo 30 frames
    """
    frames = _frame_30(seg)  # (30, 512)
    frames = frames * _HANN_512[None, :]

    band_powers = np.empty((NUM_FRAMES, N_MELS), dtype=np.float32)
    mel_max = 0.0
    for fr in range(NUM_FRAMES):
        rfft = np.fft.rfft(frames[fr], n=N_FFT)
        bp = _mel_power_from_rfft(rfft)
        band_powers[fr] = bp
        m = float(np.max(bp))
        if m > mel_max:
            mel_max = m

    mel_max = max(mel_max, 1e-10)
    db = 10.0 * np.log10(band_powers / mel_max)
    db = np.maximum(db, -80.0)
    return np.mean(db, axis=0).astype(np.float32)


# ============================================================
# VIB FEATURES
# ============================================================

def vib_features(vib_df: pd.DataFrame) -> np.ndarray:
    """
    6 vib features tu 1000 rows CSV.

    Khop firmware v5.6:
      rms = sqrt(mean(x^2))
      var = mean(x^2) - mean(x)^2  (= np.var)
      Khong remap: X->X, Y->Y, Z->Z
    """
    def rms(a): return float(np.sqrt(np.mean(a**2)))
    def var(a): return float(np.var(a))

    x = vib_df['X'].values.astype(np.float32)
    y = vib_df['Y'].values.astype(np.float32)
    z = vib_df['Z'].values.astype(np.float32)

    return np.array([rms(x), var(x),   # [13][14]
                     rms(y), var(y),   # [15][16]
                     rms(z), var(z)])  # [17][18]


def vib_features_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Nhanh hơn: tính 6 vib features trực tiếp từ 3 vector float32."""
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    z = z.astype(np.float32, copy=False)

    def rms(a): return float(np.sqrt(np.mean(a * a)))
    def var(a): return float(np.var(a))

    feat = np.array([rms(x), var(x), rms(y), var(y), rms(z), var(z)], dtype=np.float32)

    # Clip outlier truoc khi luu - khop voi training.py va firmware
    # rms_z (index 4): cap 0.30g (pha vat manh)
    # var_x (index 1), var_y (index 3), var_z (index 5): clip tai p99
    #
    # QUAN TRONG: cac gia tri p99 duoi day phai khop voi gia tri
    # duoc in ra khi chay training.py (dong "Clip var_x/var_y/var_z cap=...")
    # Neu thu them data moi, chay training.py truoc de lay p99 moi, roi cap nhat o day.
    #
    # Gia tri hien tai tinh tu 10610 vectors (2122 files):
    VIB_CLIP_RMS_Z = 0.30         # rms_z cap co dinh
    VIB_CLIP_VAR_X = 0.001097     # var_x p99  <- cap nhat neu co data moi
    VIB_CLIP_VAR_Y = 0.000978     # var_y p99  <- cap nhat neu co data moi
    VIB_CLIP_VAR_Z = 0.060464     # var_z p99  <- cap nhat neu co data moi

    feat[4] = min(feat[4], VIB_CLIP_RMS_Z)
    feat[1] = min(feat[1], VIB_CLIP_VAR_X)
    feat[3] = min(feat[3], VIB_CLIP_VAR_Y)
    feat[5] = min(feat[5], VIB_CLIP_VAR_Z)

    return feat


# ============================================================
# EXTRACT ONE FILE
# ============================================================

def extract_file(wav_path: str, csv_path: str):
    """
    Trich xuat features tu 1 cap WAV+CSV.
    Tra ve list[np.ndarray(19,)] hoac (None, msg) neu loi.
    """
    # Load audio - WAV lưu int16 sau HPF (match firmware input scale)
    y = read_wav_int16_mono(wav_path, expected_sr=SR)
    df = pd.read_csv(csv_path, usecols=["X", "Y", "Z"], dtype=np.float32)

    # Kiem tra do dai
    if len(y) < AUDIO_PER_WIN * WINS_PER_FILE:
        return None, f"Audio ngan ({len(y)} samples < {AUDIO_PER_WIN*WINS_PER_FILE})"
    if len(df) < VIB_PER_WIN * WINS_PER_FILE:
        return None, f"Vib ngan ({len(df)} rows < {VIB_PER_WIN*WINS_PER_FILE})"

    features = []
    vib_x = df["X"].values
    vib_y = df["Y"].values
    vib_z = df["Z"].values

    for i in range(WINS_PER_FILE):
        seg = y[i * AUDIO_PER_WIN : (i+1) * AUDIO_PER_WIN]
        v0 = i * VIB_PER_WIN
        v1 = (i + 1) * VIB_PER_WIN
        x = vib_x[v0:v1]
        yv = vib_y[v0:v1]
        z = vib_z[v0:v1]

        if len(seg) < AUDIO_PER_WIN or len(x) < VIB_PER_WIN:
            break

        feat = np.concatenate([mel_features(seg), vib_features_xyz(x, yv, z)])
        features.append(feat)

    return features, None


# ============================================================
# QUALITY CHECK
# ============================================================

def check_features(df: pd.DataFrame):
    """In thong ke de xac nhan features hop le."""
    mel_cols = [f"mfe_{i}" for i in range(N_MELS)]
    vib_cols = ["rms_x","var_x","rms_y","var_y","rms_z","var_z"]

    print("\n--- Mel features (dB) ---")
    print(df[mel_cols].describe().round(2).to_string())

    print("\n--- Vib features ---")
    print(df[vib_cols].describe().round(6).to_string())

    # Kiem tra rms_x ~ 1g (trong luc truc X)
    rms_x_mean = df['rms_x'].mean()
    ok = abs(rms_x_mean - 1.0) < 0.15
    print(f"\n[{'OK' if ok else 'WARN'}] rms_x mean = {rms_x_mean:.4f}g "
          f"(expected ~1.0g, truc X = doc = trong luc)")

    # Kiem tra mel range
    mel_min = df[mel_cols].min().min()
    mel_max = df[mel_cols].max().max()
    ok = mel_min >= -80 and mel_max <= 0
    print(f"[{'OK' if ok else 'WARN'}] mel range = [{mel_min:.1f}, {mel_max:.1f}] dB "
          f"(expected [-80, 0])")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--out',      default=OUT_CSV)
    args = parser.parse_args()

    print("=" * 60)
    print("  FEATURE EXTRACTION")
    print("=" * 60)
    print(f"  Data : {args.data_dir}/")
    print(f"  Out  : {args.out}")
    print(f"  Config: n_fft={N_FFT} hop={HOP_LEN} n_mels={N_MELS} "
          f"center=False wins/file={WINS_PER_FILE}")
    print()

    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if not wav_files:
        print(f"[ERROR] Khong tim thay WAV trong {args.data_dir}/")
        return

    print(f"Tim thay {len(wav_files)} file WAV\n")

    all_features = []
    errors       = []

    for idx, wav_path in enumerate(wav_files, start=1):
        if idx % 100 == 0:
            print(f"  ... {idx}/{len(wav_files)} files")
        csv_path = wav_path.replace('.wav', '.csv')
        if not os.path.exists(csv_path):
            errors.append(f"{os.path.basename(wav_path)}: thieu CSV")
            continue

        feats, err = extract_file(wav_path, csv_path)
        if err:
            errors.append(f"{os.path.basename(wav_path)}: {err}")
            continue

        all_features.extend(feats)

    if errors:
        print(f"[WARN] {len(errors)} file loi:")
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... va {len(errors)-5} file khac")
        print()

    if not all_features:
        print("[ERROR] Khong co features nao duoc trich xuat!")
        return

    cols = ([f"mfe_{i}" for i in range(N_MELS)] +
            ["rms_x","var_x","rms_y","var_y","rms_z","var_z"])
    df = pd.DataFrame(all_features, columns=cols)
    df.to_csv(args.out, index=False)

    print(f"[OK] {args.out}: {len(df)} vectors "
          f"({len(wav_files)} files x {WINS_PER_FILE} windows)\n")

    check_features(df)

    print(f"\nBuoc tiep theo:")
    print(f"  Upload {args.out} len Kaggle va chay training.py")


if __name__ == "__main__":
    main()