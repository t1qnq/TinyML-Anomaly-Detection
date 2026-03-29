# ============================================================
#  Data Collector - Python receiver
#  Nhan binary packet tu ESP32, luu WAV + CSV
#
#  Packet format (24 bytes struct, 22 bytes payload sau header):
#    [0xAA][0xBB]  header (2 bytes)
#    [ax][ay][az]  int16 x3 raw ADC (6 bytes) -> chia 256.0 = g
#    [s0..s7]      int16 x8 audio PCM sau HPF (16 bytes)
#
#  Chay:
#    python collect_data.py              # thu binh thuong
#    python collect_data.py --port COM5  # chi dinh port
# ============================================================

import serial
import struct
import wave
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse

# ============================================================
# CAU HINH
# ============================================================

PORT          = 'COM5'
BAUD          = 921600
SAMPLE_RATE   = 8000
DURATION      = 5              # giay moi mau
TOTAL_SAMPLES = 1800           # khong con dung, giu lai de tuong thich
DATASET_DIR   = "dataset_v7/normal"

# Truc trong luc (X = doc = huong xuan dat)
# Xac dinh bang cach do thuc te khi may dung yen:
#   truc nao doc duoc ~1g = GRAVITY_AXIS
GRAVITY_AXIS       = 'X'      # <- da xac nhan: truc X la trong luc
GRAVITY_EXPECT     = 1.0      # g
GRAVITY_TOL        = 0.15     # +-0.15g cho phep
GRAVITY_TILT_WARN  = 0.10     # canh bao neu |g_total - 1.0| > nguong nay

REQUIRED_AUDIO = SAMPLE_RATE * DURATION   # 40000 samples
REQUIRED_VIB   = 1000 * DURATION          # 5000 rows (1000 Hz x 5s)

PACKET_HEADER  = b'\xaa\xbb'
PACKET_FORMAT  = '<3h8h'   # 3 int16 vib + 8 int16 audio = 22 bytes payload

MIN_AUDIO_RMS  = 10.0     # LSB - nguong am thanh toi thieu
MAX_AUDIO_RMS  = 3000.0   # LSB - nguong canh bao clipping
IDLE_MODE      = False    # True khi thu idle (may tat)

# ============================================================
# PACKET READER
# ============================================================

def find_packet(ser):
    """Tim header 0xAA 0xBB, doc payload 22 bytes."""
    while True:
        b = ser.read(1)
        if b == b'\xaa':
            b2 = ser.read(1)
            if b2 == b'\xbb':
                payload = ser.read(22)
                if len(payload) == 22:
                    return struct.unpack(PACKET_FORMAT, payload)
    return None

# ============================================================
# COLLECT ONE SAMPLE
# ============================================================

def collect_sample(ser):
    """Thu 1 mau DURATION giay. Tra ve (ok, audio, vib) hoac (False, None, None)."""
    audio_buffer = []
    vib_buffer   = []

    ser.reset_input_buffer()
    t_start = time.time()

    while len(audio_buffer) < REQUIRED_AUDIO:
        data = find_packet(ser)
        if data is None:
            print("\n   [ERROR] Mat ket noi")
            return False, None, None

        # Vib: raw LSB -> g
        vib_buffer.append([data[0] / 256.0,
                            data[1] / 256.0,
                            data[2] / 256.0])
        # Audio: 8 samples int16
        audio_buffer.extend(data[3:])

        pct     = len(audio_buffer) / REQUIRED_AUDIO * 100
        elapsed = time.time() - t_start
        print(f"\r   Thu thap: {pct:.0f}% | {elapsed:.1f}s", end='', flush=True)

    print()
    return True, audio_buffer[:REQUIRED_AUDIO], vib_buffer[:REQUIRED_VIB]

# ============================================================
# QUALITY CHECK
# ============================================================

def check_quality(audio, vib):
    """
    Kiem tra chat luong mau.
    Tra ve (ok, warnings, stats).

    Kiem tra:
      1. Truc trong luc (GRAVITY_AXIS) phai ~1g
      2. |g_total| phai ~1g (thiet bi khong nghieng)
      3. Cac truc khac phai co rung dong (may dang chay)
      4. Audio phai co tin hieu
      5. Canh bao neu may dung yen (trong luc on dinh + khong rung)
    """
    warnings = []
    vib_arr  = np.array(vib)  # shape (N, 3): col 0=X, 1=Y, 2=Z

    ax = vib_arr[:, 0]
    ay = vib_arr[:, 1]
    az = vib_arr[:, 2]

    # Map ten truc -> index
    axis_map  = {'X': ax, 'Y': ay, 'Z': az}
    grav_data = axis_map[GRAVITY_AXIS]

    mean_x = float(np.mean(ax))
    mean_y = float(np.mean(ay))
    mean_z = float(np.mean(az))
    std_x  = float(np.std(ax))
    std_y  = float(np.std(ay))
    std_z  = float(np.std(az))
    rms_x  = float(np.sqrt(np.mean(ax**2)))
    rms_y  = float(np.sqrt(np.mean(ay**2)))
    rms_z  = float(np.sqrt(np.mean(az**2)))

    g_total = float(np.mean(np.sqrt(ax**2 + ay**2 + az**2)))

    grav_mean = float(np.mean(np.abs(grav_data)))

    # --- Check 1: Truc trong luc phai ~1g ---
    if abs(grav_mean - GRAVITY_EXPECT) > GRAVITY_TOL:
        warnings.append(
            f"Truc {GRAVITY_AXIS} mean={grav_mean:.3f}g "
            f"(expected {GRAVITY_EXPECT}+/-{GRAVITY_TOL}g) "
            f"-> thiet bi co the bi xoay!"
        )

    # --- Check 2: |g_total| ~1g ---
    if abs(g_total - 1.0) > GRAVITY_TILT_WARN:
        warnings.append(
            f"|g_total|={g_total:.3f}g (expected ~1.0g) "
            f"-> thiet bi bi nghieng hoac khong gan phang"
        )

    # NOTE: Khong check rung dong - gap qua hep giua may tat/chay
    # Nguoi dung tu kiem soat: chi thu khi may dang chay / khi thu idle
    non_grav_stds = {k: float(np.std(v))
                     for k, v in axis_map.items() if k != GRAVITY_AXIS}

    # --- Check 3: Audio RMS ---
    audio_arr = np.array(audio, dtype=np.float32)
    audio_rms = float(np.sqrt(np.mean(audio_arr**2)))
    if audio_rms < MIN_AUDIO_RMS:
        warnings.append(
            f"Audio qua nho (rms={audio_rms:.1f} < {MIN_AUDIO_RMS}) "
            f"-> kiem tra ket noi INMP441"
        )
    if audio_rms > MAX_AUDIO_RMS:
        warnings.append(
            f"Audio qua lon (rms={audio_rms:.1f} > {MAX_AUDIO_RMS}) "
            f"-> co the bi clipping, kiem tra vi tri dat mic"
        )

    stats = {
        'mean_x': mean_x, 'mean_y': mean_y, 'mean_z': mean_z,
        'std_x':  std_x,  'std_y':  std_y,  'std_z':  std_z,
        'rms_x':  rms_x,  'rms_y':  rms_y,  'rms_z':  rms_z,
        'g_total': g_total,
        'audio_rms': audio_rms,
    }

    ok = len(warnings) == 0
    return ok, warnings, stats

# ============================================================
# SAVE SAMPLE
# ============================================================

def save_sample(idx, audio, vib):
    os.makedirs(DATASET_DIR, exist_ok=True)
    audio_path = f"{DATASET_DIR}/sample_{idx:03d}.wav"
    vib_path   = f"{DATASET_DIR}/sample_{idx:03d}.csv"

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack('<' + 'h' * len(audio), *audio))

    pd.DataFrame(vib, columns=['X', 'Y', 'Z']).to_csv(vib_path, index=False)

# ============================================================
# PRINT STATS
# ============================================================

def print_stats(stats):
    """In stats vib ro rang de phat hien xoay truc ngay."""
    g   = GRAVITY_AXIS
    ng  = [a for a in ['X','Y','Z'] if a != g]
    print(f"   Trong luc ({g}): mean_{g.lower()}={stats[f'mean_{g.lower()}']:+.4f}g  "
          f"rms_{g.lower()}={stats[f'rms_{g.lower()}']:.4f}g  "
          f"std_{g.lower()}={stats[f'std_{g.lower()}']:.5f}")
    print(f"   Rung dong   : "
          + "  ".join(f"rms_{a.lower()}={stats[f'rms_{a.lower()}']:.4f}g "
                      f"std_{a.lower()}={stats[f'std_{a.lower()}']:.5f}"
                      for a in ng))
    print(f"   |g_total|={stats['g_total']:.4f}g  "
          f"audio_rms={stats['audio_rms']:.1f}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=PORT)
    parser.add_argument('--idle', action='store_true',
                        help='Thu data khi may tat (idle mode, bo check rung)')
    args = parser.parse_args()

    print("=" * 60)
    print("  THU THAP DU LIEU MAY GIAT (LIEN TUC)")
    print("=" * 60)
    print(f"  Port          : {args.port}")
    print(f"  Che do        : LIEN TUC - nhan Ctrl+C de dung")
    print(f"  Luu tai       : {DATASET_DIR}/")
    print(f"  Truc trong luc: {GRAVITY_AXIS} (expected ~{GRAVITY_EXPECT}g)")
    print()
    global IDLE_MODE
    IDLE_MODE = args.idle
    if IDLE_MODE:
        print("  [IDLE MODE] Thu data khi may TAT")
    print("TRUOC KHI BAT DAU, KIEM TRA:")
    print(f"  [1] Thiet bi gan CUNG vao hong may - truc {GRAVITY_AXIS} doc xuong dat")
    if IDLE_MODE:
        print(f"  [2] May giat PHAI TAT HOAN TOAN")
    else:
        print(f"  [2] May giat dang CHAY (dang giat/xa/vat)")
    print(f"  [3] ESP32 da flash data_collector.ino")
    print()
    input("Nhan Enter de bat dau...")
    print()

    try:
        ser = serial.Serial(args.port, BAUD, timeout=2)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[OK] Ket noi {args.port} @ {BAUD} baud\n")
    except Exception as e:
        print(f"[ERROR] Khong mo duoc {args.port}: {e}")
        sys.exit(1)

    # Dem file da ton tai
    existing = ([f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
                if os.path.exists(DATASET_DIR) else [])
    file_idx = len(existing) + 1
    if existing:
        print(f"[INFO] Da co {len(existing)} mau, tiep tuc tu #{file_idx}\n")

    success        = 0
    warned         = 0
    prev_grav_mean = None

    print("[INFO] Dang thu lien tuc. Nhan Ctrl+C de dung.\n")

    try:
        while True:
            print(f"[{success+1}] Mau #{file_idx}:")

            ok, audio, vib = collect_sample(ser)
            if not ok:
                time.sleep(1)
                continue

            quality_ok, warnings, stats = check_quality(audio, vib)
            print_stats(stats)

            # Phat hien thiet bi bi xoay giua cac mau
            curr_grav = stats[f'rms_{GRAVITY_AXIS.lower()}']
            if prev_grav_mean is not None:
                drift = abs(curr_grav - prev_grav_mean)
                if drift > 0.2:
                    print(f"   [WARN] Truc {GRAVITY_AXIS} thay doi {drift:.3f}g "
                          f"so voi mau truoc -> thiet bi co the bi xoay!")
            prev_grav_mean = curr_grav

            # In canh bao nhung van luu - khong dung lai hoi
            if not quality_ok:
                for w in warnings:
                    print(f"   [WARN] {w}")
                warned += 1

            save_sample(file_idx, audio, vib)
            success  += 1
            file_idx += 1
            print(f"   [OK] Luu mau #{file_idx-1} "
                  f"(tong: {success}, canh bao: {warned})\n")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[STOP] Nguoi dung dung.")

    finally:
        ser.close()
        print("=" * 60)
        print(f"  Ket qua: {success} mau da luu, {warned} mau co canh bao")
        print(f"  Du lieu: {DATASET_DIR}/")
        print()
        print("  Buoc tiep theo:")
        print("    python verify_dataset.py --data_dir", DATASET_DIR)
        print("=" * 60)


if __name__ == "__main__":
    main()