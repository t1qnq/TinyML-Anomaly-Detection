import serial
import struct
import wave
import pandas as pd
import time
import os
import sys

# --- CẤU HÌNH ---
PORT = 'COM4'
BAUD = 921600
SAMPLE_RATE_AUDIO = 8000
DURATION = 5          
TOTAL_SAMPLES = 100    
DATASET_DIR = "dataset/normal"

# Số lượng mẫu chính xác cần đạt được
REQUIRED_AUDIO_SAMPLES = SAMPLE_RATE_AUDIO * DURATION # 40,000
REQUIRED_VIB_SAMPLES = 1000 * DURATION               # 5,000

def collect_sample(current_count, total_count):
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    
    existing = [f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')]
    idx = len(existing) + 1
    
    audio_path = f"{DATASET_DIR}/normal_{idx:03d}.wav"
    vib_path = f"{DATASET_DIR}/normal_{idx:03d}.csv"

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2) 
        ser.reset_input_buffer()

        audio_buffer, vib_buffer = [], []
        print(f"\n[MẪU {current_count}/{total_count}] -> {audio_path}")
        
        # Thu cho đến khi đủ số lượng mẫu audio quy định
        while len(audio_buffer) < REQUIRED_AUDIO_SAMPLES:
            if ser.read(1) == b'\xaa':
                if ser.read(1) == b'\xbb':
                    payload = ser.read(22)
                    if len(payload) == 22:
                        data = struct.unpack('<3h8h', payload)
                        
                        # Chuyển từ LSB sang đơn vị G (1G = 9.8m/s2)
                        # Ở chế độ +/- 2G, hệ số chia là 256.0
                        ax_g = data[0] / 256.0
                        ay_g = data[1] / 256.0
                        az_g = data[2] / 256.0
                        
                        vib_buffer.append([ax_g, ay_g, az_g])
                        audio_buffer.extend(data[3:])

        # Cắt lấy đúng số lượng mẫu yêu cầu (đề phòng dư)
        final_audio = audio_buffer[:REQUIRED_AUDIO_SAMPLES]
        final_vib = vib_buffer[:REQUIRED_VIB_SAMPLES]

        # Lưu Audio
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_AUDIO)
            wf.writeframes(struct.pack('<' + ('h' * len(final_audio)), *final_audio))
            
        # Lưu Rung động
        pd.DataFrame(final_vib, columns=['X', 'Y', 'Z']).to_csv(vib_path, index=False)
        
        print(f"\n   ✅ Đã xong mẫu {idx:03d}!")
        ser.close()
        return True

    except Exception as e:
        print(f"\n   ❌ Lỗi: {e}")
        if 'ser' in locals(): ser.close()
        return False

if __name__ == "__main__":
    print("=== HỆ THỐNG THU THẬP DỮ LIỆU TỰ ĐỘNG (100 MẪU) ===")
    for i in range(1, TOTAL_SAMPLES + 1):
        if not collect_sample(i, TOTAL_SAMPLES):
            print("Dừng do lỗi kết nối.")
            break
        time.sleep(0.5)
    print("\n Hoàn thành!")