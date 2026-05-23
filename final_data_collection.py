"""Thu đồng bộ mẫu âm thanh và rung động từ ESP32-S3.

Định dạng luồng Serial:
  header  : 0xAA 0xBB
  payload : 3 giá trị int16 từ ADXL345 + 8 mẫu int16 âm thanh

Mỗi mẫu được lưu gồm:
  - một file WAV mono 16-bit ở tần số 8 kHz
  - một file CSV chứa gia tốc X, Y, Z theo đơn vị g
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import serial


PACKET_HEADER = b"\xaa\xbb"
PACKET_FORMAT = "<3h8h"
PACKET_PAYLOAD_SIZE = struct.calcsize(PACKET_FORMAT)

AUDIO_SAMPLE_RATE_HZ = 8000
VIB_SAMPLE_RATE_HZ = 1000
SERIAL_BAUD = 921600
DEFAULT_DURATION_SEC = 5

GRAVITY_AXIS = "X"
GRAVITY_EXPECTED_G = 1.0
GRAVITY_TOLERANCE_G = 0.15
GRAVITY_TOTAL_TOLERANCE_G = 0.10

MIN_AUDIO_RMS = 10.0
MAX_AUDIO_RMS = 3000.0


def read_packet(ser: serial.Serial) -> tuple[int, ...]:
    """Đọc một gói hợp lệ từ Serial và trả về tuple giá trị đã giải mã."""
    while True:
        if ser.read(1) != PACKET_HEADER[:1]:
            continue
        if ser.read(1) != PACKET_HEADER[1:]:
            continue
        payload = ser.read(PACKET_PAYLOAD_SIZE)
        if len(payload) == PACKET_PAYLOAD_SIZE:
            return struct.unpack(PACKET_FORMAT, payload)


def collect_sample(
    ser: serial.Serial,
    duration_sec: int,
) -> tuple[list[int], list[list[float]]]:
    """Thu một cửa sổ dữ liệu từ board theo thời lượng cấu hình.

    Hàm tích lũy đủ số mẫu audio và số dòng rung cần thiết cho một sample,
    đồng thời chuyển giá trị ADXL345 từ LSB sang đơn vị g.
    """
    required_audio = AUDIO_SAMPLE_RATE_HZ * duration_sec
    required_vib = VIB_SAMPLE_RATE_HZ * duration_sec

    audio_buffer: list[int] = []
    vib_buffer: list[list[float]] = []

    ser.reset_input_buffer()
    start = time.time()

    while len(audio_buffer) < required_audio:
        packet = read_packet(ser)
        vib_buffer.append([packet[0] / 256.0, packet[1] / 256.0, packet[2] / 256.0])
        audio_buffer.extend(packet[3:])

        percent = len(audio_buffer) / required_audio * 100.0
        elapsed = time.time() - start
        print(f"\r  collecting: {percent:5.1f}% | {elapsed:4.1f}s", end="", flush=True)

    print()
    return audio_buffer[:required_audio], vib_buffer[:required_vib]


def quality_report(audio: list[int], vib: list[list[float]]) -> tuple[bool, list[str], dict[str, float]]:
    """Kiểm tra chất lượng một sample và trả về cảnh báo kèm thống kê số.

    Các kiểm tra chính gồm trục trọng lực, độ nghiêng tổng thể và mức RMS
    âm thanh để phát hiện sample quá nhỏ, quá lớn hoặc cảm biến đặt sai hướng.
    """
    warnings: list[str] = []
    vib_arr = np.asarray(vib, dtype=np.float32)
    audio_arr = np.asarray(audio, dtype=np.float32)

    x = vib_arr[:, 0]
    y = vib_arr[:, 1]
    z = vib_arr[:, 2]
    axis_values = {"X": x, "Y": y, "Z": z}
    gravity = axis_values[GRAVITY_AXIS]

    stats = {
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_z": float(np.mean(z)),
        "std_x": float(np.std(x)),
        "std_y": float(np.std(y)),
        "std_z": float(np.std(z)),
        "rms_x": float(np.sqrt(np.mean(x * x))),
        "rms_y": float(np.sqrt(np.mean(y * y))),
        "rms_z": float(np.sqrt(np.mean(z * z))),
        "g_total": float(np.mean(np.sqrt(x * x + y * y + z * z))),
        "audio_rms": float(np.sqrt(np.mean(audio_arr * audio_arr))),
    }

    gravity_mean = float(np.mean(np.abs(gravity)))
    if abs(gravity_mean - GRAVITY_EXPECTED_G) > GRAVITY_TOLERANCE_G:
        warnings.append(
            f"gravity axis {GRAVITY_AXIS}={gravity_mean:.3f}g, expected about 1g"
        )

    if abs(stats["g_total"] - 1.0) > GRAVITY_TOTAL_TOLERANCE_G:
        warnings.append(f"|g|={stats['g_total']:.3f}g, sensor may be tilted")

    if stats["audio_rms"] < MIN_AUDIO_RMS:
        warnings.append(f"audio RMS too low: {stats['audio_rms']:.1f}")
    if stats["audio_rms"] > MAX_AUDIO_RMS:
        warnings.append(f"audio RMS too high: {stats['audio_rms']:.1f}")

    return len(warnings) == 0, warnings, stats


def save_sample(out_dir: Path, index: int, audio: list[int], vib: list[list[float]]) -> None:
    """Lưu một sample thành cặp file WAV và CSV cùng chỉ số."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"sample_{index:04d}.wav"
    csv_path = out_dir / f"sample_{index:04d}.csv"

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SAMPLE_RATE_HZ)
        wf.writeframes(struct.pack("<" + "h" * len(audio), *audio))

    pd.DataFrame(vib, columns=["X", "Y", "Z"]).to_csv(csv_path, index=False)


def next_sample_index(out_dir: Path, start_index: int) -> int:
    """Tìm chỉ số sample kế tiếp để không ghi đè dữ liệu đã thu trước đó."""
    if not out_dir.exists():
        return start_index
    existing = sorted(out_dir.glob("sample_*.wav"))
    if not existing:
        return start_index
    last = max(int(path.stem.split("_")[-1]) for path in existing)
    return max(start_index, last + 1)


def print_stats(stats: dict[str, float]) -> None:
    """In thống kê ngắn gọn phục vụ kiểm tra nhanh chất lượng tín hiệu."""
    print(
        "  vib: "
        f"rms=({stats['rms_x']:.4f}, {stats['rms_y']:.4f}, {stats['rms_z']:.4f})g "
        f"std=({stats['std_x']:.5f}, {stats['std_y']:.5f}, {stats['std_z']:.5f}) "
        f"|g|={stats['g_total']:.4f} audio_rms={stats['audio_rms']:.1f}"
    )


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho quá trình thu dữ liệu qua Serial."""
    parser = argparse.ArgumentParser(description="Collect ESP32-S3 sensor data")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD)
    parser.add_argument("--out", default="dataset_v6/normal")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=0, help="0 means collect until Ctrl+C")
    return parser.parse_args()


def main() -> None:
    """Mở cổng Serial và lặp thu dữ liệu cho tới khi đủ số sample yêu cầu."""
    args = parse_args()
    out_dir = Path(args.out)
    index = next_sample_index(out_dir, args.start_index)

    print("TinyML data collection")
    print(f"  port     : {args.port}")
    print(f"  baud     : {args.baud}")
    print(f"  out      : {out_dir}")
    print(f"  duration : {args.duration}s/sample")
    print(f"  start    : sample #{index}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as exc:
        raise SystemExit(f"Cannot open {args.port}: {exc}") from exc

    saved = 0
    warned = 0
    try:
        time.sleep(2)
        while args.count == 0 or saved < args.count:
            print(f"\nSample #{index}")
            audio, vib = collect_sample(ser, args.duration)
            ok, warnings, stats = quality_report(audio, vib)
            print_stats(stats)
            for warning in warnings:
                print(f"  warning: {warning}")
            if not ok:
                warned += 1

            save_sample(out_dir, index, audio, vib)
            print(f"  saved: sample_{index:04d}.wav/csv")
            saved += 1
            index += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()

    print(f"\nDone. saved={saved}, warned={warned}, out={os.fspath(out_dir)}")


if __name__ == "__main__":
    main()
