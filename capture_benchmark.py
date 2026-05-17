# ============================================================
#  capture_benchmark.py
#  Tự động đọc Serial từ ESP32-S3 và lưu vào file
#  Thay thế Serial Monitor của Arduino IDE
#
#  Cách dùng:
#    1. Flash benchmark_latency.ino lên ESP32-S3
#    2. ĐÓNG Serial Monitor (Arduino IDE) nếu đang mở
#    3. Chạy:
#       python capture_benchmark.py --port COM5
#    4. Đợi tự động hoàn tất → file serial_output.txt được tạo
#    5. Tự động chạy parse_serial_latency.py luôn (nếu có)
#
#  Yêu cầu: pip install pyserial
# ============================================================

import serial
import sys
import os
import time
import argparse
import subprocess

BAUD = 921600
TIMEOUT = 2        # Timeout đọc serial (giây)
MAX_WAIT = 120     # Tối đa đợi 120 giây
OUTPUT_FILE = "serial_output.txt"

def find_com_port():
    """Tự động tìm COM port của ESP32."""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    esp_ports = []
    for p in ports:
        desc = (p.description or "").lower()
        vid = p.vid or 0
        # XIAO ESP32S3 thường có VID 0x303A (Espressif) hoặc 0x1A86 (CH340)
        if vid in (0x303A, 0x1A86) or "esp" in desc or "cp210" in desc or "ch340" in desc:
            esp_ports.append(p.device)
        # Fallback: liệt kê tất cả
    if esp_ports:
        return esp_ports[0]
    if ports:
        return ports[0].device
    return None

def main():
    parser = argparse.ArgumentParser(description="Capture ESP32 benchmark Serial output")
    parser.add_argument("--port", default=None, help="COM port (e.g., COM5). Auto-detect nếu bỏ trống.")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--no-parse", action="store_true", help="Không tự động chạy parse sau khi capture")
    args = parser.parse_args()

    # Tìm port
    port = args.port
    if not port:
        port = find_com_port()
        if not port:
            print("[ERROR] Không tìm thấy COM port. Chỉ định bằng --port COMx")
            sys.exit(1)
    
    print("=" * 60)
    print("  CAPTURE BENCHMARK — TinyML Inference Latency")
    print("=" * 60)
    print(f"  Port   : {port}")
    print(f"  Baud   : {BAUD}")
    print(f"  Output : {args.output}")
    print(f"  Timeout: {MAX_WAIT}s")
    print()
    print("  LƯU Ý: Đóng Serial Monitor Arduino IDE trước khi chạy!")
    print()

    # Mở serial
    try:
        ser = serial.Serial(port, BAUD, timeout=TIMEOUT)
        time.sleep(0.5)
        ser.reset_input_buffer()
        print(f"[OK] Kết nối {port} @ {BAUD} baud")
    except serial.SerialException as e:
        print(f"[ERROR] Không mở được {port}: {e}")
        print("  → Kiểm tra: Arduino IDE Serial Monitor đã đóng chưa?")
        sys.exit(1)

    # Reset ESP32 bằng cách toggle DTR
    print("[INFO] Reset ESP32 để bắt đầu benchmark...")
    ser.dtr = False
    time.sleep(0.1)
    ser.dtr = True
    time.sleep(1.5)  # Đợi boot
    ser.reset_input_buffer()

    # Capture
    lines = []
    csv_count = 0
    done = False
    t_start = time.time()
    last_print = ""

    print("[INFO] Đang capture... (tự động dừng khi gặp [DONE])\n")

    try:
        while not done and (time.time() - t_start) < MAX_WAIT:
            try:
                raw = ser.readline()
                if not raw:
                    continue
                
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                lines.append(line)

                # Đếm CSV lines
                if line.startswith("[CSV]") and "model" not in line:
                    csv_count += 1
                    # Progress bar
                    if csv_count % 100 == 0:
                        elapsed = time.time() - t_start
                        pct = csv_count / 3000 * 100  # 3 models x 1000
                        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                        print(f"\r  [{bar}] {pct:.0f}% ({csv_count}/3000) | {elapsed:.0f}s", end="", flush=True)

                # In các dòng quan trọng
                if line.startswith("[BENCH]") or line.startswith("[STATS]") or line.startswith("[DONE]"):
                    print(f"\n  {line}")
                
                if line.startswith("[DONE]"):
                    done = True

            except UnicodeDecodeError:
                continue

    except KeyboardInterrupt:
        print("\n[STOP] Người dùng dừng.")

    finally:
        ser.close()

    elapsed = time.time() - t_start
    print(f"\n\n[OK] Capture hoàn tất: {len(lines)} dòng, {csv_count} measurements, {elapsed:.0f}s")

    # Lưu file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Đã lưu: {args.output} ({os.path.getsize(args.output):,} bytes)")

    if csv_count == 0:
        print("\n[WARN] Không có dữ liệu [CSV] nào!")
        print("  → Có thể ESP32 chưa flash benchmark_latency.ino")
        print("  → Hoặc benchmark chưa kịp chạy (thử tăng timeout)")
        return

    # Tự động chạy parse
    if not args.no_parse:
        parse_script = os.path.join(os.path.dirname(__file__), "parse_serial_latency.py")
        histogram_out = os.path.join("latex", "chap4", "image", "latency_histogram.png")
        
        if os.path.exists(parse_script):
            print(f"\n[AUTO] Đang chạy parse_serial_latency.py...")
            cmd = [sys.executable, parse_script, 
                   "--input", args.output, 
                   "--output", histogram_out]
            subprocess.run(cmd)
        else:
            print(f"\n[INFO] Chạy thủ công:")
            print(f"  python parse_serial_latency.py --input {args.output} --output {histogram_out}")

if __name__ == "__main__":
    main()
