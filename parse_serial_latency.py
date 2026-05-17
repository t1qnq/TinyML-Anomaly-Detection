# ============================================================
#  parse_serial_latency.py
#  Parse Serial output từ benchmark_latency.ino
#  Vẽ histogram phân phối latency cho luận văn
#
#  Cách dùng:
#    1. Copy toàn bộ output Serial vào file serial_output.txt
#    2. python parse_serial_latency.py --input serial_output.txt
#    -> Output: latency_histogram.png + latency_stats.txt
# ============================================================

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def parse_serial(filepath):
    """Parse [CSV] lines từ Serial output."""
    data = defaultdict(list)
    stats = {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # Parse CSV: [CSV] MODEL,run,latency_us
            m = re.match(r'\[CSV\]\s*(\w+),(\d+),(\d+)', line)
            if m:
                model = m.group(1)
                latency = int(m.group(3))
                data[model].append(latency)

            # Parse STATS: [STATS] MODEL,mean,std,min,max,median,p95,p99
            m2 = re.match(r'\[STATS\]\s*(\w+),([\d.]+),([\d.]+),(\d+),(\d+),(\d+),(\d+),(\d+)', line)
            if m2:
                model = m2.group(1)
                stats[model] = {
                    'mean': float(m2.group(2)),
                    'std': float(m2.group(3)),
                    'min': int(m2.group(4)),
                    'max': int(m2.group(5)),
                    'median': int(m2.group(6)),
                    'p95': int(m2.group(7)),
                    'p99': int(m2.group(8)),
                }

    return data, stats

def plot_histogram(data, stats, output_path):
    """Vẽ histogram phân phối latency cho 3 models."""
    models = ['GENTLE', 'STRONG', 'SPIN']
    colors = ['#3498DB', '#2ECC71', '#E67E22']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('Phân phối Độ trễ Suy luận trên ESP32-S3 (N=1000/model)',
                 fontweight='bold', fontsize=14)

    for i, (model, color) in enumerate(zip(models, colors)):
        ax = axes[i]
        if model not in data or len(data[model]) == 0:
            ax.set_title(f'{model}\n(Không có dữ liệu)')
            continue

        vals = np.array(data[model])
        ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor='white')

        # Thống kê
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean={mean_val:.0f} µs')
        ax.axvline(mean_val + 2*std_val, color='orange', linestyle=':',
                   linewidth=1.5, label=f'+2σ={mean_val+2*std_val:.0f} µs')

        ax.set_title(f'{model}\n(N={len(vals)})', fontweight='bold')
        ax.set_xlabel('Latency (µs)')
        if i == 0:
            ax.set_ylabel('Tần suất')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Histogram saved: {output_path}")
    plt.show()

def print_latex_table(data):
    """In bảng LaTeX sẵn sàng paste vào luận văn."""
    print("\n=== BẢNG LATEX (copy vào chap4_sec3.tex) ===\n")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{Mean (µs)} & \\textbf{Std} & "
          "\\textbf{Min} & \\textbf{Max} & \\textbf{P95} & \\textbf{P99} \\\\ \\hline")

    for model in ['GENTLE', 'STRONG', 'SPIN']:
        if model in data and len(data[model]) > 0:
            vals = np.array(data[model])
            s = np.sort(vals)
            print(f"{model} & {np.mean(vals):.1f} & {np.std(vals):.1f} & "
                  f"{np.min(vals)} & {np.max(vals)} & "
                  f"{s[int(len(s)*0.95)]} & {s[int(len(s)*0.99)]} \\\\ \\hline")

    all_vals = np.concatenate([np.array(data[m]) for m in ['GENTLE','STRONG','SPIN']
                               if m in data and len(data[m])>0])
    if len(all_vals) > 0:
        s = np.sort(all_vals)
        print(f"\\textbf{{Tổng hợp}} & \\textbf{{{np.mean(all_vals):.1f}}} & "
              f"\\textbf{{{np.std(all_vals):.1f}}} & "
              f"\\textbf{{{np.min(all_vals)}}} & \\textbf{{{np.max(all_vals)}}} & "
              f"\\textbf{{{s[int(len(s)*0.95)]}}} & "
              f"\\textbf{{{s[int(len(s)*0.99)]}}} \\\\ \\hline")

    print("\\end{tabular}")
    print("\\caption[Bảng 4.1: Thống kê độ trễ suy luận]{Thống kê độ trễ suy luận "
          "trên chip ESP32-S3 (N=1000 mẫu mỗi mô hình)}")
    print("\\label{tab:latency_stats}")
    print("\\end{table}")

def main():
    parser = argparse.ArgumentParser(description='Parse benchmark latency output')
    parser.add_argument('--input', default='serial_output.txt',
                        help='Serial output file from benchmark_latency.ino')
    parser.add_argument('--output', default='latency_histogram.png',
                        help='Output histogram image')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        print("Hướng dẫn:")
        print("  1. Flash benchmark_latency.ino lên ESP32-S3")
        print("  2. Mở Serial Monitor (921600 baud)")
        print("  3. Copy toàn bộ output vào file serial_output.txt")
        print("  4. Chạy lại script này")
        return

    data, stats = parse_serial(args.input)

    total = sum(len(v) for v in data.values())
    print(f"[OK] Parsed {total} measurements from {len(data)} models")
    for model, vals in data.items():
        arr = np.array(vals)
        print(f"  {model}: N={len(vals)}, "
              f"mean={np.mean(arr):.1f}µs, "
              f"std={np.std(arr):.1f}µs, "
              f"min={np.min(arr)}, max={np.max(arr)}")

    plot_histogram(data, stats, args.output)
    print_latex_table(data)

if __name__ == "__main__":
    main()
