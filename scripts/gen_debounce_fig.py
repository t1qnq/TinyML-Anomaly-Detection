# ============================================================
#  gen_debounce_fig.py
#  Sinh hình minh họa Sliding Window Debounce (Hysteresis)
#  cho Hình 4.1 trong luận văn Chương 4.
#
#  Chạy:
#    python scripts/gen_debounce_fig.py
#    -> Output: latex/chap4/image/debounce_sliding_window.png
#
#  Logic khớp firmware_v11.ino:
#    ALARM_WINDOW    = 10
#    ALARM_ENTER_THR = 5   (>=5 HIGH trong 10 win -> ALARM)
#    ALARM_EXIT_THR  = 1   (<=1 HIGH trong 10 win -> OK)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

# ----------------------------
# CONFIG (khớp firmware v11)
# ----------------------------
ALARM_WINDOW = 10
ALARM_ENTER_THR = 5
ALARM_EXIT_THR = 1
THRESHOLD = 0.0672  # Ngưỡng SPIN mode

N_FRAMES = 100
np.random.seed(42)

# ----------------------------
# 1. Sinh tín hiệu MAE giả lập
# ----------------------------
mae_signal = np.zeros(N_FRAMES)

# Pha 1: Bình thường (frame 0-18)
mae_signal[0:19] = np.random.normal(0.040, 0.005, 19)

# Spike ngẫu nhiên (frame 19-21): vượt ngưỡng nhưng ngắn
mae_signal[19] = 0.078
mae_signal[20] = 0.073
mae_signal[21] = 0.045

# Pha 2: Bình thường (frame 22-49)
mae_signal[22:50] = np.random.normal(0.042, 0.004, 28)

# Pha 3: Bất thường bền vững (frame 50-79)
mae_signal[50:55] = np.random.normal(0.055, 0.008, 5)  # Tăng dần
mae_signal[55:80] = np.random.normal(0.088, 0.012, 25)  # Vượt ngưỡng rõ

# Pha 4: Phục hồi (frame 80-99)
mae_signal[80:85] = np.random.normal(0.060, 0.008, 5)   # Giảm dần
mae_signal[85:100] = np.random.normal(0.040, 0.005, 15)  # Về bình thường

# Clip về dải hợp lý
mae_signal = np.clip(mae_signal, 0.025, 0.115)

# ----------------------------
# 2. Mô phỏng Sliding Window + Hysteresis
# ----------------------------
over_thr = mae_signal > THRESHOLD  # Boolean: từng frame có vượt ngưỡng không

ring_buffer = [False] * ALARM_WINDOW
ring_idx = 0
in_alarm = False

high_counts = np.zeros(N_FRAMES, dtype=int)
alarm_state = np.zeros(N_FRAMES, dtype=bool)

for i in range(N_FRAMES):
    # Ghi vào ring buffer
    ring_buffer[ring_idx] = over_thr[i]
    ring_idx = (ring_idx + 1) % ALARM_WINDOW

    # Đếm số HIGH trong 10 win gần nhất
    high_count = sum(ring_buffer)
    high_counts[i] = high_count

    # Hysteresis logic (khớp firmware)
    if not in_alarm and high_count >= ALARM_ENTER_THR:
        in_alarm = True
    elif in_alarm and high_count <= ALARM_EXIT_THR:
        in_alarm = False

    alarm_state[i] = in_alarm

# ----------------------------
# 3. Vẽ hình 2 panel
# ----------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 7.5),
    gridspec_kw={'height_ratios': [1.2, 1]},
    sharex=True
)
fig.suptitle(
    'Cơ chế lọc nhiễu Sliding Window Debounce với Hysteresis bất đối xứng',
    fontweight='bold', fontsize=14, y=0.97
)

frames = np.arange(N_FRAMES)

# ── PANEL TRÊN: Tín hiệu MAE ──
# Tô vùng ALARM (nền đỏ nhạt)
for i in range(N_FRAMES):
    if alarm_state[i]:
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.12, color='red', linewidth=0)

# Tô vùng vượt ngưỡng (fill under)
ax1.fill_between(
    frames, mae_signal, THRESHOLD,
    where=(mae_signal > THRESHOLD),
    alpha=0.25, color='#FF6B6B', label='_nolegend_'
)

ax1.plot(frames, mae_signal, color='#2C3E50', linewidth=1.5,
         label='Tín hiệu MAE thực tế', zorder=3)
ax1.axhline(y=THRESHOLD, color='#E74C3C', linestyle='--', linewidth=2,
            label=f'Ngưỡng báo động τ = {THRESHOLD}')

# Đánh dấu spike ngẫu nhiên
ax1.annotate('Spike nhiễu\n(lọc thành công)',
             xy=(20, mae_signal[20]), xytext=(28, 0.095),
             fontsize=9, ha='center', color='#E67E22',
             arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E67E22'))

# Đánh dấu vào ALARM
alarm_entry = None
alarm_exit = None
for i in range(1, N_FRAMES):
    if alarm_state[i] and not alarm_state[i - 1]:
        alarm_entry = i
    if not alarm_state[i] and alarm_state[i - 1]:
        alarm_exit = i

if alarm_entry:
    ax1.axvline(x=alarm_entry, color='#C0392B', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.annotate(f'Kích hoạt ALARM\n(frame {alarm_entry})',
                 xy=(alarm_entry, 0.105), xytext=(alarm_entry - 12, 0.110),
                 fontsize=9, ha='center', color='#C0392B',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor='#C0392B'))

if alarm_exit:
    ax1.axvline(x=alarm_exit, color='#27AE60', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.annotate(f'Thoát ALARM\n(frame {alarm_exit})',
                 xy=(alarm_exit, 0.060), xytext=(alarm_exit + 8, 0.095),
                 fontsize=9, ha='center', color='#27AE60',
                 arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', edgecolor='#27AE60'))

ax1.set_ylabel('Sai số tái tạo (MAE)', fontsize=12)
ax1.set_ylim(0.020, 0.120)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ── PANEL DƯỚI: Sliding Window Counter ──
# Tô vùng ALARM
for i in range(N_FRAMES):
    if alarm_state[i]:
        ax2.axvspan(i - 0.5, i + 0.5, alpha=0.12, color='red', linewidth=0)

# Đường high_count
ax2.plot(frames, high_counts, color='#2980B9', linewidth=2.0,
         label='Số lượng HIGH trong cửa sổ 10 win', zorder=3)
ax2.fill_between(frames, high_counts, alpha=0.15, color='#2980B9')

# Đường ngưỡng Entry/Exit
ax2.axhline(y=ALARM_ENTER_THR, color='#E74C3C', linestyle='--', linewidth=2,
            label=f'Ngưỡng vào ALARM (≥{ALARM_ENTER_THR}/10)')
ax2.axhline(y=ALARM_EXIT_THR, color='#27AE60', linestyle='--', linewidth=2,
            label=f'Ngưỡng thoát ALARM (≤{ALARM_EXIT_THR}/10)')

# Vẽ các chấm màu cho từng frame
for i in range(N_FRAMES):
    color = '#E74C3C' if over_thr[i] else '#27AE60'
    marker_size = 20 if over_thr[i] else 10
    ax2.scatter(i, high_counts[i], color=color, s=marker_size, zorder=4, alpha=0.7)

ax2.set_xlabel('Thời gian (Cửa sổ / Windows)', fontsize=12)
ax2.set_ylabel('Số HIGH trong 10 win', fontsize=12)
ax2.set_ylim(-0.5, 10.5)
ax2.set_yticks(range(0, 11, 2))
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Legend bổ sung cho chấm màu
red_dot = mpatches.Patch(color='#E74C3C', label='● Frame vượt ngưỡng (HIGH)')
green_dot = mpatches.Patch(color='#27AE60', label='● Frame bình thường (OK)')
alarm_bg = mpatches.Patch(color='red', alpha=0.12, label='█ Vùng trạng thái ALARM')
ax2.legend(
    handles=[ax2.get_legend_handles_labels()[0][0],
             ax2.get_legend_handles_labels()[0][1],
             ax2.get_legend_handles_labels()[0][2],
             red_dot, green_dot, alarm_bg],
    loc='upper left', fontsize=8, ncol=2
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ----------------------------
# 4. Lưu file
# ----------------------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_dir = os.path.join(repo_root, 'latex', 'chap4', 'image')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'debounce_sliding_window.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {out_path}")
plt.show()
