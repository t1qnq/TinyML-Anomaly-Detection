import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'DejaVu Sans'

OUT_DIR = os.path.join('doc', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

# ====================================================
# FIG 1: Data Distribution (Raw vs Augmented)
# ====================================================
df = pd.read_csv('train_features_v6.csv')
X_raw = df.values
VAR_Z_THR1 = 0.105845
VAR_Z_THR2 = 0.386260

gentle = (X_raw[:, 18] < VAR_Z_THR1).sum()
strong = ((X_raw[:, 18] >= VAR_Z_THR1) & (X_raw[:, 18] < VAR_Z_THR2)).sum()
spin   = (X_raw[:, 18] >= VAR_Z_THR2).sum()
total  = len(df)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phan tich Phan phoi Du lieu Goc va sau Tang cuong Du lieu',
             fontsize=13, fontweight='bold')

colors = ['#3498DB', '#2ECC71', '#E67E22']
labels = ['GENTLE\n(Pha giat/ngam)', 'STRONG\n(Pha xa/dao long)', 'SPIN\n(Pha vat)']
raw_counts = [gentle, strong, spin]
aug_counts = [20000, 10000, 10000]

# (a) Raw
bars1 = axes[0].bar(labels, raw_counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.55)
axes[0].set_title('(a) Phan phoi Du lieu Goc (Raw Dataset)\nTong: {:,} mau'.format(total),
                  fontsize=11, fontweight='bold', pad=10)
axes[0].set_ylabel('So luong mau (Samples)', fontsize=11)
axes[0].set_ylim(0, 15500)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar, count in zip(bars1, raw_counts):
    pct = count / total * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                '{:,}\n({:.1f}%)'.format(count, pct),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# (b) Augmented
bars2 = axes[1].bar(labels, aug_counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.55)
axes[1].set_title('(b) Sau Tang cuong Du lieu (After Augmentation)\nTong: 40,000 mau',
                  fontsize=11, fontweight='bold', pad=10)
axes[1].set_ylabel('So luong mau (Samples)', fontsize=11)
axes[1].set_ylim(0, 25000)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
for bar, count in zip(bars2, aug_counts):
    pct = count / 40000 * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                '{:,}\n({:.1f}%)'.format(count, pct),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout(pad=2.5)
out1 = os.path.join(OUT_DIR, 'fig_data_distribution.png')
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] Saved:', out1)

# ====================================================
# FIG 2: Var-Z histogram (3 separate subplots)
# ====================================================
var_z = X_raw[:, 18]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Phan phoi Phuong sai Truc Z (Var_Z) - Phan loai Tri-State tren 14,000 mau',
             fontsize=13, fontweight='bold')

groups = [
    (var_z[var_z < VAR_Z_THR1],                                       '#3498DB', 'GENTLE',
     'Var_Z < {:.3f}  ({} mau)'.format(VAR_Z_THR1, (var_z < VAR_Z_THR1).sum())),
    (var_z[(var_z >= VAR_Z_THR1) & (var_z < VAR_Z_THR2)],             '#2ECC71', 'STRONG',
     '{:.3f} <= Var_Z < {:.3f}  ({} mau)'.format(VAR_Z_THR1, VAR_Z_THR2,
      ((var_z >= VAR_Z_THR1) & (var_z < VAR_Z_THR2)).sum())),
    (var_z[var_z >= VAR_Z_THR2],                                       '#E67E22', 'SPIN',
     'Var_Z >= {:.3f}  ({} mau)'.format(VAR_Z_THR2, (var_z >= VAR_Z_THR2).sum())),
]

thresholds = [
    [(VAR_Z_THR1, '#2C3E50', 'Nguong 1: {:.3f}'.format(VAR_Z_THR1))],
    [(VAR_Z_THR1, '#2C3E50', 'Nguong 1: {:.3f}'.format(VAR_Z_THR1)),
     (VAR_Z_THR2, '#8E44AD', 'Nguong 2: {:.3f}'.format(VAR_Z_THR2))],
    [(VAR_Z_THR2, '#8E44AD', 'Nguong 2: {:.3f}'.format(VAR_Z_THR2))],
]

for i, (ax, (data, color, name, subtitle), thr_list) in enumerate(zip(axes, groups, thresholds)):
    ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    for thr_val, thr_color, thr_label in thr_list:
        ax.axvline(thr_val, color=thr_color, linestyle='--', linewidth=2, label=thr_label)
    ax.set_title('({}) {}\n{}'.format(chr(97+i), name, subtitle), fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Phuong sai Truc Z (Var_Z)', fontsize=10)
    ax.set_ylabel('So luong mau' if i == 0 else '', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
out2 = os.path.join(OUT_DIR, 'fig_varZ_tristate.png')
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] Saved:', out2)

