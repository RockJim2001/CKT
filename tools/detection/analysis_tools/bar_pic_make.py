import numpy as np
import matplotlib.pyplot as plt

scales = ['Small', 'Medium', 'Large']
baseline = np.array([15.43, 47.28, 27.73])
afpn = np.array([13.46, 51.79, 37.92])
full_model = np.array([15.94, 56.74, 37.48])

gain_afpn = afpn - baseline
gain_full = full_model - baseline

x = np.arange(len(scales))
width = 0.32

fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=300)

# 关键：白底
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 关键：低饱和论文配色
bars1 = ax.bar(x - width/2, gain_afpn, width,
               label='+AFPN vs Baseline',
               color='#4C78A8', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, gain_full, width,
               label='Full model vs Baseline',
               color='#F58518', edgecolor='black', linewidth=0.6)

# 关键：零线变浅变细
ax.axhline(0, color='#7A7A7A', linewidth=0.8)

# 不要网格
# ax.grid(...)

ax.set_xlabel('Scale group', fontsize=14)
ax.set_ylabel('Novel AP gain (%)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=13)
ax.tick_params(axis='y', labelsize=12)

# 关键：图例上移
ax.legend(frameon=False, ncol=2, loc='upper center',
          bbox_to_anchor=(0.5, 1.14), fontsize=12)

# 关键：标注位置分开处理
def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        xc = bar.get_x() + bar.get_width() / 2
        if h >= 0:
            ax.text(xc, h + 0.22, f'{h:+.2f}',
                    ha='center', va='bottom', fontsize=12)
        else:
            ax.text(xc, h - 0.22, f'{h:+.2f}',
                    ha='center', va='top', fontsize=12)

add_labels(bars1)
add_labels(bars2)

# 关键：给顶部和底部留白
ax.set_ylim(-3, 11.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig('scale_wise_novel_ap_gain_final.pdf', bbox_inches='tight')
plt.savefig('scale_wise_novel_ap_gain_final.png', dpi=1200, bbox_inches='tight')
plt.show()