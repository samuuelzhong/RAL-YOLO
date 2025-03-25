import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

dic_all = {}
with open(r'val_RTMM\miou.txt','r') as f:
    for line in f:
        line = line.strip()
        line = line.split(':')
        slope_type = line[0]
        mious = line[1].strip()
        mious = mious[1:-1].split(',')
        mious = list(map(float, mious))
        dic_all[slope_type] = mious


print(dic_all)
slope_types = {'0': 'Bare', '1':'Meadow', '2': 'Forest'}
colors = ['#653700', '#a2a415', '#76a973']
xtic = ['-30', '-20', '0', '20', '50']

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.color'] = 'gray'
fig = plt.figure(figsize=(8, 6), dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)
plt.yticks([])
plt.xticks([])

# 遍历 y 轴的每个刻度，并设置其可见性为 False
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线



for slope_type, miou in enumerate(dic_all.values()):
    print(slope_type, miou)
    slope_type = slope_type // 5
    color = colors[slope_type]

    x_stride = [i for i in range(1,len(miou)+1)]
    # 绘制网格、名称、条形

    plt.plot(x_stride, miou, color=color, linewidth=2, linestyle='-', marker='o', label=slope_types[str(slope_type)], alpha=0.5)
    for i, x in enumerate(x_stride):
        plt.annotate(
                    f'Seed_{xtic[i]}',
                    xy=(x, 0.7), xytext=(0, -10), textcoords='offset points', fontsize=22, color='gray', va='top', ha='center',zorder=4)

    for i, each_miou in enumerate(miou):
        plt.annotate(
            f'{each_miou:.3f}',
            xy=(x_stride[i], each_miou), xytext=(0, 2), textcoords='offset points', fontsize=22, color=color, va='bottom', ha='center',
            zorder=4)
        if i == 4:
            plt.annotate(
                slope_types[str(slope_type)],
                xy=(x_stride[i], each_miou), xytext=(20, 0), textcoords='offset points', fontsize=22,
                color=color, va='center', ha='left',
                zorder=4)

y_stride = [0.7+i*0.05 for i in range(7)]
for iy, y in enumerate(y_stride):
    plt.plot([0.5,5.5], [y,y], color='gray', linewidth=1.5, linestyle='-'if iy==0 else'--', zorder=1)
    plt.annotate(
        f'{y:.2f}',
        xy=(0.45, y), xytext=(0, 0), textcoords='offset points', fontsize=22, color='gray', va='center', ha='right',
        zorder=4)

polygon = patches.Polygon(
    [[0.5, y_stride[0]], [0.5, y_stride[-1]], [5.5, y_stride[-1]], [5.5, y_stride[0]]], closed=True,
    edgecolor=None, facecolor=(235 / 255.0, 235 / 255.0, 235 / 255.0), zorder=1, alpha=0.3)
ax.add_patch(polygon)


for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.ylim([0.65, 1.01])
plt.xlim([0, 6])
plt.annotate(
    'Number of Seed',
    xy=(3.7, 0.655), xytext=(0, 0), textcoords='offset points', fontsize=24, color=(80 / 255.0, 80 / 255.0, 80 / 255.0), va='center', ha='right',
    zorder=4, weight='bold')
plt.annotate(
    'mIoU',
    xy=(0.05, 0.85), xytext=(0, 0), textcoords='offset points', fontsize=24, color=(80 / 255.0, 80 / 255.0, 80 / 255.0), va='center', ha='right',
    zorder=4, weight='bold', rotation=90)
plt.title('Performance of Area Segmentation', fontsize=24, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold', pad=-10)
plt.savefig(rf'.\val_RTMM\mIoUs.tif', format='tiff')
plt.show()

