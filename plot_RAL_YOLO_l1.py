import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

dic_all = {}
with open(r'val_RAL_YOLO\dic_test_p.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(':')
        slope_type, seeds = line[0].split('_')
        seeds = seeds[3:-1]
        seeds = '0' if seeds=='None' else seeds
        value = line[1].strip()
        map50, mp, mr = value[2:-1].split(',')
        type = slope_type + '_' + seeds
        dic_all[type] = [map50, mp, mr]

print(dic_all)
slope_types = {'0': 'bare', '1':'turf', '2': 'plant'}
seed_type = {'0': '-30', '1':'-20', '2': '0', '3': '20', '4': '50'}
colorsb = [(i/255, i/255, 60/255) for i in range(40, 250, 45)]
colorst = [(i/255, i/255, 120/255) for i in range(40, 220, 36)]
colorsp = [(i/255, i/255, 180/255) for i in range(40, 250, 45)]
colors = [colorsb, colorst, colorsp]
evaluate = ['Rockfall', 'Roadrock', 'Rock']

for slope_type in range(1):
    color = colors[slope_type+1]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.color'] = 'gray'

    fig = plt.figure(figsize=(10,5.5),dpi=100,constrained_layout=True)
    ax = fig.add_subplot(111)

    plt.xticks([])
    plt.yticks([])

    # 遍历 y 轴的每个刻度，并设置其可见性为 False
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)  # 主刻度线
        tick.tick2line.set_visible(False)  # 次刻度线

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)  # 主刻度线
        tick.tick2line.set_visible(False)  # 次刻度线

    scale = 0.8

    xax = [i  for i in range(2, 7, 1)]
    yax = [i * scale for i in range(1, 4, 1)]
    print(xax)
    print(yax)

    # 绘制网格、名称、条形
    for i, x in enumerate(xax):
        plt.plot([x, x], [1 * scale,3.6 * scale], color=color[i], linewidth=1.5, linestyle='--')
        polygon = patches.Polygon([[x-0.5, 3.6*scale], [x+0.5, 3.6*scale], [x+0.5, 3.90*scale], [x-0.5, 3.90*scale]], closed=True,
                                  edgecolor=None, facecolor=color[i], zorder=3, alpha=0.7)
        plt.annotate(
                    f'Seed: {seed_type[str(i)]}',
                    xy=(x, 3.6*scale), xytext=(0, 14), textcoords='offset points', fontsize=24, color='w', va='center', ha='center',zorder=4)
        ax.add_patch(polygon)

    for i, y in enumerate(yax):
        plt.plot([1.5, 6.5], [y,y], color=(180/255, 180/255, 180/255), linewidth=1.5, linestyle='--')
        plt.plot([0, 1], [y, y], color='w')
        plt.annotate(
                    evaluate[i],
                    xy=(1.55, y), xytext=(-11, 0), textcoords='offset points', fontsize=24, color='black', va='center', ha='right')

    a = 0.35
    for iy, y in enumerate(yax):
        for ix, x in enumerate(xax):
            print(iy)
            search = slope_types[str(slope_type)] + '_'+ seed_type[str(ix)]
            dic = dic_all[search]
            eva = float(dic[2-iy])
            ellipse = patches.Ellipse((x, y), width=1.1 * .7 *(eva ** 1.5), height=0.9 * .7 * (eva ** 1.5), edgecolor=None,
                                      facecolor=[(1-a)*1+a*i for i in color[ix]], zorder=4, alpha=0.6)
            ax.add_patch(ellipse)
            #ellipse = patches.Ellipse((x, y), width=1.1 * .7 * (eva ** 1.5), height=0.9 * .7 * (eva ** 1.5), edgecolor=color[ix],
            #                          facecolor=None, zorder=4, alpha=1, fill=False)
            #ax.add_patch(ellipse)

            plt.annotate(
                f'{eva:.3f}',
                xy=(x, y), xytext=(6, 2), textcoords='offset points', fontsize=24, color='black',
                va='bottom', ha='left', zorder=5)
            # 将圆添加到坐标轴上

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xlim(0.3,6.7)
    plt.ylim()

    plt.savefig(rf'.\val_RAL_YOLO\{slope_types[str(slope_type)]}m_aps.tif', format='tiff')
    plt.show()

