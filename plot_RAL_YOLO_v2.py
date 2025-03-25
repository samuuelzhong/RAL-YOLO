import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

dic_all = {}
with open(r'val_RAL_YOLO\dic_test.txt', 'r') as f:
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



slope_types = {'0': 'bare', '1':'turf', '2': 'plant'}
slope_rtypes = ['Bare', 'Meadow', 'Forest']
seed_type = {'0': '-30', '1':'-20', '2': '0', '3': '20', '4': '50'}
colorsb = [(i/255, i/255, 60/255) for i in range(40, 250, 45)]
colorst = [(i/255, i/255, 120/255) for i in range(40, 250, 45)]
colorsp = [((255-i)/255, (255-i)/255, i/255) for i in range(40, 210, 30)]
colorsp.reverse()

color_r = ['#653700', '#a2a415', '#76a973']
colors = [colorsb, colorst, colorsp]
evaluate = ['mAP_50', 'mP', 'mR']


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.color'] = 'gray'

fig = plt.figure(figsize=(10,5.5),dpi=100,constrained_layout=True)
ax = fig.add_subplot(111)
plt.xticks([])
plt.yticks([])

for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)

yscale = 1.2
xscale = 0.8

xax = [i * xscale for i in range(3, 12, 1)]
yax = [i * yscale for i in range(1, 6, 1)]


# 绘制圆圈与对应位置的指标
for ix, x in enumerate(xax):
    for iy, y in enumerate(yax):
        seed = seed_type[str(iy)]
        slope = slope_types[str(ix%3)]
        search = slope + '_' + seed
        ls = list(map(float, dic_all[search]))
        eva = ls[ix//3]
        print(ls)
        ellipse = patches.Ellipse((x+0.3*(ix//3), y), width=0.9 * .7 * (eva ** 2.5), height=1.1 * .7 * (eva ** 2.5), edgecolor=None,
                                  facecolor=color_r[ix%3], zorder=3, alpha=0.3)
        ax.add_patch(ellipse)
        plt.annotate(
                    f'{eva:.3f}',
                    xy=(x+0.3*(ix//3), y), xytext=(0, -2), textcoords='offset points', fontsize=16, color='black', va='center', ha='center',zorder=4)


for ix, x in enumerate(xax):
    if ix in [1, 4, 7]:
        if ix < 7:
            plt.plot([ix // 3 * 2.68 + 4.5, 4.5 + (ix // 3) * 2.68], [1, 6.28],
                     color=(180 / 255, 180 / 255, 180 / 255), linewidth=1.5, linestyle='--')
        plt.annotate(
                    evaluate[(ix-1)//3],
                    xy=(x+0.3*(ix//3), 6.7), xytext=(0, 0), textcoords='offset points', fontsize=16, color=(100/255, 100/255, 100/255),
            va='center', ha='center', weight='bold')
        plt.annotate(
                    slope_rtypes[(ix-1)//3],
                    xy=(x+0.3*(ix//3), 0.4), xytext=(0, -2), textcoords='offset points', fontsize=16, color=color_r[ix//3],
            va='center', ha='left', weight='bold', alpha=0.5)
        ellipse = patches.Ellipse((x+0.3*(ix//3)-0.3, 0.4), width=0.9 * .7*0.6, height=1.1 * .7*0.6, edgecolor=None,
                                  facecolor=color_r[ix//3], zorder=3, alpha=0.3)
        ax.add_patch(ellipse)

a = 0.1
b = 0.7
for iy, y in enumerate(yax):
    #for ix, x in enumerate(xax):
        #polygon = patches.Polygon([[ix//3*2.68+2, y-0.5], [ix//3*2.68+2, y+0.5], [4.4+(ix//3)*2.68, y+0.5], [4.4+(ix//3)*2.68, y-0.5]], closed=True,
        #                          edgecolor=None, facecolor=[(1-a)*1+a*i for i in colorsp[iy]], zorder=1)
        #plt.plot([ix//3*2.68+2, 4.38+(ix//3)*2.68], [y+0.5,y+0.5], color=(180/255, 180/255, 180/255), linewidth=1)
        #ax.add_patch(polygon)
    plt.annotate(
        'Seed_'+seed_type[str(iy)],
        xy=(1.5, y), xytext=(0, -2), textcoords='offset points', fontsize=16, color=(100/255, 100/255, 100/255),
        va='center', ha='center', zorder=5, weight='bold')

#plt.annotate('Performance of RAL-YOLO on MWRV3 Dataset', xy=(2.5, 0), xytext=(0, -20), textcoords='offset points', fontsize=18, color=(100 / 255.0, 100 / 255.0, 100 / 255.0),
#          fontweight='bold')
'''
    search = slope_types[str(slope_type)] + '_'+ seed_type[str(iy)]
    dic = dic_all[search]
    eva = float(dic[2-iy])
    ellipse = patches.Ellipse((x, y), width=1.1 * .7 *(eva ** 1.5), height=0.9 * .7 * (eva ** 1.5), edgecolor=None,
                              facecolor=[(1-a)*1+a*i for i in colorst[iy]], zorder=4, alpha=0.6)
    ax.add_patch(ellipse)

    plt.annotate(
        f'{eva:.3f}',
        xy=(x, y), xytext=(6, 2), textcoords='offset points', fontsize=24, color='black',
        va='bottom', ha='left', zorder=5)
        # 将圆添加到坐标轴上
'''


for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlim(0.3,11)
plt.ylim(0,7)

plt.savefig(rf'.\val_RAL_YOLO\m_aps.tif', format='tiff')
plt.show()

