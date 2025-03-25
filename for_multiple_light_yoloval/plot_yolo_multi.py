import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties


with open(r'dic_test.txt','r') as f:

    '''dic format: [seed, map, mp, mr]'''

    dic_Bare = []
    dic_Meadow = []
    dic_Forest = []
    for i, line in enumerate(f):
        line = line.strip()
        line = line.split(':')
        slope_type, seeds = line[0].split('_')
        seeds = seeds[3:-1]
        seeds = '0' if seeds=='None' else seeds
        value = line[1].strip()
        map50, mp, mr = value[2:-1].split(',')
        ls = [seeds, map50, mp, mr]
        if i//5 == 0:
            dic_Bare.append(ls)
        if i//5 == 1:
            dic_Forest.append(ls)
        if i//5 == 2:
            dic_Meadow.append(ls)


#print(ls_all)
eva_types = ['mAP_50', 'mP', 'mR']
seed_c_ind = {'-30':0, '-20':1, '0':2, '20':3, '50':4}
slope_type = ['Bare', 'Meadow', 'Forest']

# 色带
colorsb = [(i/255, 80/255, 30/255) for i in range(80, 180, 20)]
colorst = [(i/255, 150/255, 10/255) for i in range(80, 180, 20)]
colorsp = [(i/255, 180/255, i/255) for i in range(80, 180, 20)]
colors = [colorsb, colorst, colorsp]

# y轴比例
yscale = 1
y_tic = [i*yscale for i in range(15)]


for ie, eva_name in enumerate(eva_types):
    sortedeva_Bare = sorted(dic_Bare, key=lambda x: x[ie+1], reverse=False)
    sortedeva_Meadow = sorted(dic_Meadow, key=lambda x: x[ie+1], reverse=False)
    sortedeva_Forest = sorted(dic_Forest, key=lambda x: x[ie+1], reverse=False)
    ls_all = [sortedeva_Bare, sortedeva_Meadow, sortedeva_Forest]

    print(sortedeva_Bare)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.color'] = 'gray'

    fig = plt.figure(figsize=(3,7),dpi=100,constrained_layout=True)
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


    for iy, y in enumerate(y_tic):

        '''
        y: 0 ~ 15
        sl_type 0: Bare, 1: Meadow, 2: Forest
        cont index 0~5
        '''

        sl_type, cont = iy//5, iy%5
        print(sl_type, cont)
        eva = float(ls_all[sl_type][cont][1:][ie])
        se_type = ls_all[sl_type][cont][0]
        color_ind = seed_c_ind[str(se_type)]
        ellipse = patches.Ellipse((eva, y), width=0.022 , height=0.3 , edgecolor=None,
                                  facecolor=colors[sl_type][color_ind], zorder=4, alpha=1)
        plt.plot([0.6, eva],[y, y], color=colors[sl_type][color_ind], linewidth=1.5,
                 linestyle='--', zorder=3, alpha=1)
        ax.add_patch(ellipse)

        plt.annotate(
            se_type,
            xy=(0.6, y), xytext=(-5, -2), textcoords='offset points', fontsize=20, color=colors[sl_type][color_ind], va='center',
            ha='right', zorder=5)
        plt.annotate(
            f'{eva:.3f}',
            xy=(eva, y), xytext=(8, -2), textcoords='offset points', fontsize=20, color=(40/255, 40/255, 40/255), va='center',
            ha='left', zorder=4)

        polygon = patches.Polygon(
            [[0.6, sl_type*5-0.4], [0.6, sl_type*5+5-0.6], [1, sl_type*5+5-0.6], [1, sl_type*5-0.4]], closed=True,
            edgecolor=None, facecolor=colors[sl_type][2], zorder=1, alpha=0.03)
        ax.add_patch(polygon)


    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xlim(0.6,1.1)
    plt.ylim(-1,16)
    plt.xlabel(eva_name, fontsize=24, weight='bold', color=(80/255, 80/255, 80/255), labelpad=-10)
    plt.savefig(rf'.\{eva_name}.tif', format='tiff')
    plt.show()

