import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
dic_params={'YOLOv5': [2, 7, 21, 46, 86],
            'YOLOv8': [3, 11, 25, 43, 68],
            'YOLOv9': [2, 7, 20, 25, 57],
            'YOLOv10': [2, 8, 16, 25, 31],
            'YOLOv11':[2, 9, 20, 25, 57],}

road_aps = {}
rock_aps = {}

with open(r".\val_SOTA_comparison\dic_roadap_all.txt") as f:
    for line in f:
        line = line.split(':')
        road_ap = line[1].strip()
        road_ap = road_ap[1:-1].split(',')
        road_ap = list(map(float, road_ap))
        road_aps[line[0]] = road_ap

print(road_aps)

with open(r".\val_SOTA_comparison\dic_rockap_all.txt") as f:
    for line in f:
        line = line.split(':')
        rock_ap = line[1].strip()
        rock_ap = rock_ap[1:-1].split(',')
        rock_ap = list(map(float, rock_ap))
        rock_aps[line[0]] = rock_ap

print(rock_aps)

colors = {'YOLOv5':'#8cffdb', 'YOLOv8':'#a24857', 'YOLOv9':'#e6daa6', 'YOLOv10':'#b7c9e2', 'YOLOv11':'#ffd8b1'}
ml_zorder = {'YOLOv8':5, 'YOLOv5':2, 'YOLOv9':3, 'YOLOv10':2, 'YOLOv11':1}
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.color'] = 'gray'

fig = plt.figure(figsize=(10,8),dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)

for i, (model_name, aps) in enumerate(road_aps.items()):
    plt.plot(dic_params[model_name], [ap*100 for ap in aps] , label=model_name, marker='o', color=colors[model_name], zorder=ml_zorder[model_name])

plt.title('Performance on RIRW Object Detection Dataset', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold', pad=10)
plt.xlabel('Number of Parameters(M)', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold')
plt.ylabel('AP50_75 of Road(%)', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold')

plt.xticks(np.arange(0, 90+1, 10),fontsize=26)
plt.yticks(np.arange(90, 100, 1), fontsize=26)
plt.xlim([0, 90])
plt.ylim([90, 100])
plt.tick_params(axis='both', colors='gray')

plt.grid(True)

# 遍历 y 轴的每个刻度，并设置其可见性为 False
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

polygon = patches.Polygon([[55, 90], [55, 94.5], [90, 94.5], [90, 90]], closed=True, edgecolor=(180 / 255.0, 180 / 255.0, 180 / 255.0), facecolor='white', zorder=3)
ax.add_patch(polygon)
plt.legend(fontsize=28, frameon=False, bbox_to_anchor=(0.62, -0.03), loc='lower left')
# 设置边界框颜色
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')

plt.savefig(r'.\val_SOTA_comparison\road_aps.tif', format='tiff')
plt.show()

fig = plt.figure(figsize=(10,8),dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)
for model_name, aps in rock_aps.items():
    plt.plot(dic_params[model_name], [ap*100 for ap in aps] , label=model_name, marker='o', color=colors[model_name], zorder=ml_zorder[model_name])

#for model_name, aps in rock_aps.items():
#    size = ['n', 's', 'm', 'l', 'x'] if model_name != 'YOLOv9' else ['t', 's', 'm', 'c', 'e']
#    for i , (x, y) in enumerate(zip(dic_params[model_name], [ap*100 for ap in aps])):
#        plt.annotate(
#            model_name[4:]+size[i],
#            xy=(x, y), xytext=(1, 1), textcoords='offset points', color=colors[model_name], fontsize=20, zorder=5
#        )

plt.title('Performance on RIRW Object Detection Dataset', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold', pad=10)
plt.xlabel('Number of Parameters(M)', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold')
plt.ylabel(u'AP50_75 of Rock(%)', fontsize=30, color=(80 / 255.0, 80 / 255.0, 80 / 255.0),
          fontweight='bold')


plt.xticks(np.arange(0, 90+1, 10),fontsize=26)
plt.yticks(np.arange(70, 90, 2), fontsize=26)
plt.xlim([0, 90])
plt.ylim([70, 90])
plt.tick_params(axis='both', colors='gray')

# 遍历 y 轴的每个刻度，并设置其可见性为 False
ax = plt.gca()
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

# 设置边界框颜色
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
plt.grid(True)
polygon = patches.Polygon([[55, 70], [55, 79], [90, 79], [90, 70]], closed=True, edgecolor=(180 / 255.0, 180 / 255.0, 180 / 255.0), facecolor='white', zorder=3)
ax.add_patch(polygon)
plt.legend(fontsize=28, frameon=False, bbox_to_anchor=(0.62, -0.03), loc='lower left')
plt.savefig(r'.\val_SOTA_comparison\rock_aps.tif', format='tiff')
plt.show()