import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'runs\train\yolov8n\results.csv')

# Assuming df is a pandas DataFrame with the required columns and has at least 300 rows
# First, let's slice the DataFrame to get only the first 300 rows
df_300 = df.head(300)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.color'] = 'gray'
# Now, let's plot the data
fig = plt.figure(figsize=(5,5),dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)
color = {'box_loss':'#fedf08', 'cls_loss':'#5cac2d', 'dfl_loss':'#0a888a'}
# Plotting for 'train' metrics on the first subplot
plt.plot(df_300['train/box_loss'], label='train/box_loss', color=color['box_loss'])
plt.plot(df_300['train/cls_loss'], label='train/cls_loss', color=color['cls_loss'])
plt.plot(df_300['train/dfl_loss'], label='train/dfl_loss', color=color['dfl_loss'])
plt.title('Training Losses', weight='bold', fontsize=20,color=(80/255.0, 80/255.0, 80/255.0))
plt.grid(True)
plt.tick_params(axis='both', colors='gray', labelsize=18)
# 遍历 y 轴的每个刻度，并设置其可见性为 False
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
plt.xlim((0, 100))
plt.xlabel('Epoch', weight='bold', fontsize=20, color=(80/255.0, 80/255.0, 80/255.0))
plt.ylim((0, 4))
plt.ylabel('Loss', weight='bold', fontsize=20, color=(80/255.0, 80/255.0, 80/255.0))
polygon = patches.Polygon([[44, 2.75], [44, 4], [100, 4], [100, 2.75]], closed=True, edgecolor=(180 / 255.0, 180 / 255.0, 180 / 255.0), facecolor='white', zorder=3)
ax.add_patch(polygon)
plt.legend(fontsize=18, frameon=False, bbox_to_anchor=(0.41, 0.66), loc='lower left')
plt.savefig(r'yolov8n_loss\loss.tif', format='tiff')
plt.show()

fig = plt.figure(figsize=(5,5),dpi=100, constrained_layout=True)
ax = fig.add_subplot(111)
# Plotting for 'val' metrics on the second subplot
plt.plot(df_300['val/box_loss'], label='val/box_loss', color=color['box_loss'])
plt.plot(df_300['val/cls_loss'], label='val/cls_loss', color=color['cls_loss'])
plt.plot(df_300['val/dfl_loss'], label='val/dfl_loss', color=color['dfl_loss'])
plt.title('Validation Losses', weight='bold', fontsize=20,color=(80/255.0, 80/255.0, 80/255.0))
plt.grid(True)
plt.tick_params(axis='both', colors='gray', labelsize=18)
# 遍历 y 轴的每个刻度，并设置其可见性为 False
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线

for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)  # 主刻度线
    tick.tick2line.set_visible(False)  # 次刻度线
for spine in plt.gca().spines.values():
    spine.set_edgecolor('gray')
plt.xlim((0,100))
plt.xlabel('Epoch', weight='bold', fontsize=20, color=(80/255.0, 80/255.0, 80/255.0))
plt.ylim((0, 4))
plt.ylabel('Loss', weight='bold', fontsize=20, color=(80/255.0, 80/255.0, 80/255.0))
polygon = patches.Polygon([[44, 2.75], [44, 4], [100, 4], [100, 2.75]], closed=True, edgecolor=(180 / 255.0, 180 / 255.0, 180 / 255.0), facecolor='white', zorder=3)
ax.add_patch(polygon)
plt.legend(fontsize=18, frameon=False, bbox_to_anchor=(0.43, 0.66), loc='lower left')
plt.savefig(r'yolov8n_loss\loss2.tif', format='tiff')
plt.show()

