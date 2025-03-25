import numpy as np

late_D_ls = []
lateFN = 5
Latemask = np.zeros((512, 512), dtype=np.uint8)
gt_MCM_path = r'input_infor\ROCKFALL\gt_plant.txt'

late_dic = {}
frame_max = 0
with open(gt_MCM_path, 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(' ')
        frame, cls, *xyxy = list(map(int, line))
        if frame > frame_max:
            frame_max = frame

frame_ls = [[] for i in range(frame_max)]

with open(gt_MCM_path, 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(' ')
        frame, cls, *xyxy = list(map(int, line))
        if cls == 0 or cls == 2:
            frame_ls[frame-1].append([cls,xyxy])
            #print(frame_ls[frame-1],frame-1)

p_road = np.zeros_like(Latemask)
p_road_new = np.zeros_like(Latemask)

ps = []
kernel_size = 3
latethres = 3

l_static_result = []
l_moving_result_list = []

tprf = 0
fprf = 0
tpr = 0
fpr = 0
for i, frame in enumerate(frame_ls):
    late_pop = []
    p = []
    for cls, xyxy in frame:

        xc, yc = (xyxy[0] + xyxy[2])//2, (xyxy[1] + xyxy[3])//2
        if len(late_D_ls) < lateFN:
            Latemask[yc, xc] += 1
            late_pop.append([cls, [xc, yc]])
            p.append([cls,xyxy])

    late_D_ls.append(late_pop)
    ps.append(p)
    # 累积帧检测（出栈）
    if len(late_D_ls) == lateFN:

        for cls, xyxy in ps[-1]:
            xc, yc = (xyxy[0] + xyxy[2])//2, (xyxy[1] + xyxy[3])//2
            if np.sum(Latemask[yc - kernel_size:yc + kernel_size,
                      xc - kernel_size:xc + kernel_size]) >= latethres:
                l_static_result.append([str(i+1), '2', '1.00', str(xyxy[0]), str(xyxy[1]), str(xyxy[2]), str(xyxy[3])])
                if cls == 2:
                    tprf += 1
                else:
                    fprf += 1
            else:
                l_moving_result_list.append([str(i+1), '0', '1.00', str(xyxy[0]), str(xyxy[1]), str(xyxy[2]), str(xyxy[3])])
                if cls == 0:
                    tpr += 1
                else:
                    fpr += 1
            # 出栈
            for xc, yc in late_D_ls[0]:
                Latemask[yc, xc] -= 1
        late_D_ls.pop(0)

pred_MCM_path = r'val_MCM\pred_MCM_plant.txt'
with open(pred_MCM_path, 'w') as f:
    for s in l_static_result:
        f.write(' '.join(s) + '\n')
    for s in l_moving_result_list:
        f.write(' '.join(s) + '\n')

import seaborn as sns
import matplotlib.pyplot as plt

confusion_matrix_data = np.zeros((2,2), dtype=int)
confusion_matrix_data[0][0] += tpr
confusion_matrix_data[0][1] += fpr
confusion_matrix_data[1][0] += fprf
confusion_matrix_data[1][1] += tprf
# 绘制混淆矩阵
# 设置字体大小
sns.set(font_scale=5)  # 比如 1.2 可以根据需要调整
# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 10))

sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Rock", "Rockfall"], yticklabels=["Rock", "Rockfall"])
# sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Forest')
# 保存图片，设置 dpi=600
plt.savefig(r".\val_MCM\MCM-plant_confusion_matrix.tif", dpi=600, bbox_inches='tight', format='tiff')
plt.show()
print('A_rf', tprf/(tprf+fprf),tprf)
print('A_r', tpr/(tpr+fpr))