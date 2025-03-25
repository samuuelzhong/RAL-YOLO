import matplotlib.pyplot as plt
import csv
import numpy as np

dic_params={'YOLOv5': [1.9, 7.2, 21.2, 46.5, 86.7],
            'YOLOv8': [3.1, 11.1, 25.9, 43.7, 68.2],
            'YOLOv9': [2.1, 7.3, 20.2, 25.6, 58.2],
            'YOLOv10': [2.3, 7.2, 15.4, 24.4, 29.5],
            'YOLOv11':[2.6, 9.4, 20.1, 25.3, 56.9],}

road_aps = {}
rock_aps = {}

with open(r"val_SOTA_comparison\dic_roadap_all.txt") as f:
    for line in f:
        line = line.split(':')
        road_ap = line[1].strip()
        road_ap = road_ap[1:-1].split(',')
        road_ap = list(map(float, road_ap))
        road_aps[line[0]] = road_ap

print(road_aps)

with open(r"val_SOTA_comparison\dic_rockap_all.txt") as f:
    for line in f:
        line = line.split(':')
        rock_ap = line[1].strip()
        rock_ap = rock_ap[1:-1].split(',')
        rock_ap = list(map(float, rock_ap))
        rock_aps[line[0]] = rock_ap

print(rock_aps)

# 指定输出文件名
filename = 'val_SOTA_comparison\\output.csv'

# 写入CSV文件
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入标题（可选）
    writer.writerow(['params', 'Road', 'Rock'])

    # 写入数据
    for key, value in dic_params.items():
        para = [i for i in value]
        print(para)
        road = [f'{i:.3f}' for i in road_aps[key]]
        rock = [f'{i:.3f}' for i in rock_aps[key]]
        combine = list(zip(para, road, rock))
        for row in combine:
            writer.writerow(row)


