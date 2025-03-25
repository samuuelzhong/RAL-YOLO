from ultralytics.models import YOLO
import numpy as np
import os


if __name__ == '__main__':
    dic_roadap_all = {}
    dic_rockap_all = {}
    dic_speed = {}

    for i in ['8', '9', '10', '11']:
        roadap = []
        rockap = []
        speed = []

        if i == '8' or i == '10' or i == '11':
            for params in ['n', 's', 'm', 'l', 'x']:
                ivers = i + params
                yolover = rf'yolov{ivers}' if i != '11' else rf'yolo{ivers}'
                weight_path = rf"runs\train\{yolover}\weights\best.pt"

                model = YOLO(model=weight_path)
                val_result = model.val(data='./val.yaml')
                road_ap50_75 = np.mean(val_result.box.all_ap[0][0:6])
                rock_ap50_75 = np.mean(val_result.box.all_ap[1][0:6])

                roadap.append(road_ap50_75)
                rockap.append(rock_ap50_75)
                speed.append(val_result.speed['inference'])

        if i == '9':
            for params in ['t', 's', 'm', 'c', 'e']:
                ivers = i + params
                yolover = rf'yolov{ivers}' if i != '11' else rf'yolo{ivers}'
                weight_path = rf"runs\train\{yolover}\weights\best.pt"

                model = YOLO(model=weight_path)
                val_result = model.val(data='./val.yaml')
                road_ap50_75 = np.mean(val_result.box.all_ap[0][0:6])
                rock_ap50_75 = np.mean(val_result.box.all_ap[1][0:6])

                roadap.append(road_ap50_75)
                rockap.append(rock_ap50_75)
                speed.append(val_result.speed['inference'])

        yolonum = f'YOLOv{i}'
        dic_roadap_all[yolonum] = roadap
        dic_rockap_all[yolonum] = rockap
        dic_speed[yolonum] = speed
        print(roadap, rockap, speed)

    print(dic_roadap_all)
    print(dic_rockap_all)
    print(dic_speed)

    with open(r'val_SOTA_comparison\dic_roadap_all.txt', 'w') as f:
        for k, v in dic_roadap_all.items():
            f.write(f'{k}: {v}\n')

    with open(r'val_SOTA_comparison\dic_rockap_all.txt', 'w') as f:
        for k, v in dic_rockap_all.items():
            f.write(f'{k}: {v}\n')

    with open(r'val_SOTA_comparison\dic_speed_all.txt', 'w') as f:
        for k, v in dic_rockap_all.items():
            f.write(f'{k}: {v}\n')








