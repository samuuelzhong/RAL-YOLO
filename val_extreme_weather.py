from ultralytics.models import YOLO
from ultralytics import RTDETR
import numpy as np
import os

if __name__ == '__main__':
    weight_path = rf"E:\project\OD_fullmodel\ultralytics-main\runs\detect\rtdetr-l\weights\best.pt"
    model = RTDETR(model=weight_path)
    val_result = model.val(data='./val.yaml')
    road_ap50_75 = np.mean(val_result.box.all_ap[0][0:6])
    rock_ap50_75 = np.mean(val_result.box.all_ap[1][0:6])
    print(road_ap50_75, rock_ap50_75)
