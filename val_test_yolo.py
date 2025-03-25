import numpy as np
from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO(model=r".\runs\train\yolov8n\weights\best.pt")
    for slopetype in ['bare','plant','turf'][-1:]:
        for seed in ['-30', '-20', '0', '20', '50'][-1:]:
            val_result = model.val(data=rf'E:\project\OD_fullmodel\ultralytics-main\for_multiple_light_yoloval\val_path\{slopetype}_{seed}.yaml')
            road_ap50_75 = np.mean(val_result.box.all_ap[0][0:6])
            rock_ap50_75 = np.mean(val_result.box.all_ap[1][0:6])









