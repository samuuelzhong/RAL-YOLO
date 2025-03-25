import os

import numpy as np
import math
from test_iou import fast_hist, per_class_iu
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch

def checkroad_pixel(canva, frame, polygons):
    canva = cv2.cvtColor(canva, cv2.COLOR_BGR2GRAY)
    for polygon in polygons:
        if polygon is not None:
            xmin, ymin, xmax, ymax = polygon
            Target = frame[ymin:ymax, xmin:xmax]
            TargetGra = cv2.cvtColor(Target, cv2.COLOR_BGR2GRAY)
            TargetB = TargetGra.copy()
            TargetB = 255 - cv2.threshold(TargetGra, 190, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((5, 5), np.uint8)
            TargetE1 = cv2.erode(TargetB, kernel, iterations=5)
            TargetD = cv2.dilate(TargetE1, kernel, iterations=10)
            TargetE2 = cv2.erode(TargetD, kernel, iterations=5)

            contours, hierarchy = cv2.findContours(TargetE2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 找到最大区域并填充
            area = []
            for j in range(len(contours)):
                area.append(cv2.contourArea(contours[j]))

            TargetMax = TargetE2.copy()
            #TargetConv = TargetMax.copy()
            if len(area)>0:
                max_idx = np.argmax(area)
                for k in range(len(contours)):
                    if k != max_idx:
                        TargetMax = cv2.fillPoly(TargetMax, [contours[k]], 0)

                '''
                TargetConv = TargetMax.copy()
                contours, _ = cv2.findContours(TargetMax, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    hull = cv2.convexHull(contour)
                    TargetConv = cv2.fillConvexPoly(TargetConv, hull, 255)
                '''
            canva[ymin:ymax, xmin:xmax] = TargetMax
    return canva, TargetGra, TargetB, TargetE1, TargetD, TargetE2, TargetMax


def back2imsize(s, scale=512):
    target = float(s) * scale
    return int(round(target))



def val_RTMM(png_folder, squre_folder, seg_folder, slope_type, cj_seed=-20):
    png_name = os.listdir(png_folder)
    frame_num = 0
    miou = []
    for png in png_name:
        print(png)
        name = os.path.splitext(png)[0]
        frame_num = int(name.split(' ')[-1][1:-1])
        png_path = os.path.join(png_folder, name) + '.png'
        seg_path = os.path.join(seg_folder, name) + '.jpg'
        squre_path = os.path.join(squre_folder, name) + '.txt'
        frame = cv2.imread(png_path)
        if cj_seed:
            # 将numpy数组转换为PIL图片
            image_pil = Image.fromarray(frame)

            # 使用colorjitter处理图片
            torch.manual_seed(cj_seed)
            color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
            image_pil_transformed = color_jitter(image_pil)

            # 将处理后的PIL图片转换回numpy数组
            frame = np.array(image_pil_transformed)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with open(squre_path, 'r') as f:
            for line in f:
                road = line.strip()
                road = road.split(' ')[1:]
                roadsq = [list(map(back2imsize, road))]
                xc, yc, w, h = roadsq[0]
                xmin, ymin, xmax, ymax = xc-w//2, yc-h//2, xc+w//2, yc+h//2
                roadsq = [[xmin, ymin, xmax, ymax]]

        canva = np.zeros_like(frame)
        if frame_num <= 2:
            p_road, TargetGra, TargetB, TargetE1,TargetD, TargetE2, TargetMax = checkroad_pixel(canva, frame, roadsq)  # 初始的分割
            p_road_new = p_road.copy()
            height, width = p_road_new.shape
            xmino, ymino, xmaxo, ymaxo = roadsq[0]
            ole_c = np.float32([[xmino, ymino], [xmaxo, ymino], [xmino, ymaxo], [xmaxo, ymaxo]])
            #hist = fast_hist(p_road_new.flatten(), frame.flatten(), seg_folder)

        elif frame_num > 2:
            xminn, yminn, xmaxn, ymaxn = roadsq[0]
            deltay = math.fabs(yminn - ymino)
            deltax = math.fabs(xminn - xmino)
            if deltay < 15 and deltax < 15:
                new_c = np.float32([[xminn, yminn], [xmaxn, yminn], [xminn, ymaxn], [xmaxn, ymaxn]])

                # 防止镜头晃动
                M = cv2.getPerspectiveTransform(ole_c, new_c)
                # 执行透视变换
                p_road_new = np.zeros_like(p_road)
                trans = cv2.warpPerspective(p_road[ymino:ymaxo, xmino:xmaxo], M,
                                                    (xmaxn - xminn, ymaxn - yminn))
                o_size = p_road_new[yminn:ymaxn, xminn:xmaxn].shape
                t_size = trans.shape
                if t_size!=o_size:
                    trans = cv2.resize(trans, (o_size[1], o_size[0]))
                    #print(trans.shape)
                p_road_new[yminn:ymaxn, xminn:xmaxn] = trans
                ole_c = new_c

        gt_road_new = cv2.imread(seg_path, 0)
        gt = gt_road_new.copy()
        gt[gt>0]=1
        iou_p_road_new=p_road_new.copy()
        iou_p_road_new[iou_p_road_new>0]=1
        hist = fast_hist(iou_p_road_new.flatten(), gt.flatten(), 2)
        iou = np.mean(per_class_iu(hist))
        out = f'val_RTMM\\{slope_type}{cj_seed}\\frame'
        os.makedirs(out, exist_ok=True)
        out = f'val_RTMM\\{slope_type}{cj_seed}\\RTMM'
        os.makedirs(out, exist_ok=True)
        out = f'val_RTMM\\{slope_type}{cj_seed}\\GT'
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\frame\\test{frame_num}.jpg', frame)
        cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\RTMM\\test{frame_num}.jpg', p_road_new)
        cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\GT\\test{frame_num}.jpg', gt_road_new)
        if frame_num<=2:
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetGra.jpg', TargetGra)
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetB.jpg', TargetB)
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetE1.jpg', TargetE1)
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetD.jpg', TargetD)
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetE2.jpg', TargetE2)
            cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetMax.jpg', TargetMax)
            #cv2.imwrite(f'val_RTMM\\{slope_type}{cj_seed}\\TargetConv.jpg', TargetConv)

        miou.append(iou)
    miou = np.mean(miou)
    return miou

dic_all = {}
for slope_type in ['bare', 'turf', 'plant']:
    mious = []
    for cj_seed in [-30,-20,0,20,50]:
        png_folder = rf'E:\project\OD_fullmodel\ultralytics-main\input_infor\SEG\{slope_type}_png'
        squre_folder = rf'E:\project\OD_fullmodel\ultralytics-main\input_infor\SEG\squre_{slope_type[0]}'
        seg_folder = rf'E:\project\OD_fullmodel\ultralytics-main\input_infor\SEG\{slope_type}_segmentation'
        miou = val_RTMM(png_folder, squre_folder, seg_folder, slope_type, cj_seed)
        print('Done ',f'{slope_type}_{cj_seed} miou:',miou)
        mious.append(miou)
        dic_all[f'{slope_type}_{cj_seed}'] = mious

with open('val_RTMM\\miou.txt', 'w') as f:
    for slope_type, miou in dic_all.items():
        f.write(f'{slope_type}: {miou}\n')