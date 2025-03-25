# 该文件将读取掩码跟石块转换为YOLO格式用于验证
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def normalize(s, scale=512):
    target = float(s) / scale
    return target

def trans(png_folder, squre_folder, gt_path, out_folder):
    png_name = os.listdir(png_folder)
    #统计行数
    with open(gt_path, 'r') as f:
        totalnum = sum(1 for _ in f)

    # 读取石头真实标签
    with open(gt_path, 'r') as f:
        rocks_dic,rock_ls = {},[]
        for line_number, rockline in enumerate(f):
            if line_number == 0:
                oldname = rockline.split(' ')[0]
            rockline = rockline.strip()
            namer = rockline.split(' ')[0]
            rockline = rockline.split(' ')[2:]
            rcxmin, rcymin, rcxmax, rcymax = list(map(normalize, rockline))
            rcxc, rcyc, rcw, rch = (rcxmin + rcxmax)/2, (rcymin + rcymax)/2, rcxmax - rcxmin, rcymax - rcymin
            if oldname == namer:
                rock_ls.append([rcxc, rcyc, rcw, rch])
                if line_number+1 == totalnum:
                    rocks_dic[oldname] = rock_ls
            else:
                rocks_dic[oldname] = rock_ls
                oldname = namer
                rock_ls = []
                rock_ls.append([rcxc, rcyc, rcw, rch])

    # 读取公路真实标签
    road_dic = {}
    for png in png_name:
        name = os.path.splitext(png)[0]
        squre_path = os.path.join(squre_folder, name) + '.txt'
        frame_num = int(name.split(' ')[1][1:-1])
        #print(frame_num)
        # 读取道路txt
        with open(squre_path, 'r') as rf:
            for line in rf:
                road = line.strip()
                road = road.split(' ')[1:]
                xc, yc, w, h = list(map(float, road))
        road_dic[frame_num] = [[xc, yc, w, h]]
    #print(rocks_dic,'\n')

    #print(road_dic)
    for png in png_name:
        name = os.path.splitext(png)[0]
        frame_num = int(name.split(' ')[1][1:-1])
        txt_path = os.path.join(out_folder, f'test{frame_num}') + '.txt'
        with open(txt_path, 'w') as f:
            road_p, rock_p = None, None
            if frame_num in road_dic:
                road_p = road_dic[frame_num]
            if str(frame_num) in rocks_dic:
                rock_p = rocks_dic[str(frame_num)]
            if road_p is not None:
                for xc, yc, w, h in road_p:
                    f.write('0' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n')
            if rock_p is not None:
                for xc, yc, w, h in rock_p:
                    f.write('1' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n')



for slope_type in ['bare', 'turf', 'plant']:
    gt_path = rf'E:\project\OD_fullmodel\ultralytics-main\input_infor\ROCKFALL\{slope_type}_test_gt\gt_boxes.txt'
    png_folder = rf'E:\project\RAL-YOLO\test\ROCKFALL\{slope_type}_png'
    squre_folder = rf'E:\project\RAL-YOLO\test\ROCKFALL\squre_{slope_type[0]}'
    seg_folder = rf'E:\project\RAL-YOLO\test\ROCKFALL\{slope_type}_segmentation'
    out_folder = rf'E:\project\OD_fullmodel\ultralytics-main\for_multiple_light_yoloval\labels{slope_type}'
    os.makedirs(out_folder, exist_ok=True)
    miou = trans(png_folder, squre_folder, gt_path, out_folder)
    print('Done')


