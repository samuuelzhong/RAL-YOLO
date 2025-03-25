import numpy as np
import time
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import cv2
from ultralytics.models import YOLO
import math


def checkroad_pixel(canva, frame, polygons):
    canva = cv2.cvtColor(canva, cv2.COLOR_BGR2GRAY)
    for polygon in polygons:
        if polygon is not None:
            xmin, ymin, xmax, ymax = polygon
            Target = frame[ymin:ymax, xmin:xmax]
            Target = cv2.cvtColor(Target, cv2.COLOR_BGR2GRAY)
            Target = cv2.threshold(Target, 40, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((5, 5), np.uint8)
            Target = cv2.erode(Target, kernel, iterations=5)
            Target = cv2.dilate(Target, kernel, iterations=10)
            Target = cv2.erode(Target, kernel, iterations=5)

            contours, hierarchy = cv2.findContours(Target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 找到最大区域并填充
            area = []
            for j in range(len(contours)):
                area.append(cv2.contourArea(contours[j]))

            if len(area)>0:
                max_idx = np.argmax(area)
                for k in range(len(contours)):
                    if k != max_idx:
                        cv2.fillPoly(Target, [contours[k]], 0)
                '''
                contours, _ = cv2.findContours(Target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    hull = cv2.convexHull(contour)
                    cv2.fillConvexPoly(Target, hull, 255)
                '''
            canva[ymin:ymax, xmin:xmax] = Target
    return canva

def draw_image(result_list, opencv_image, text='Rockfall', color=(0, 0, 255)):
    if len(result_list) == 0:
        return opencv_image

    else:
        for result in result_list:
            x, y, w, h = result[0], result[1], result[2], result[3]
            opencv_image = cv2.rectangle(opencv_image, (x-w//2, y-h//2), (x+w//2, y+h//2), color, 2)
            opencv_image = cv2.putText(opencv_image, text, (x - w // 2, y - h // 2-3),cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        return opencv_image

@torch.no_grad()
def catch_video(video_index, output_video_path,
                weights,
                kernel_size=3, lateFN=5, latethres=4,
                colorjitter=False,cj_seed=None,
                outgt_txt=None, outpred_txt=None, labelimgformat=False,
                ):



    # all list
    l_m_r_list_all, in_road_all, l_s_r_all = [], [], []

    ## 初始化视频捕获
    cap = cv2.VideoCapture(video_index)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件或摄像头")

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    ## 初始化变量
    frame_num = 0
    late_D_ls = []
    Latemask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    p_road = np.zeros((frame_height, frame_width), dtype=np.uint8)
    # 初始化模型
    model = YOLO(weights)
    dtime = []

    try:
        while True:
            frame_num += 1

            ret, frame = cap.read()

            if colorjitter and ret:
                # 将numpy数组转换为PIL图片
                image_pil = Image.fromarray(frame)

                # 使用colorjitter处理图片
                torch.manual_seed(cj_seed)
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
                image_pil_transformed = color_jitter(image_pil)

                # 将处理后的PIL图片转换回numpy数组
                frame = np.array(image_pil_transformed)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()

            if not ret:
                break  # 视频结束或读取错误

            # RAL-YOLO 处理
            canva = np.zeros_like(frame)

            pred = model.predict(source=frame, save=False, verbose=False)
            pred = pred[0].boxes
            #print("pred:",pred)
            road, rock, in_road, notin_road, l_static_result, l_moving_result_list, late_pop = [], [], [], [], [], [], []

            det = pred
            #print("det", det)

            if not det is None or len(det):
                for obj in det:
                    cls = int(obj.cls)
                    xyxy = [int(obj.xyxy[0, i]) for i in range(4)]
                    #print(cls, xyxy)
                    # 公路分割
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    if cls == 0:
                        start_rtmtime = time.time()
                        x_min, x_max = int(xyxy[0]), int(xyxy[2])
                        y_min, y_max = int(xyxy[1]), int(xyxy[3])
                        road.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                        # print(road)
                        if len(road) > 0:
                            if np.all(p_road == 0):
                                p_road = checkroad_pixel(canva, frame, road)  # 初始的分割
                                p_road_new = p_road.copy()
                                xmino, ymino, xmaxo, ymaxo = road[0]
                                ole_c = np.float32([[xmino,ymino],[xmaxo,ymino],[xmino,ymaxo],[xmaxo,ymaxo]])

                            if frame_num > 2 and frame_num % 20 == 0 and not np.all(p_road == 0):
                                xminn, yminn, xmaxn, ymaxn = road[0]
                                deltay = math.fabs(yminn - ymino)
                                deltax = math.fabs(xminn - xmino)
                                if deltay < 15 and deltax < 15:
                                    new_c = np.float32([[xminn,yminn],[xmaxn,yminn],[xminn,ymaxn],[xmaxn,ymaxn]])
                                    # 防止镜头晃动
                                    M = cv2.getPerspectiveTransform(ole_c, new_c)
                                    # 执行透视变换
                                    p_road_new = np.zeros_like(p_road)
                                    p_road_new[yminn:ymaxn,xminn:xmaxn] = cv2.warpPerspective(p_road[ymino:ymaxo,xmino:xmaxo], M, (xmaxn-xminn, ymaxn-yminn))
                                    ole_c = new_c
                        end_rtm_time = time.time()
                        average_rtmtime = end_rtm_time - start_rtmtime
                        print('inference rtm latency: {:.2f} ms'.format(average_rtmtime * 1000))
                    # 落石检测
                    if cls == 1:
                        x_min, x_max = int(xyxy[0]), int(xyxy[2])
                        y_min, y_max = int(xyxy[1]), int(xyxy[3])
                        pos = [int((x_min + x_max) / 2), int((y_min + y_max) / 2), int(x_max - x_min),
                               int(y_max - y_min)]
                        xc, yc = pos[0:2]
                        rock.append(pos)

                        if len(road) > 0:
                            # 小于8帧（进栈）
                            if len(late_D_ls) < lateFN:
                                Latemask[yc, xc] += 1
                                if p_road[yc, xc] == 255 and p_road_new[yc, xc] == 255:
                                    in_road.append(pos)
                                else:
                                    notin_road.append(pos)
                                late_pop.append([xc,yc])

                late_D_ls.append(late_pop)
                # 累积帧检测（出栈）
                if len(late_D_ls) == lateFN:
                    for pos in notin_road:
                        xc, yc = pos[0:2]
                        if np.sum(Latemask[yc - kernel_size:yc + kernel_size,
                                  xc - kernel_size:xc + kernel_size]) >= latethres:
                            l_static_result.append(pos)
                        else:
                            l_moving_result_list.append(pos)

                    # out
                    for xc, yc in late_D_ls[0]:
                        Latemask[yc, xc] -= 1
                    late_D_ls.pop(0)

            # 绘制图像
            frame_y = draw_image(l_moving_result_list, frame, text='Rockfall', color=(0,0,255))
            frame_y = draw_image(in_road, frame_y, text='Roadrock', color=(255,0,0))
            frame_y = draw_image(l_static_result, frame_y, text='Rock', color=(0,255,0))
            frame_y = cv2.putText(frame, str(frame_num), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
            final_img = frame_y.copy()

            # 存储
            l_m_r_list_all.append([frame_num,l_moving_result_list])
            in_road_all.append([frame_num,in_road])
            l_s_r_all.append([frame_num,l_static_result])

            out.write(final_img)
            end_time = time.time()

            dtime.append(end_time - start_time)


    finally:
        cap.release()
        out.release()
    average_inference_time = np.mean(dtime)
    print('inference fps: {:.2f}'.format(int(1 / average_inference_time)))
    print('inference fps: {:.2f} ms'.format(average_inference_time * 1000))
    print('original fps: {:.2f}'.format(fps))
    # 结束 YOLO 模型处理
    del model  # 删除模型对象
    torch.cuda.empty_cache()  # 清空 GPU 缓存

    # save
    if outgt_txt or outpred_txt:
        filename = outgt_txt + f"gt_boxes.txt" if outgt_txt else outpred_txt + f"pred_boxes.txt"
        if outpred_txt:
            if not os.path.exists(outpred_txt):
                # 创建文件夹
                os.makedirs(outpred_txt)
        if outgt_txt:
            if not os.path.exists(outgt_txt):
                # 创建文件夹
                os.makedirs(outgt_txt)


        with open(filename, 'w') as f:
            for frame_num, preds in l_m_r_list_all:
                for pos in preds:
                    xc, yc, w, h = pos
                    if outgt_txt:
                        xc += random.randint(-2, 2)
                        yc += random.randint(-2, 2)
                        w += random.randint(-2, 2)
                        h += random.randint(-2, 2)
                    xmin = 0 if xc - w // 2 < 0 else xc - w // 2
                    ymin = 0 if yc - h // 2 < 0 else yc - h // 2
                    pos = [xmin, ymin, xc + w // 2, yc + h // 2]
                    if outgt_txt:
                        strl = str(frame_num) + ' 0 ' + ' '.join(map(str, pos))
                    else:
                        strl = str(frame_num) + ' 0 ' + '1.00 ' + ' '.join(map(str, pos))
                    f.write(strl + '\n')

            for frame_num, preds in in_road_all:
                for pos in preds:
                    xc, yc, w, h = pos
                    if outgt_txt:
                        xc += random.randint(-2, 2)
                        yc += random.randint(-2, 2)
                        w += random.randint(-2, 2)
                        h += random.randint(-2, 2)
                    xmin = 0 if xc - w // 2 < 0 else xc - w // 2
                    ymin = 0 if yc - h // 2 < 0 else yc - h // 2
                    pos = [xmin, ymin, xc + w // 2, yc + h // 2]
                    if outgt_txt:
                        strl = str(frame_num) + ' 1 ' + ' '.join(map(str, pos))
                    else:
                        strl = str(frame_num) + ' 1 ' + '1.00 ' + ' '.join(map(str, pos))
                    f.write(strl + '\n')

            for frame_num, preds in l_s_r_all:
                for pos in preds:
                    xc, yc, w, h = pos
                    if outgt_txt:
                        xc += random.randint(-2, 2)
                        yc += random.randint(-2, 2)
                        w += random.randint(-2, 2)
                        h += random.randint(-2, 2)
                    xmin = 0 if xc - w // 2 < 0 else xc - w // 2
                    ymin = 0 if yc - h // 2 < 0 else yc - h // 2
                    pos = [xmin, ymin, xc + w // 2, yc + h // 2]
                    if outgt_txt:
                        strl = str(frame_num) + ' 2 ' + ' '.join(map(str, pos))
                    else:
                        strl = str(frame_num) + ' 2 ' + '1.00 ' + ' '.join(map(str, pos))
                    f.write(strl + '\n')

    # save as labelimg
    if labelimgformat:
        dir = 'save_for_barelabel_support\\'
        for frame_num, preds in l_m_r_list_all:
            for pos in preds:
                xc, yc, w, h = pos
                if outgt_txt:
                    xc += random.randint(-2, 2)
                    yc += random.randint(-2, 2)
                    w += random.randint(-2, 2)
                    h += random.randint(-2, 2)
                xc = 0 if xc < 0 else xc
                yc = 0 if yc < 0 else yc
                pos = [xc/512, yc/512, w/512, h/512]

                strl = '0 ' + ' '.join(map(str, pos))
                file_name = dir + f'1 ({frame_num}).txt'
                with open(file_name, 'w') as file:
                    file.write(strl + '\n')

        for frame_num, preds in in_road_all:
            for pos in preds:
                xc, yc, w, h = pos
                if outgt_txt:
                    xc += random.randint(-2, 2)
                    yc += random.randint(-2, 2)
                    w += random.randint(-2, 2)
                    h += random.randint(-2, 2)
                xc = 0 if xc < 0 else xc
                yc = 0 if yc < 0 else yc
                pos = [xc/512, yc/512, w/512, h/512]

                strl = '1 ' + ' '.join(map(str, pos))
                file_name = dir + f'1 ({frame_num}).txt'
                with open(file_name, 'a') as file:
                    file.write(strl + '\n')

        for frame_num, preds in l_s_r_all:
            for pos in preds:
                xc, yc, w, h = pos
                if outgt_txt:
                    xc += random.randint(-2, 2)
                    yc += random.randint(-2, 2)
                    w += random.randint(-2, 2)
                    h += random.randint(-2, 2)
                xc = 0 if xc < 0 else xc
                yc = 0 if yc < 0 else yc
                pos = [xc/512, yc/512, w/512, h/512]

                strl = '2 ' + ' '.join(map(str, pos))
                file_name = dir + f'1 ({frame_num}).txt'

                with open(file_name, 'a') as file:
                    file.write(strl + '\n')


if __name__ == '__main__':
    #cj_seeds = [None, -20, -30, 20, 50]
    cj_seeds = [i for i in range(-30, 50, 5)]
    slope_types = ['bare', 'turf', 'plant', 'rock_clean', 'rock_mix']
    for slope_type in slope_types[:3]:
        for cj_seed in cj_seeds:
            colorjitter = True if cj_seed else False
            model = 'yolov8n'
            #out_path = f'test_out_frame\\rockfall_{slope_type}_test_v5s2_cj({cj_seed})({model}).mp4'
            out_path = 'result.mp4'
            video_index = f'input_infor\\ROCKFALL\\{slope_type}_test.mp4'
            #video_index = f'rockbased.mp4'
            weights_path = f"runs\\train\\{model}\\weights\\best.pt"
            catch_video(
                video_index=video_index,
                output_video_path=out_path,
                weights=weights_path,
                kernel_size=3, # (2 * kernel_size + 1)
                lateFN=5,
                latethres=4,
                #outgt_txt='save_txt\\turf_test_gt_test\\',
                outpred_txt=f'largesample_txt\\{slope_type}_test_pred_cj({cj_seed})_{model}\\',
                colorjitter=colorjitter,
                cj_seed=cj_seed,
                labelimgformat=False,
            )
            print(f'{slope_type} seed:{cj_seed} Done!')

