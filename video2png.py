import os
import numpy as np
import cv2

def video2png(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件或摄像头")
    frame_num = 0
    try:
        while True:
            frame_num += 1
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(output_folder, "%d.png" % frame_num)
                cv2.imwrite(output_path, frame)
            else:
                break

    finally:
        cap.release()




video_folder = r"E:\project\OD_fullmodel\ultralytics-main\test_out_frame"
list_files = os.listdir(video_folder)
for file_name in list_files:
    if file_name.endswith(".mp4"):
        print(file_name)
        video_path = os.path.join(video_folder, file_name)
        file_name = os.path.splitext(file_name)[0]
        file_folder = os.path.join(video_folder, file_name)
        os.makedirs(file_folder, exist_ok=True)
        print(video_path,file_name)
        video2png(video_path,file_folder)
