import cv2
from ultralytics.models import YOLO
from ultralytics import RTDETR
# 加载模型
#model = YOLO(f"runs\\train\\yolov8n\\weights\\best.pt")
model = RTDETR(r'E:\project\OD_fullmodel\ultralytics-main\runs\detect\train5\weights\best.pt')
# 打开视频文件
video_path = "rockleft.mp4"
cap = cv2.VideoCapture(video_path)
# 获取视频帧的维度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_locator.mp4', fourcc, 20.0, (frame_width, frame_height))
#循环视频帧
while cap.isOpened():
    # 读取某一帧
    success, frame = cap.read()
    if success:
        # 使用yolov8进行预测
        results = model(frame)
        #可视化结果
        annotated_frame = results[0].plot()
        #将带注释的帧写入视频文件
        out.write(annotated_frame)
    else:
        # 最后结尾中断视频帧循环
        break
#释放读取和写入对象
cap.release()
out.release()
