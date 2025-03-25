#"E:\project\OD_fullmodel\jpg_rockclean\rock_clean_00187.jpg"
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolov8n/weights/best.pt')  # select your model.pt path
    model.predict(source=r"E:\project\OD_fullmodel\jpg_rockmix\rock_mix_00123.jpg",
                  imgsz=320,
                  project='runs/detect/feature',
                  name='yolov8',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True,  # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )