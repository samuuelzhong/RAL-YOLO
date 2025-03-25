import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'E:\project\OD_fullmodel\ultralytics-main\runs\detect\train5\weights\last.pt')
    # model.load('') # loading pretrain weights
    model.train(data='./data.yaml',
                resume=True
                )
