import numpy as np
import cv2
import os

def gain_polygon(txtfilepath):
    with open(txtfilepath,'r') as file:
        polygons = []
        for line in file:
            line = line.strip()
            line = line.split(' ')
            cls, poly = line[0], line[1:]
            polygons.append([cls,poly])

    return polygons

def plot_polygon(data_n, polygons, im, frame_num):
    texts_type = {'SWSR':{'0': 'Road', '1': 'Rock'},'MWRV':{'0': 'Rockfall', '1': 'Roadrock', '2': 'Rock'}}
    texts = texts_type[data_n]
    colors = {'0': (0,0,255), '1': (255,0,0), '2': (0,255,0)}
    for polygon in polygons:
        cls = polygon[0]
        poly = polygon[1]
        x, y, w, h = [int(float(i)*512) for i in poly]
        print([cls, x, y, w, h])
        im = cv2.rectangle(im, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), colors[cls], 2)
        im = cv2.putText(im, texts[cls], (x - w // 2, y - h // 2-3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[cls], 2)
    #if data_n == 'MWRV':
    #    im = cv2.putText(im, str(frame_num), (10, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return im

def gain_im(im_foler):
    im_dic = {}
    list_files = os.listdir(im_foler)
    for file_name in list_files:
        im_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(im_foler, file_name)
        im = cv2.imread(file_path)
        im_dic[im_name] = im

    return im_dic

if __name__ == '__main__':
    txtfilepath = r"E:\project\OD_fullmodel\ultralytics-main\for_multiple_light_yoloval\turf_-30\labels\val"
    im_foler = r"E:\project\OD_fullmodel\ultralytics-main\for_multiple_light_yoloval\turf_-30\images\val"
    data_n = ['SWSR', 'MWRV'][1]
    txt_lst = os.listdir(txtfilepath)
    im_dic = gain_im(im_foler)
    out_path = r'E:\project\OD_fullmodel\ultralytics-main\for_multiple_light_yoloval\label_visualize'
    for txt in txt_lst:
        name = os.path.splitext(txt)[0]
        txt_path = os.path.join(txtfilepath, txt)
        polygons = gain_polygon(txt_path)

        im = im_dic[name]
        #frame_num = name.split(' ')[1][1:-1]
        frame_num = name.split(' ')
        im = plot_polygon(data_n, polygons, im, frame_num)
        cv2.imwrite(os.path.join(out_path, name + '.png'), im)
