import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def read_frome_folder(folder_path):
    list_of_files = os.listdir(folder_path)
    dic_im = {}
    for file in list_of_files:
        if file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            im = cv2.imread(file_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            file_name = os.path.splitext(file)[0]
            dic_im[file_name] = im

    return dic_im

def selct_from_dicls(dicls, framerange):
    images = []
    for i in framerange:
        #print(dicls)
        images.append(dicls[str(i)])

    return images

def plotone(images_s, sloptype):

    x, y = len(images_s[0]),len(images_s)
    print(x,y)# 3行,5列
    fig, axes = plt.subplots(x, y, figsize=(10, 10*(x/y)))

    images = []
    for j in range(len(images_s[0])):
        for i in range(len(images_s)):
            images.append(images_s[i][j])
    # 将图片填充到每个子图中
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(f'val_RAL_YOLO\\RAL_YOLO_actual\\{sloptype}.tif', format='tiff', dpi=300)
    plt.show()


if __name__ == '__main__':

    '''2(ori,pred,gt后两者同行)x5(seed)x2(frame)x3(slopetype)'''

    pred_folders = r'E:\project\OD_fullmodel\ultralytics-main\val_RTMM'
    #ori_folder = r'E:\project\RAL-YOLO\test\ROCKFALL'
    plist_files = os.listdir(pred_folders)
    seed_index = {'0':0, '-30':1, '-20':2, '20':3, '50':4}
    sel = {'bare':[324,325,326], 'turf':[58,59,60], 'plant':[345,346,347]}

    bare_ls, turf_ls, plt_ls = [[]for i in range(5)], [[]for i in range(5)], [[]for i in range(5)]
    for file_name in plist_files:
        if file_name.endswith(".mp4"):
            pass
        else:
            slopetype, seed = file_name[0:4], seed[4:]
            print(slopetype, seed)
            seed = seed.split(')')[0][3:]
            seed = seed if seed != 'None' else '0'
            if slopetype == 'bare':
                pred_folder_path = os.path.join(pred_folders, file_name, 'RTMM')
                pred_folder_path = os.path.join(pred_folders, file_name, 'frame')
                dicimgs = read_frome_folder(pred_folder_path)
                #print(type(dicimgs))
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                print(len(sel_imgs))
                index = seed_index[seed]
                bare_ls[index] += sel_imgs

            elif slopetype == 'turf':
                pred_folder_path = os.path.join(pred_folders, file_name)
                dicimgs = read_frome_folder(pred_folder_path)
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                index = seed_index[seed]
                turf_ls[index] += sel_imgs

            else:
                slopetype, seed = file_name[0:5], seed[5:]
                pred_folder_path = os.path.join(pred_folders, file_name)
                dicimgs = read_frome_folder(pred_folder_path)
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                index = seed_index[seed]
                plt_ls[index] += sel_imgs

    plotone(bare_ls, 'bare')
    plotone(turf_ls, 'turf')
    plotone(plt_ls, 'plant')




