import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def read_frome_folder(folder_path, selcframe):
    #print(selcframe)
    list_of_files = os.listdir(folder_path)
    dic_im = {}
    for file in list_of_files:
        if file.endswith(".png"):
            file_name = os.path.splitext(file)[0]
            if int(file_name) in selcframe:
                file_path = os.path.join(folder_path, file)
                im = cv2.imread(file_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
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
    fig, axes = plt.subplots(y, x, figsize=(10*(y/x), 10))

    images = []
    for i in range(len(images_s)):
        for j in range(len(images_s[i])):
            images.append(images_s[i][j])
    print(len(images))
    # 将图片填充到每个子图中
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(f'val_MWRV3\\MWRV3_actual\\{sloptype}.tif', format='tiff', dpi=300)
    plt.show()


if __name__ == '__main__':
    pred_folders = r'test_out_frame'
    #ori_folder = r'E:\project\RAL-YOLO\test\ROCKFALL'
    gt_folder = r'label_visualize\MWRV'
    plist_files = os.listdir(pred_folders)
    #orilist_files = os.listdir(ori_folder)
    gt_list_files = os.listdir(gt_folder)
    seed_index = {'-30':0, '-20':1, '0':2, '20':3, '50':4}
    sel = {'bare':[225,227,229,233,235], 'turf':[30,32,35,38,41], 'plant':[336,338,339,341,348]}

    bare_ls, turf_ls, plt_ls = [[]for i in range(5)], [[]for i in range(5)], [[]for i in range(5)]
    for file_name in plist_files:
        if file_name.endswith(".mp4"):
            pass
        else:
            all = file_name.split("_")
            print(file_name)
            slopetype, seed = all[1], all[4]
            print(slopetype, seed)
            seed = seed.split(')')[0][3:]
            seed = seed if seed != 'None' else '0'
            if slopetype == 'bare':
                pred_folder_path = os.path.join(pred_folders, file_name)
                dicimgs = read_frome_folder(pred_folder_path, sel[slopetype])
                #print(type(dicimgs))
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                print(len(sel_imgs))
                index = seed_index[seed]
                bare_ls[index] += sel_imgs

            elif slopetype == 'turf':
                pred_folder_path = os.path.join(pred_folders, file_name)
                dicimgs = read_frome_folder(pred_folder_path, sel[slopetype])
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                index = seed_index[seed]
                turf_ls[index] += sel_imgs

            elif slopetype == 'plant':
                pred_folder_path = os.path.join(pred_folders, file_name)
                dicimgs = read_frome_folder(pred_folder_path, sel[slopetype])
                sel_imgs = selct_from_dicls(dicimgs, sel[slopetype])
                index = seed_index[seed]
                plt_ls[index] += sel_imgs

    plotone(bare_ls, 'bare')
    plotone(turf_ls, 'turf')
    plotone(plt_ls, 'plant')





