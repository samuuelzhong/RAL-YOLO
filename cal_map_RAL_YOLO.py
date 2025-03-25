#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np


try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# AP VOC2007
def cal_ap_VOC2007(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

# AP VOC2012
def cal_ap_VOC2012(rec, prec):
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
def voc_eval(det_data, gt_boxes, obj_type, iou, map_type):
    """
    Top level function that does the PASCAL VOC evaluation.
    assume det_data: [("imagename", "obj_type", confidenc, xmin, ymin, xmax, ymax), ...]
    assume gt_boxes: [("imagename", "obj_type", xmin, ymin, xmax, ymax), ...]
    obj_type: Category name
    [iou_thresh]: Overlap threshold (default = 0.5)
    [map_type]: To use VOC2007's 11 point AP or VOC2012 AP
    """
    # recs : {"imagename": [{"obj_type":"obj_type", "bbox":[xmin,ymin,xmax,ymax]}, ...]}
    recs = {}
    for item in gt_boxes:
        #print(item)
        if item[0] in recs.keys():
            recs[item[0]].append({"obj_type":item[1], "bbox":[item[2], item[3], item[4], item[5]]})
        else:
            recs[item[0]] = [{"obj_type":item[1], "bbox":[item[2], item[3], item[4], item[5]]}]
    
    # extract gt objects for this class
    class_recs = {}   # {"imagename" : {'bbox': [[xmin,ymin,xmax,ymax],...], 'det': [False,...]}}
    npos = 0    # gt box的数量
    for imagename in recs.keys():
        # [{"obj_type":"obj_type", "bbox":[xmin,ymin,xmax,ymax]}, ...], obj_type只有一类
        R = [obj for obj in recs[imagename] if obj['obj_type'] == obj_type]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det}

    # read dets
    image_ids = [x[0] for x in det_data if x[1] == obj_type]  # imagename list
    confidence = np.array([x[2] for x in det_data if x[1] == obj_type]) # 置信度list
    BB = np.array([x[3:] for x in det_data if x[1] == obj_type]) # [[xmin,ymin,xmax,ymax], ...]

    # sort by confidence
    sorted_ind = np.argsort(-confidence) # 置信度list索引按照置信度降序排序
    sorted_scores = np.sort(-confidence) # 置信度降序排序, 后续未使用此变量
    BB = BB[sorted_ind, :]  # bound box按照置信度降序排序
    image_ids = [image_ids[x] for x in sorted_ind] # imagename按照置信度降序排序

    # go down dets and mark TPs and FPs
    if map_type == "COCO":
        iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    else:
        iou_list = [iou]

    ap = 0.0
    mrec = 0.0
    mprec = 0.0  
    for iou_thresh in iou_list:
        #print(image_ids)
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            #print(d)
            #print(class_recs)
            if image_ids[d] in class_recs:
                R = class_recs[image_ids[d]]
                R["det"] = [False] * len(R["det"])
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute IoU
                    # 相交区域
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni    # IoU
                    ovmax = np.max(overlaps)   # 最大交并比
                    jmax = np.argmax(overlaps)

                # 如果最大IoU大于阈值
                if ovmax > iou_thresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        P = prec.reshape(1, prec.shape[0])
        R = rec.reshape(1, rec.shape[0])
        f1 = 2 * P * R / (P + R + np.finfo(np.float64).eps)
        i = smooth(f1.mean(0), 0.1).argmax()-1
        P, R, f1 = P[:,i], R[:, i], f1[:,i]
        mrec = mrec + rec[-1] / len(iou_list)
        mprec = mprec + prec[-1] / len(iou_list)
        if map_type == "VOC2007":
            p = cal_ap_VOC2007(rec, prec)
        else:
            p = cal_ap_VOC2012(rec, prec)
        ap = ap + p / len(iou_list)

    if map_type == "COCO":
        return sorted_scores, np.nan, np.nan, np.nan, mrec, mprec, ap

    return sorted_scores, tp, fp, npos-tp, R, P, ap

ls_preds_path = os.listdir(r'largesample_txt')
dic_test = {}

for folder in ls_preds_path:
    pred_path = os.path.join(r'largesample_txt', folder, 'pred_boxes.txt')
    slope_type, cjseed = folder.split('_')[0], folder.split('_')[3]
    gt_path = rf'input_infor\ROCKFALL\{slope_type}_test_gt\gt_boxes.txt'

    gt = []
    pred = []
    iou_list = [0.5]
    objecttype = { '0': 'Rockfall', '1': 'Roadrock', '2': 'Rock'}
    with open(pred_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = tuple(line.split(' '))
            pred.append((line[0], line[1], float(line[2]), int(line[3]), int(line[4]), int(line[5]), int(line[6])))

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = tuple(line.split(' '))
            line = (line[0], line[1], int(line[2]), int(line[3]), int(line[4]), int(line[5]))
            gt.append(line)

    ap50_75=[]
    apall=[]
    P50 = []
    R50 = []
    for i in ['0','1','2']:
        for t, iou in enumerate(iou_list):
            sorted_scores, tp, fp, _, r, p, ap = voc_eval(pred, gt, obj_type=i, iou=iou, map_type=None)
            #print(f'class:{objecttype[i]} ap_{int(iou*100)}:',ap,' ','P:',p,' ','R:',r)
            if t <= 5:
                ap50_75.append(ap)
            if t == 0:
                P50.append(p)
                R50.append(r)
        #print(f'class:{objecttype[i]} ap50_75:{np.mean(ap50_75)}')
        apall.append(np.mean(ap50_75))

    map50 = np.mean(apall)
    p50 = np.mean(P50)
    R50 = np.mean(R50)
    print(f'{folder} map50:{map50:.3f} all:{apall}')
    print(f'{folder} mP50:{p50:.3f}')
    print(f'{folder} mR50:{R50:.3f}')
    print('\n')

    area = slope_type + '_' +cjseed
    dic_test[area] = [map50, p50, R50]

with open(r'largesample_txt\dic_test.txt', 'w') as f:
    for k, v in dic_test.items():
        f.write(f'{k}: {v}\n')

