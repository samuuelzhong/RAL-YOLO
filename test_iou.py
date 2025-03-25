import numpy as np

# 获取混淆矩阵
def fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)
def get_results(gts, ps, n_classes):
    """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    """
    hist = np.zeros((n_classes, n_classes))
    for ind in range(len(gts)):
        print(set(gts[ind].flatten()))
        print(set(ps[ind].flatten()))
        hist += fast_hist(gts[ind].flatten(), ps[ind].flatten(), n_classes)
        PA = per_class_PA_Recall(hist)
        iou = per_class_iu(hist)
        Precision = per_class_Precision(hist)

    mean_PA = np.nanmean(per_class_PA_Recall(hist))
    mean_IoU = np.nanmean(per_class_iu(hist))
    mean_Precision = np.nanmean(per_class_Precision(hist))
    Accuracy = per_Accuracy(hist)

    return mean_IoU, mean_PA, Accuracy, mean_Precision




