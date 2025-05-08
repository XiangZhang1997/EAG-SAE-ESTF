import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from skimage import morphology
import cv2

def computeF1(pred, gt):

    gt = torch.where(gt > 0.0, 1, 0)
    pred_b = torch.where(pred >= 0.5, 1, 0)
    gt_b = gt

    tp = (gt_b * pred_b).sum()
    tn = ((1 - gt_b) * (1 - pred_b)).sum()
    fp = ((1 - gt_b) * pred_b).sum()
    fn = (gt_b * (1 - pred_b)).sum()

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)


    pred_auc = pred[0].detach().cpu().numpy().flatten()
    gt_auc = gt[0].detach().cpu().numpy().flatten()

    pred_auc_list = pred_auc.tolist()
    with open('/coronary_dataset/save_pre_no_b/chuac/pred_auc.txt', 'w') as file:
        for item in pred_auc_list:
            file.write(f"{item}\n")
    gt_auc_list = gt_auc.tolist()
    with open('/coronary_dataset/save_pre_no_b/chuac/gt_auc.txt', 'w') as file:
        for item in gt_auc_list:
            file.write(f"{item}\n")

    if len(set(gt_auc)) > 1:
        auc1 = roc_auc_score(gt_auc, pred_auc)
        fpr1, tpr1, _ = roc_curve(gt_auc, pred_auc)
        precision1, recall1, thresholds = precision_recall_curve(gt_auc, pred_auc)
        precision1 = np.fliplr([precision1])[0]  # so the array is increasing (you won't get negative AUC)
        recall1 = np.fliplr([recall1])[0]  # so the array is increasing (you won't get negative AUC)
        pr_value1 = auc(recall1, precision1)
    else:
        print("skip")
        auc1 = 0.1
        pr_value1 = 0.1
        fpr1, tpr1, precision1, recall1 = [0.1],[0.1],[0.1],[0.1]
    acc = (tp + tn) / (tp + fp + fn + tn + epsilon)
    pre = tp / (tp + fp + epsilon)
    sen = tp / (tp + fn + epsilon)
    spe = tn / (tn + fp + epsilon)
    iou = (tp + epsilon)/ (tp + fp + fn + epsilon)
    f1 = (2 * tp  + epsilon )/ (2 * tp + fp + fn + epsilon)

    pr_value = torch.tensor(pr_value1)
    auc1 = torch.tensor(auc1)
    acc = torch.tensor(acc)
    f1_score = torch.tensor(f1_score)
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    spe = torch.tensor(spe)



    return pr_value, auc1, acc, f1_score, precision, recall, spe, fpr1, tpr1, recall1, precision1

def computeTopo(pred, gt):

    pred = pred[0].detach().cpu().numpy().astype(int)  # float data does not support bit_and and bit_or
    gt = gt[0].detach().cpu().numpy().astype(int)

    pred = morphology.skeletonize(pred >= 0.5)
    gt = morphology.skeletonize(gt >= 0.5)

    cor_intersection = gt & pred

    com_intersection = gt & pred

    cor_tp = np.sum(cor_intersection)
    com_tp = np.sum(com_intersection)

    sk_pred_sum = np.sum(pred)
    sk_gt_sum = np.sum(gt)

    smooth = 1e-7
    correctness = cor_tp / (sk_pred_sum + smooth)
    completeness = com_tp / (sk_gt_sum + smooth)

    quality = cor_tp / (sk_pred_sum + sk_gt_sum - com_tp + smooth)

    return torch.tensor(correctness * 100), torch.tensor(completeness * 100), torch.tensor(quality * 100)

def computeTopo_withthreshold(pred, gt):


    gt = torch.where(gt > 0.0, 1, 0)
    pred = torch.where(pred >= 0.5, 1, 0)
    pred = pred[0].detach().cpu().numpy().astype(int) 
    gt = gt[0].detach().cpu().numpy().astype(int)

    pred = morphology.skeletonize(pred >= 0.5)
    gt = morphology.skeletonize(gt >= 0.5)

    tp = 0
    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[1]):
            if pred[i, j] > 0 and np.sum(gt[i - 2:i + 3, j - 2:j + 3]) > 0:
                tp += 1
    cor_tp = tp
    com_tp = tp
    sk_pred_sum = np.sum(pred)
    sk_gt_sum = np.sum(gt)
    smooth = 1e-7
    correctness = cor_tp / (sk_pred_sum + smooth)
    completeness = com_tp / (sk_gt_sum + smooth)
    quality = cor_tp / (sk_pred_sum + sk_gt_sum - com_tp + smooth)

    return torch.tensor(correctness * 100), torch.tensor(completeness * 100), torch.tensor(quality * 100)

from medpy.metric.binary import hd95
from skimage.morphology import skeletonize, skeletonize_3d
def cl_score(v, s):

    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):

    v_p = torch.where(v_p >= 0.5, 1, 0)
    v_l = torch.where(v_l > 0.0, 1, 0)
    v_p = v_p.detach().cpu().numpy().astype(float)# float data does not support bit_and and bit_or
    if torch.is_tensor(v_l):
        v_l = v_l.detach().cpu().numpy().astype(float)  # float data does not support bit_and and bit_or
    # print(v_p.shape,v_l.shape)
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    cldsc = 2*tprec*tsens/(tprec+tsens)

    try:
        myhd95 = hd95(v_p, v_l)
    except RuntimeError as e:
        print(f"Err: {e}")
        myhd95 = 1.0 

    cldsc = torch.tensor(cldsc)
    myhd95 = torch.tensor(myhd95)
    return cldsc, myhd95

def count_connect_component(predict, target, threshold=0.5, connectivity=8):
    if threshold != None:
        predict = predict[0].cpu().detach()
        predict = torch.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = torch.where(target > 0.0, 1, 0)
        target = target[0].cpu().detach()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(target, dtype=np.uint8)*255, connectivity=connectivity)
    VCA = torch.tensor(pre_n/gt_n)
    return VCA


class Evaluator:

    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred, batch):
        gt_mask = batch.get('anno_mask')

        for pred_, _gt_mask in zip(pred, gt_mask):
            pr_value_, auc_, acc_,f1_, precision_, recall_, spe_, fpr_, tpr_, recall1_, precision1_ = computeF1(pred_, _gt_mask)
            cor_, com_, quality_ = computeTopo_withthreshold(pred_, _gt_mask) 
            cldice_, hd95_ = clDice(pred_, _gt_mask)
            VCA = count_connect_component(pred_, _gt_mask)

            return pr_value_, auc_, acc_, f1_, precision_, recall_, spe_,\
                   quality_, cor_, com_, cldice_, hd95_, VCA, \
                   fpr_, tpr_, recall1_, precision1_
