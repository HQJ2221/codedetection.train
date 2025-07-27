"""
This script is based on /mmdet/core/evaluation/widerface.py
To compact our code-detection model, we only use one .mat file for testing
"""

import datetime
import os
import pickle

import numpy as np
import tqdm
from scipy.io import loadmat

def bbox_overlap(a, b):
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    o = inter / (aarea + barea - inter)
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def np_round(val, decimals=4):
    return val


def get_gt_boxes(mat_path):
    """gt dir: (code_val.mat)"""

    gt_mat = loadmat(mat_path)

    # FIXME: this is based on .mat file's structure
    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']
    # gt_list = gt_mat['gt_list']

    return facebox_list, event_list, file_list


def norm_score(pred):
    """norm score pred {key: [[x1,y1,x2,y2,s]]}"""

    max_score = -1
    min_score = 2

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64) / diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh, mpp):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    # print(_pred.shape, _gt.shape)
    # 这里把坐标格式从 (x1, y1, w, h) 转换为 (x1, y1, x2, y2)，但是我们不需要
    # _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    # _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    # _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    # _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    gt_overlap_list = mpp.starmap(
        bbox_overlap,
        zip([_gt] * _pred.shape[0], [_pred[h] for h in range(_pred.shape[0])]))

    for h in range(_pred.shape[0]):

        gt_overlap = gt_overlap_list[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        # print(max_overlap)

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    fp = np.zeros((pred_info.shape[0], ), dtype=np.int32)
    # last_info = [-1, -1]
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)  # valid pred number
            pr_info[t, 1] = pred_recall[r_index]  # valid gt number

            if t > 0 and pr_info[t, 0] > pr_info[t - 1, 0] and pr_info[
                    t, 1] == pr_info[t - 1, 1]:
                fp[r_index] = 1
    return pr_info, fp


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def wider_evaluation(pred, mat_path, iou_thresh=0.5):
    # pred = get_preds(pred)
    pred = norm_score(pred)
    thresh_num = 1000
    # thresh_num = 2000
    facebox_list, event_list, file_list = get_gt_boxes(mat_path)
    event_num = len(event_list)
    from multiprocessing import Pool

    # from multiprocessing.pool import ThreadPool
    mpp = Pool(8)
    iou_th = iou_thresh
    count_code = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    # [hard, medium, easy]
    # high_score_count = 0
    # high_score_fp_count = 0
    for i in range(event_num):
        event_name = str(event_list[i][0][0])
        img_list = file_list[i][0]
        pred_list = pred[event_name]
        # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
        gt_bbx_list = facebox_list[i][0]

        for j in range(len(img_list)):
            img_name = str(img_list[j][0][0])
            pred_info = pred_list[img_name][:5]  # ensure pred is (Nx5)

            gt_boxes = gt_bbx_list[j][0].astype('float')
            count_code += len(gt_boxes)

            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            
            # print(pred_info[0], gt_boxes[0])

            ignore = np.ones(gt_boxes.shape[0], dtype=np.int32)
            # pred_info = np_round(pred_info, 4)
            # gt_boxes = np_round(gt_boxes, 4)
            pred_recall, proposal_list = image_eval(
                pred_info, gt_boxes, ignore, iou_th, mpp)

            _img_pr_info, fp = img_pr_info(thresh_num, pred_info,
                                            proposal_list, pred_recall)

            pr_curve += _img_pr_info
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_code)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]

    for srecall in np.arange(0.1, 1.0001, 0.1):
        rindex = len(np.where(recall <= srecall)[0]) - 1
        rthresh = 1.0 - float(rindex) / thresh_num
        # print('Recall-Precision-Thresh:', recall[rindex], propose[rindex],
        #       rthresh)

    ap = voc_ap(recall, propose)

    return ap
