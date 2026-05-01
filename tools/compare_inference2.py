"""
Unified evaluation script for code detection models:
- YUNet (fine-tuned, ONNX)
- WeChat QBar detector (baseline, ONNX)

Supports:
- Evaluation on multi-class code dataset (6 classes)
- Metrics: AP (based on IoU), Acc (classification accuracy of detected boxes)
- FPS (inference time per image)
- Single image inference with visualization
"""

import argparse
import os
import shutil
import math
import time
import yaml
from collections import defaultdict, Counter
from itertools import product
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nms import multiclass_nms

# ----------------------------------------------------------------------
# Helper functions (NMS, drawing, etc.)
# ----------------------------------------------------------------------
TYPE_MAP = {
    0: 'aruco',
    1: 'barcode',
    2: 'kuihua',
    3: 'qrcode',
    4: 'pdf417',
    5: 'datamatrix'
}
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'

def nms(dets, thresh, opencv_mode=True):
    """Non-maximum suppression.
    Args:
        dets: (N, 5) [x1, y1, x2, y2, score]
        thresh: IoU threshold
        opencv_mode: use cv2.dnn.NMSBoxes for speed
    Returns:
        keep indices
    """
    if opencv_mode:
        boxes = dets[:, :4].copy()
        scores = dets[:, -1]
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        keep = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=thresh,
            eta=1,
            top_k=5000
        )
        return keep.flatten() if len(keep) > 0 else []
    else:
        # pure numpy version (fallback)
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, -1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


def resize_img(img, mode):
    """Resize image according to mode.
    mode can be:
        'ORIGIN'   : no resize
        'AUTO'     : pad to multiple of 32
        'VGA'      : 640x480 (keeping aspect ratio)
        '640,640'  : explicit (W,H)
    """
    if mode == 'ORIGIN':
        return img, 1.0
    elif mode == 'AUTO':
        h, w = img.shape[:2]
        new_h = ((h - 1) & (-32)) + 32
        new_w = ((w - 1) & (-32)) + 32
        det_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        det_img[:h, :w] = img
        return det_img, 1.0
    else:
        if mode == 'VGA':
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(',')))
        assert len(input_size) == 2
        target_w, target_h = max(input_size), min(input_size)
        # keep aspect ratio
        h, w = img.shape[:2]
        if w > h:
            new_w, new_h = target_w, target_h
        else:
            new_w, new_h = target_h, target_w
        # resize
        scale = min(new_w / w, new_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        # pad to target size
        det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = resized
        return det_img, scale


def draw_detections(img, detections, out_path, gt_labels=None):
    """Draw bounding boxes and labels on image.
    detections: list of dict with keys 'bbox', 'score', 'class_id'
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        cls_id = det['class_id']
        cls_name = TYPE_MAP.get(cls_id, str(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label = f'{cls_name}:{score:.2f}'
        cv2.putText(img, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
    # 绘制真实框（绿色）
    if gt_labels is not None:
        for gt in gt_labels:
            x1, y1, x2, y2 = gt['bbox']
            cls_id = gt.get('class_id', None)
            label = TYPE_MAP.get(cls_id, str(cls_id))
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)  # 绿框
            # cv2.putText(img, label, (int(x1), int(y1)-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 蓝字（避免和预测文字混淆）
    cv2.imwrite(out_path, img)


# ----------------------------------------------------------------------
# Dataset for multi-class code detection (new format)
# ----------------------------------------------------------------------

class CodeDataset:
    """
    Dataset for code detection.
    Expected annotation format: one global text file with following structure:
        # <image_path> <width> <height>
        <x1> <y1> <x2> <y2> <class_id>
        ...
        # <another_image_path> ...
    Each image block starts with a comment line starting with '#'.
    The image path is relative to the dataset root.
    """
    def __init__(self, root_dir, ann_file, for_qbar=False):
        self.root = root_dir
        self.ann_file = ann_file
        self.samples = []  # each: (img_path, width, height, bboxes, labels)
        self.for_qbar = for_qbar
        self._load_annotations()

    def _load_annotations(self):
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                # parse header: # img_path width height
                parts = line[1:].strip().split()
                if len(parts) < 3:
                    i += 1
                    continue
                img_path = parts[0]
                width = int(parts[1])
                height = int(parts[2])
                i += 1
                bboxes = []
                labels = []
                # read bboxes until next header or EOF
                while i < len(lines) and not lines[i].strip().startswith('#'):
                    parts = lines[i].strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2, cls = parts[:5]
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))
                    i += 1
                # if not (self.for_qbar and len(set(labels)) == 1 and labels[0] == 3):
                #     continue
                self.samples.append({
                    'img_path': os.path.join(self.root, img_path),
                    'width': width,
                    'height': height,
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int32)
                })
            else:
                i += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['img_path'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {sample['img_path']}")
        return img, sample['bboxes'], sample['labels']

    @property
    def num_classes(self):
        # determine from annotations
        max_label = 0
        for s in self.samples:
            if len(s['labels']):
                max_label = max(max_label, s['labels'].max())
        return max_label + 1


# ----------------------------------------------------------------------
# Base Detector and YUNet implementation
# ----------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def tic(self):
        self._start = time.time()

    def toc(self):
        self.total += time.time() - self._start
        self.count += 1

    def avg(self):
        return self.total / self.count if self.count else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class Detector:
    def __init__(self, model_file, nms_thresh=0.45):
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        assert os.path.exists(model_file)
        self.session = None
        self._load_model()

    def _load_model(self):
        onnx_model = onnx.load(self.model_file)
        onnx.checker.check_model(onnx_model)
        # analysis num of params in model
        total_params = sum(np.prod(p.dims) for p in onnx_model.graph.initializer)
        print(f"{GREEN}Total parameters in model: {total_params}{RESET}")
        self.session = ort.InferenceSession(self.model_file, None)

    def detect(self, img, score_thresh=0.5, mode='640,640'):
        raise NotImplementedError


class YUNet(Detector):
    """YUNet detector for multi-class code detection."""
    def __init__(self, model_file, nms_thresh=0.45, num_classes=6):
        super().__init__(model_file, nms_thresh)
        self.num_classes = num_classes
        self.strides = [8, 16, 32]
        self.nk = 4  # number of keypoints (not used)
        self.timer = Timer()

    def _forward(self, img, score_thresh):
        """Run inference on preprocessed image (H,W,3).
        Returns:
            bboxes: (N,6) [x1,y1,x2,y2,score,class_id]
            inf_time: inference time in seconds
        """
        input_size = (img.shape[1], img.shape[0])  # (W,H)
        blob = np.transpose(img, [2,0,1]).astype(np.float32)[np.newaxis, ...]
        start = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        inf_time = time.time() - start

        # outputs: [cls_pred_stride8, cls_pred_stride16, cls_pred_stride32,
        #           obj_pred_stride8, ... , reg_pred_stride8,... , kps_pred...]
        num_scales = len(self.strides)
        scores_list = []
        bboxes_list = []
        for idx, stride in enumerate(self.strides):
            cls_pred = outputs[idx].squeeze(0)               # (N, C)
            obj_pred = outputs[idx + num_scales].reshape(-1, 1)  # (N,1)
            reg_pred = outputs[idx + 2*num_scales].reshape(-1, 4)  # (N,4)
            # generate anchor centers
            fm_h = img.shape[0] // stride
            fm_w = img.shape[1] // stride
            anchor_y, anchor_x = np.mgrid[:fm_h, :fm_w]
            anchor_centers = np.stack([anchor_x, anchor_y], axis=-1).reshape(-1,2) * stride
            # decode bbox
            bbox_cxy = reg_pred[:, :2] * stride + anchor_centers
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            tl_x = bbox_cxy[:,0] - bbox_wh[:,0]/2
            tl_y = bbox_cxy[:,1] - bbox_wh[:,1]/2
            br_x = bbox_cxy[:,0] + bbox_wh[:,0]/2
            br_y = bbox_cxy[:,1] + bbox_wh[:,1]/2
            bbox = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)
            # score = cls_score * obj_score
            cls_score = np.max(cls_pred, axis=1, keepdims=True)  # (N,1)
            cls_label = np.argmax(cls_pred, axis=1)              # (N,)
            score = cls_score * obj_pred
            # filter by threshold
            mask = (score[:,0] > score_thresh)
            if np.any(mask):
                scores_list.append(score[mask])
                bboxes_list.append(np.hstack([bbox[mask], score[mask], cls_label[mask, None]]))
        if not scores_list:
            return np.zeros((0,6)), inf_time
        bboxes = np.vstack(bboxes_list)   # (M,6)
        return bboxes, inf_time

    def detect(self, img, score_thresh=0.5, mode='640,640'):
        """Detect codes in image.
        Returns:
            detections: list of dict with keys 'bbox', 'score', 'class_id'
            inf_time: inference time
        """
        self.timer.tic()
        det_img, det_scale = resize_img(img, mode)
        self.timer.toc()  # preprocess time (not counted)
        bboxes, inf_time = self._forward(det_img, score_thresh)
        # scale back to original image
        bboxes[:, :4] /= det_scale
        # NMS
        keep = nms(bboxes[:, :5], self.nms_thresh)
        bboxes = bboxes[keep]
        # convert to list of dicts
        detections = []
        for b in bboxes:
            x1, y1, x2, y2, score, cls_id = b
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': float(score),
                'class_id': int(cls_id)
            })
        return detections, inf_time


# ----------------------------------------------------------------------
# WeChat QBar Detector (baseline)
# ----------------------------------------------------------------------

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

class Integral(nn.Module):
    """Integral layer for distribution regression."""
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x

# ----------------------------- QBarDetector 类（使用 ONNX Runtime） -----------------------------
class QBarDetector:
    def __init__(self, model_dir, nms_thresh=0.45, score_thresh=0.2, num_threads=1):
        """
        Args:
            model_dir: 目录，包含 qbar_detector.yaml 和对应的 ONNX 模型文件
            nms_thresh: NMS 的 IoU 阈值
            score_thresh: 分类置信度阈值（后处理中还会使用）
            num_threads: ONNX Runtime 的线程数（设为1保证单线程公平对比）
        """
        self.model_dir = model_dir
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

        # 加载配置
        config_path = os.path.join(model_dir, 'qbar_detector.yaml')
        with open(config_path, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        # 加载 ONNX 模型（使用 ONNX Runtime）
        onnx_path = os.path.join(model_dir, self.cfg['onnx_path'])
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        self.session = ort.InferenceSession(onnx_path, sess_options,
                                            providers=['CPUExecutionProvider'])

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # 根据原代码，预期输出顺序为：cls_stride_8, cls_stride_16, cls_stride_32,
        #                           dis_stride_8, dis_stride_16, dis_stride_32
        # 但为了保证正确性，我们按名称排序或假设顺序与 layer_names 一致
        # 原 layer_names = ('cls_pred_stride_8','cls_pred_stride_16','cls_pred_stride_32',
        #                   'dis_pred_stride_8','dis_pred_stride_16','dis_pred_stride_32')
        # 如果模型输出的名称与之一致，可以按名称索引；否则按顺序取前6个。
        self.cls_names = ['cls_pred_stride_8', 'cls_pred_stride_16', 'cls_pred_stride_32']
        self.dis_names = ['dis_pred_stride_8', 'dis_pred_stride_16', 'dis_pred_stride_32']
        # 检查输出名是否包含这些字符串，否则按位置
        if all(any(name in oname for oname in self.output_names) for name in self.cls_names + self.dis_names):
            self.use_named_output = True
        else:
            self.use_named_output = False
            print("Warning: Output names do not match expected patterns, using positional indexing.")

        # 初始化分布积分模块
        self.distribution_project = Integral(self.cfg.get("regmax", 16))
        # 将 Integral 移到 CPU（因为后处理在 CPU 进行）
        self.distribution_project = self.distribution_project.cpu()

        # 缓存 anchor 中心（可选）
        self.center_cache = {}

    # --------------------- 预处理（完全复刻原 process_qbar_det） ---------------------
    def _preprocess(self, img):
        """预处理：动态缩放、转灰度、归一化到[-1,1]，输出形状 (1,1,H,W)"""
        min_input_size = self.cfg["min_input_size"]
        max_input_size = self.cfg["max_input_size"]
        set_width = min_input_size
        set_height = min_input_size

        height, width = img.shape[:2]
        if width <= max_input_size and height <= max_input_size:
            if width >= height:
                set_width = min_input_size
                set_height = math.ceil(height * min_input_size / width)
            else:
                set_height = min_input_size
                set_width = math.ceil(width * min_input_size / height)
        else:
            resize_ratio = math.sqrt(width * height / (min_input_size * min_input_size))
            set_width = width / resize_ratio
            set_height = height / resize_ratio

        set_height = int((set_height + 31) // 32 * 32)
        set_width  = int((set_width  + 31) // 32 * 32)

        resized = cv2.resize(img, (set_width, set_height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 128.0 - 1.0
        # 增加通道维和 batch 维 -> (1,1,H,W)
        blob = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
        return blob, (set_height, set_width), (height, width)

    # --------------------- 前向推理（ONNX Runtime） ---------------------
    def _forward(self, blob):
        """执行 ONNX 推理，返回原始输出列表（numpy arrays）"""
        start = time.time()
        outputs = self.session.run(None, {self.input_name: blob})
        inf_time = time.time() - start

        if self.use_named_output:
            # 按名称提取
            cls_outs = [outputs[self.output_names.index(n)] for n in self.cls_names]
            dis_outs = [outputs[self.output_names.index(n)] for n in self.dis_names]
        else:
            # 按位置：前3个分类，后3个距离
            cls_outs = outputs[:3]
            dis_outs = outputs[3:6]
        return cls_outs, dis_outs, inf_time

    # --------------------- Anchor 生成（与原代码完全一致） ---------------------
    def _get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    # --------------------- 后处理解码（复刻原 get_bboxes） ---------------------
    def _get_bboxes(self, cls_preds, reg_preds, img_metas):
        """将网络输出转换为检测框列表（与原逻辑一致）"""
        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        input_h = img_metas["input_height"]
        input_w = img_metas["input_width"]
        scale_h = img_metas["orig_input_height"] / input_h
        scale_w = img_metas["orig_input_width"] / input_w
        input_shape = (input_h, input_w)

        featmap_sizes = [
            (math.ceil(input_h / stride), math.ceil(input_w / stride))
            for stride in [8, 16, 32]
        ]
        mlvl_priors = []
        for i, stride in enumerate([8, 16, 32]):
            priors = self._get_single_level_center_priors(
                batch_size, featmap_sizes[i], stride, torch.float32, device)
            mlvl_priors.append(priors)
        center_priors = torch.cat(mlvl_priors, dim=1)

        # 距离解码
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)

        scores = cls_preds
        result_list = []
        for i in range(batch_size):
            score = scores[i]
            bbox = bboxes[i]
            # 添加虚拟背景类（使类别数+1，便于 multiclass_nms）
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=self.score_thresh,
                nms_cfg=dict(type="nms", iou_threshold=self.nms_thresh),
                max_num=100,
            )
            if len(results) > 0:
                det_boxes, det_labels = results[0], results[1]
                for j in range(det_boxes.shape[0]):
                    x1, y1, x2, y2, conf = det_boxes[j].tolist()
                    cls_id = int(det_labels[j])
                    # 还原到原始图像尺寸
                    x1 = int(round(x1 * scale_w))
                    y1 = int(round(y1 * scale_h))
                    x2 = int(round(x2 * scale_w))
                    y2 = int(round(y2 * scale_h))
                    result_list.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": float(conf),
                        "class_id": cls_id   # 整数类别ID，与YUNet一致
                    })
        return result_list

    # --------------------- 完整检测接口（对外） ---------------------
    def detect(self, img, score_thresh=None, mode=None):
        """
        检测图像中的码
        Args:
            img: BGR 图像 (H,W,3)
            score_thresh: 可选，覆盖默认的置信度阈值
            mode: 无用，保留以兼容接口
        Returns:
            detections: list of dict, each with keys 'bbox', 'score', 'class_id'
            inf_time: 模型推理时间（秒）
        """
        if score_thresh is not None:
            self.score_thresh = score_thresh

        # 预处理
        blob, (input_h, input_w), (orig_h, orig_w) = self._preprocess(img)

        # 推理
        cls_outs, dis_outs, inf_time = self._forward(blob)

        # 将 numpy 转为 torch tensor
        cls_tensors = [torch.from_numpy(out) for out in cls_outs]
        dis_tensors = [torch.from_numpy(out) for out in dis_outs]

        # 合并多尺度输出
        cls_preds = torch.cat(cls_tensors, dim=1)   # (1, total_points, num_classes)
        reg_preds = torch.cat(dis_tensors, dim=1)   # (1, total_points, 4*(regmax+1))

        # 后处理
        img_metas = {
            "input_height": input_h,
            "input_width": input_w,
            "orig_input_height": orig_h,
            "orig_input_width": orig_w
        }
        detections = self._get_bboxes(cls_preds, reg_preds, img_metas)

        return detections, inf_time


# ----------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------

def bbox_iou(bbox, gt_bbox):
    """Compute IoU between two boxes."""
    x1 = max(bbox[0], gt_bbox[0])
    y1 = max(bbox[1], gt_bbox[1])
    x2 = min(bbox[2], gt_bbox[2])
    y2 = min(bbox[3], gt_bbox[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (bbox[2]-bbox[0]+1) * (bbox[3]-bbox[1]+1)
    area2 = (gt_bbox[2]-gt_bbox[0]+1) * (gt_bbox[3]-gt_bbox[1]+1)
    iou = inter / (area1 + area2 - inter + 1e-6)
    return iou


def evaluate_detections(predictions, groundtruths, iou_thresh=0.5):
    """
    Evaluate AP and classification accuracy.
    predictions: list of dicts per image: {'bboxes': list of [x1,y1,x2,y2], 'scores': list, 'labels': list}
    groundtruths: list of dicts per image: {'bboxes': list, 'labels': list}
    Returns:
        ap: Average Precision (overall)
        acc: Classification accuracy (for matched detections)
        per_class_acc: dict
    """
    # Collect all detections and ground truths for AP calculation (ignore class)
    all_detections = []
    all_gt = []
    image_ids = []
    for idx, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        for bbox, score, label in zip(pred['bboxes'], pred['scores'], pred['labels']):
            all_detections.append([idx, bbox[0], bbox[1], bbox[2], bbox[3], score, label])
        for bbox, label in zip(gt['bboxes'], gt['labels']):
            all_gt.append([idx, bbox[0], bbox[1], bbox[2], bbox[3], label])
        image_ids.append(idx)

    # Compute AP using VOC method (per class then average)
    num_classes = max([d[-1] for d in all_gt] + [0]) + 1
    aps = []
    ap_per_class = {}
    for c in range(num_classes):
        # gather detections for class c
        det_c = [d for d in all_detections if d[-1] == c]
        gt_c = [g for g in all_gt if g[-1] == c]
        if len(gt_c) == 0:
            continue
        # sort detections by score descending
        det_c.sort(key=lambda x: x[5], reverse=True)
        # assign ground truth boxes
        gt_used = {g[0]: [False]*len([g2 for g2 in gt_c if g2[0]==g[0]]) for g in gt_c}
        tp = np.zeros(len(det_c))
        fp = np.zeros(len(det_c))
        for i, d in enumerate(det_c):
            img_id = d[0]
            # find all gt boxes in same image
            gt_boxes = [g for g in gt_c if g[0]==img_id]
            if not gt_boxes:
                fp[i] = 1
                continue
            overlaps = [bbox_iou(d[1:5], g[1:5]) for g in gt_boxes]
            max_overlap = max(overlaps) if overlaps else 0
            max_idx = np.argmax(overlaps) if overlaps else -1
            if max_overlap >= iou_thresh:
                # check if already used
                if not gt_used[img_id][max_idx]:
                    tp[i] = 1
                    gt_used[img_id][max_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        # compute precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / len(gt_c)
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        # VOC AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
        aps.append(ap)
        ap_per_class[c] = ap
    ap = np.mean(aps) if aps else 0.0

    # Compute classification accuracy for matched detections (IoU > thresh)
    correct = 0
    total_matched = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for pred, gt in zip(predictions, groundtruths):
        # match detections to ground truth by IoU
        gt_boxes = gt['bboxes']
        gt_labels = gt['labels']
        used_gt = [False] * len(gt_boxes)
        for pbox, pscore, plabel in zip(pred['bboxes'], pred['scores'], pred['labels']):
            best_iou = 0
            best_idx = -1
            for i, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
                if used_gt[i]:
                    continue
                iou = bbox_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh:
                total_matched += 1
                per_class_total[gt_labels[best_idx]] += 1
                if plabel == gt_labels[best_idx]:
                    correct += 1
                    per_class_correct[gt_labels[best_idx]] += 1
                used_gt[best_idx] = True
    acc = correct / total_matched if total_matched > 0 else 0.0
    per_class_acc = {c: per_class_correct[c]/per_class_total[c] if per_class_total[c] else 0.0
                     for c in per_class_total}
    return ap, ap_per_class, acc, per_class_acc


# ----------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------
def remap_label(label):
    if label == 5:
        return 3
    return label


def evaluate_model(detector, dataset, score_thresh=0.5, mode='640,640', iou_thresh=0.5, draw_vis=False, out_dir='./data/yunetcode/out/'):
    """Run evaluation on whole dataset.
    Returns:
        ap, acc, fps, per_class_acc
    """
    predictions = []
    total_inf_time = 0.0
    counter = Counter()
    if not os.path.exists(out_dir):
       os.mkdir(out_dir)
    for img, gt_bboxes, gt_labels in tqdm(dataset):
        detections, inf_time = detector.detect(img, score_thresh, mode)
        total_inf_time += inf_time
        # convert to evaluation format
        pred_bboxes = [d['bbox'] for d in detections]
        pred_scores = [d['score'] for d in detections]
        pred_labels = [remap_label(d['class_id']) for d in detections]
        predictions.append({
            'bboxes': pred_bboxes,
            'scores': pred_scores,
            'labels': pred_labels
        })
        if draw_vis:
            if len(set(gt_labels)) > 1:
                counter['6'] += 1
                out_name = f'6_{counter["6"]:04d}.jpg'
            else:
                counter[str(gt_labels[0])] += 1
                out_name = f'{gt_labels[0]}_{counter[str(gt_labels[0])]:04d}.jpg'
            out_path = os.path.join(out_dir, out_name)
            draw_detections(
                img.copy(), detections, out_path, 
                gt_labels=[{'bbox': b, 'class_id': l} for b, l in zip(gt_bboxes, gt_labels)]
            )
            # print(f'Saved visualization to {out_path}')
        # ground truth (already in dataset)
        # Note: we need to pass ground truth for each image; the dataset returns them
    # Build ground truth list for evaluation
    groundtruths = []
    for idx in range(len(dataset)):
        _, bboxes, labels = dataset[idx]
        groundtruths.append({'bboxes': bboxes.tolist(), 'labels': [remap_label(l) for l in labels.tolist()]})
    ap, ap_per_class, acc, per_class_acc = evaluate_detections(predictions, groundtruths, iou_thresh)
    fps = len(dataset) / total_inf_time if total_inf_time > 0 else 0
    return ap, ap_per_class, acc, per_class_acc, fps


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate code detection models')
    parser.add_argument('--model', type=str, required=True,
                        help='Model type: yunet or qbar')
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to ONNX model file (for yunet) or model directory (for qbar)')
    parser.add_argument('--dataset_root', type=str, default='./data/yunetcode/images/',
                        help='Root directory of dataset')
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Global annotation file (format described in CodeDataset)')
    parser.add_argument('--score_thresh', type=float, default=0.5,
                        help='Score threshold for detection')
    parser.add_argument('--nms_thresh', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--mode', type=str, default='640,640',
                        help='Input resize mode: ORIGIN, AUTO, VGA, or "W,H"')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation on dataset')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image file for inference (if not eval)')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='Output directory for visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    # Load dataset
    dataset = CodeDataset(args.dataset_root, args.ann_file, for_qbar=(args.model.lower()=='qbar'))
    print(f'Dataset loaded: {len(dataset)} images, {dataset.num_classes} classes')

    # Initialize detector
    if args.model.lower() == 'yunet':
        detector = YUNet(args.model_file, nms_thresh=args.nms_thresh,
                         num_classes=dataset.num_classes)
        print('YUNet detector initialized.')
    elif args.model.lower() == 'qbar':
        detector = QBarDetector(args.model_file, nms_thresh=args.nms_thresh)
        print('QBar detector initialized.')
    else:
        raise ValueError(f'Unknown model type: {args.model}')

    if args.eval:
        ap, ap_per_class, acc, per_class_acc, fps = evaluate_model(
            detector, dataset, args.score_thresh, args.mode, draw_vis=False)
        print('\n=== Evaluation Results ===')
        print(f'Model: {args.model}')
        print(f'{BLUE}AP (IoU={args.score_thresh}): {ap:.4f}{RESET}')
        print(f'Per-class AP:')
        for c, a in ap_per_class.items():
            if a >= 0.85:
                print(f'{GREEN}  Class {c}: {a:.4f}{RESET}')
            elif a >= 0.6:
                print(f'{YELLOW}  Class {c}: {a:.4f}{RESET}')
            else:
                print(f'{RED}  Class {c}: {a:.4f}{RESET}')

        print(f'{BLUE}Classification Accuracy: {acc:.4f}{RESET}')
        print(f'Per-class Accuracy:')
        per_class_acc = {c: a for c, a in sorted(per_class_acc.items())}
        for c, a in per_class_acc.items():
            if a >= 0.85:
                print(f'{GREEN}  Class {c}: {a:.4f}{RESET}')
            elif a >= 0.6:
                print(f'{YELLOW}  Class {c}: {a:.4f}{RESET}')
            else:
                print(f'{RED}  Class {c}: {a:.4f}{RESET}')
        print(f'{BLUE}FPS: {fps:.2f}{RESET}')
    else:
        if args.image is None:
            print('Please provide --image for single image inference (or --eval for evaluation).')
            return
        img = cv2.imread(args.image)
        if img is None:
            print(f'Failed to read image: {args.image}')
            return
        detections, inf_time = detector.detect(img, args.score_thresh, args.mode)
        print(f'Detected {len(detections)} codes')
        print(f'{BLUE}  FPS: {1/inf_time:.2f}{RESET}')
        out_path = os.path.join(args.out_dir, f'{args.model}_result.jpg')
        os.makedirs(args.out_dir, exist_ok=True)
        draw_detections(img, detections, out_path)


if __name__ == '__main__':
    main()