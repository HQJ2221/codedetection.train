import os
import cv2
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nms import multiclass_nms  # 确保你项目中有这个文件

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
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
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x






class QBar_MODEL(object):
    def __init__(self,model_path):
        self.model_path = model_path
        
        self.qbar_detector_config, self.qbar_detector = self.load_model_config(self.model_path,'qbar_detector.yaml')

        self.qbar_sr_config, self.qbar_sr = self.load_model_config(self.model_path,'qbar_sr.yaml')

        self.qbar_seg_config, self.qbar_seg = self.load_model_config(self.model_path,'qbar_seg.yaml')

        self.distribution_project = Integral(self.qbar_detector_config["regmax"])



    def load_model_config(self,root_path,config_path):
        with open(os.path.join(root_path,config_path), "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        onnx_path = config['onnx_path']

        dnn_engine  = cv2.dnn.readNetFromONNX(os.path.join(root_path,onnx_path))

        return config,dnn_engine


    def net_forward_qbar_det(self,image):
        blob = cv2.dnn.blobFromImage(image)
        self.qbar_detector.setInput(blob)
        layer_names = ('cls_pred_stride_8', 'cls_pred_stride_16', 'cls_pred_stride_32', 'dis_pred_stride_8', 'dis_pred_stride_16', 'dis_pred_stride_32')
        out = self.qbar_detector.forward(layer_names)
        return out
    
    def net_forward_qbar_sr(self,image):
        blob = cv2.dnn.blobFromImage(image)
        self.qbar_sr.setInput(blob)
        out = self.qbar_sr.forward()
        return out

    def net_forward_qbar_seg(self,image):
        blob = cv2.dnn.blobFromImage(image)
        self.qbar_seg.setInput(blob)
        out = self.qbar_seg.forward()
        return out



    
    def process_qbar_det(self,image):

        min_input_size = self.qbar_detector_config["min_input_size"]
        max_input_size = self.qbar_detector_config["max_input_size"]
        set_width = min_input_size
        set_height = min_input_size

        width = image.shape[1]
        height = image.shape[0]
        # 如果图像的宽和高都小于640, 则长边对齐到256
        if width <= max_input_size and height <= max_input_size:
            if width >= height:
                set_width = min_input_size
                set_height = math.ceil(height * 1.0 * min_input_size / width)
            else:
                set_height = min_input_size
                set_width = math.ceil(width * 1.0 * min_input_size / height)
        else:  # 如果图像的宽和高都大于640, 则保证面积不大于minInputSize*minInputSize
            resize_ratio = math.sqrt(width * height * 1.0 / (min_input_size * min_input_size))
            set_width = width / resize_ratio
            set_height = height / resize_ratio

        set_height = int((set_height + 32 - 1) / 32) * 32
        set_width = int((set_width + 32 - 1) / 32) * 32

        image = cv2.resize(image, (set_width, set_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = image / 128 - 1
        image = image[:,:,np.newaxis]
        return image

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)



    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["input_height"], img_metas["input_width"]

        scale_height = img_metas["orig_input_height"] / input_height
        scale_width = img_metas["orig_input_width"] / input_width
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in [8,16,32]
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate([8,16,32])
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.2,
                nms_cfg=dict(type="nms", iou_threshold=0.45),
                max_num=100,
            )


            if(len(results)>0):
                for j in range(0,results[0].shape[0]):
                    result_list.append({"bbox":list(map(int,[results[0][j][0]*scale_width,
                                                results[0][j][1]*scale_height,
                                                results[0][j][2]*scale_width,
                                                results[0][j][3]*scale_height,
                                                ])),
                                        "score":float(results[0][j][4]),
                                        "class":self.qbar_detector_config["cls_name"][results[1][j]]
                                        })
        return result_list

        
    def post_process_qbar_det(self,blob_input,orig_image,outputs):

        cls_scores=[]
        bbox_preds = []

        for i in range(0,len(outputs)//2):
            cls_scores.append(outputs[i])
            bbox_preds.append(outputs[i+len(outputs)//2])
        
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        meta = {"input_height":blob_input.shape[0] ,"input_width":blob_input.shape[1],"orig_input_height":orig_image.shape[0],"orig_input_width":orig_image.shape[1]}
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)

        return result_list
    

    def draw_boxes(self,image,results):
        for i in range(0,len(results)):
            image = cv2.rectangle(image, (results[i]["bbox"][0], results[i]["bbox"][1]), (results[i]["bbox"][2], results[i]["bbox"][3]), (0, 0, 255), 2)
            image = cv2.putText(image,results[i]["class"],(results[i]["bbox"][0], results[i]["bbox"][1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)

        return image

    
    def process_qbar_detect(self,image):
        blob_input = self.process_qbar_det(image)

        outputs = self.net_forward_qbar_det(blob_input)

        outputs = [torch.from_numpy(output) for output in outputs]

        result_list = self.post_process_qbar_det(blob_input,image,outputs)

        return result_list
    
    def process_qbar_sr(self,image):
        out_image = self.net_forward_qbar_sr(image[:,:,np.newaxis].astype(np.float32))

        return out_image.astype(np.uint8)
    
    def process_qbar_seg(self,image):

        image = cv2.resize(image, (self.qbar_seg_config["set_w"], self.qbar_seg_config["set_h"]))

        image = image.astype(np.float32) / 128.0 -1.0

        out_image = self.net_forward_qbar_seg(image[:,:,np.newaxis])

        return out_image
