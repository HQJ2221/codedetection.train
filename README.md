# Code Detection(draft ver.)

- This repo is just for test and share in my 2025 Spring research.
- It is still being improved, remaining unsafe for use.

## About

> Codes 99.99% stem from repo [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
> 
> My team just modify a little bit to make it compact for qrcode detection and support multi-class training.
> 
> Thanks for all previous work by all geniuses!
>
> Please check part of the detail [here](https://github.com/HQJ2221/codedetection.train/blob/main/report.md).

- Now we have made the fine-tuned YuNet support 4 types of codes: `QR Code`, `1D Barcode`, `ArUco` and `Kuihua Code`(known as mini program code in WeChat)
- Upcoming support: `pdf417`, `datamatrix`

**Performance**

- The fine-tuned YuNet has passed the same test with the model that WeChat uses(in `OpenCV`), and we compare mainly in AP score and FPS aspect.
- Results show that the **FPS** of the fine-tuned YuNet is almost **3 times** as WeChat model
- AP score are similar:
    - in QR Code test, AP score can exceed WeChat model by about 2%.
    - but in 1D Barcode test, AP score is not good enough.

## How to train and inference

- run this to train the model
- Notice: You should config your env first(view [here](https://github.com/ShiqiYu/libfacedetection.train))

```sh
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/yunet_n.py --cfg-options runner.max_epochs=1
```


- run onnx to inference

```
python tools/compare_inference.py [onnxmodel] --mode 640,640 --image [singal_image_dir] --score_thresh 0.2 --nms_thresh 0.45
```

- or if you want to take one testset into evaluation:
    - Notice that this line will automatically set `${root}/data/widerface/WIDER_test/images/images/*.jpg` as all evaluated photos, and `${root}/data/widerface/labelv2/val/gt/output.mat` as evaluation `.mat` file.

```
python tools/compare_inference.py [onnxmodel] --eval --mode 640,640 --score_thresh 0.2 --nms_thresh 0.45
```




### WeChat QBar Model Evaluation Guide

This document describes how to evaluate the WeChat QBar detection model and clarifies the default threshold parameters used during evaluation.

#### Evaluation Command

From the project root directory, run the following command to evaluate the QBar model:

```bash
python ./tools/qbar_compare.py --eval
```

#### Default Thresholds

The score threshold and NMS IoU threshold are hardcoded in the following file:

- File: `qbar_model.py`
- Lines: Around 220–230, inside the `get_bboxes()` method

```python
results = multiclass_nms(
    bbox,
    score,
    score_thr=0.2,  # Score threshold
    nms_cfg=dict(type="nms", iou_threshold=0.45),  # NMS IoU threshold
    max_num=100,
)
```


