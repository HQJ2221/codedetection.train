# Code Detection(draft ver.)

- This repo is just for test and share in my 2025 Spring research.
- It is uncompleted, so it's unsafe for use.
- We are still working to improve it

## About

- Codes 99.99% stem from repo [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
- My team just modify a little bit to make it compact for qrcode detection and support multi-class training.
- Thanks for all previous work by all geniuses!

## Notes about running

- run this to train the model
- Notice: You should config your env first!

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
## Notice for pull request

- Please use clear annotation when you update the code base:
    - update for documents(include README): `Docs: [content]`
    - update model(configs, mmdet or the structure of model): `Feat: [content]`
    - add new tools(scripts) or modify: `Test: [content]`
- You should specify details of what you do with the codes. Thanks for contribution!!
