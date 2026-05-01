# Code Detection(draft ver.)

## About

> Codes mostly stem from repo [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
> 
> My team modify part of the framework to make it compact for graphic code detection and support multi-class training.
>
> Please check part of the detail [here](https://github.com/HQJ2221/codedetection.train/blob/main/report.md).

- Now we have made the fine-tuned YuNet support 4 types of codes: `QR Code`, `1D Barcode`, `ArUco`, `Kuihua Code`(also known as *小程序码* in WeChat), `pdf417` and `datamatrix`
- Upcoming support: wait for update.

**Performance**

- The fine-tuned YuNet has passed the same test with the model that WeChat uses(in `OpenCV`), and we compare mainly in AP score and FPS aspect.
- Results show that the **FPS** of the fine-tuned YuNet is almost **3 times** as WeChat model
- AP score are slightly lower:
    - before introducing `datamatrix`, QRCode mAP@0.5 can exceed 0.85, but after, it drastically drops to around 0.4 (and mAP@0.4 only 0.5)
    - for 1D Barcode, recent experiment shows its mAP@0.5 = 0.806
    - WeChatDetector(in `OpenCV` 3rd party) can reach 0.9+ for both.

## How to train and inference

- run this to train the model (I've proved that with less than 20k train data, YuNet converges well within 32 epochs)
- Notice: You should config your env first(view [here](https://github.com/ShiqiYu/libfacedetection.train))

```sh
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/yunetcode.py --cfg-options runner.max_epochs=32
```

- training checkpoint will be saved as `.pth` file, and use `tools/yunet2onnx.py` to convert it to `onnx`. (btw, use `tools/wandb_log.py` to view metrics since sync wandb logging haven't designed yet)
- run onnx to inference with single image.

```
python tools/compare_inference2.py --model yunet --model_file [onnxmodel] \
  --ann_file [annotation_file] --score_thresh 0.5 --nms_thresh 0.5 --image [image_path]
```

- or if you want to take one testset into evaluation:
    - Notice that you need to configurate the test-set path in configs file(E.g. `configs/yunetcode.py`) and make sure the annotation file is in the same format as the training set, and then run:

```
python tools/compare_inference2.py --model yunet --model_file [onnxmodel] \
  --ann_file [annotation_file] --score_thresh 0.5 --nms_thresh 0.5 --eval
```




### WeChat QBar Model Evaluation Guide

This document describes how to evaluate the WeChat QBar detection model and clarifies the default threshold parameters used during evaluation.

#### Evaluation Command

From the project root directory, run the following command to evaluate the QBar model:

```bash
python ./tools/qbar_compare.py --eval
```




