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

```sh
python tools/compare_inference.py ./onnx/yunet_n_640_640_withRotate.onnx --mode 640,640 --image ./1.jpg --score_thresh 0.3 --nms_thresh 0.45
```

## Notice for pull request

- Please use clear annotation when you update the code base:
    - update for documents(include README): `Docs: [content]`
    - update model(configs, mmdet or the structure of model): `Feat: [content]`
    - add new tools(scripts) or modify: `Test: [content]`
- You should specify details of what you do with the codes. Thanks for contribution!!