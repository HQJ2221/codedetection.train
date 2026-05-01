# yunet test
python tools/compare_inference2.py --model yunet --model_file onnx/yunetcode_0501.onnx \
  --ann_file data/yunetcode/labelv2/test/labels.txt --score_thresh 0.4 --nms_thresh 0.5 --eval


# # qbar test
# python tools/compare_inference2.py --model qbar --model_file models \
#   --ann_file data/yunetcode/labelv1/test/labels.txt --score_thresh 0.2 --nms_thresh 0.45 --eval