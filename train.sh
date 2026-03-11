CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python tools/train.py ./configs/yunet_n.py --cfg-options runner.max_epochs=32