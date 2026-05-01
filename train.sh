# python data/yunetcode/scripts/train_test_split.py --label_file ./data/yunetcode/labelv2/labels.txt

echo -e "\033[92mTrain-test split completed. Start training......\033[0m"

CUDA_VISIBLE_DEVICES=1,3,4,5,7 CUDA_LAUNCH_BLOCKING=1 \
python tools/train.py ./configs/yunetcode.py --cfg-options runner.max_epochs=32