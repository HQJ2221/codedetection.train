"""
Upload training logs from a MMCV JSON file to Weights & Biases.

Usage:
    python tools/wandb_log.py <json_file> --project <project_name> [--run_name <name>] [--entity <entity>]

Example:
    python tools/wandb_log.py work_dirs/yunetcode/20260411_132529.log.json --run_name e120_rand_0411-1325
"""

import argparse
import json
import sys

import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Upload MMCV training logs to wandb.')
    parser.add_argument('json_file', type=str, help='Path to the JSON log file.')
    parser.add_argument('--run_name', type=str, help='WandB run name. If not provided, uses exp_name from file.')
    parser.add_argument('--no-config', action='store_true',
                        help='Do not upload any config (seed/exp_name) to wandb.config.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 读取整个文件（可能很大，逐行处理）
    try:
        with open(args.json_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if len(lines) == 0:
        print("Error: File is empty.")
        sys.exit(1)

    # 第一行是元数据（env_info, config, seed, exp_name）
    try:
        metadata = json.loads(lines[0])
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}")
        sys.exit(1)

    # 确定 run 名称
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = metadata.get('exp_name', 'unnamed_experiment')

    # 准备 wandb 配置（仅保留关键字段，避免过大的 env_info 和 config 字符串）
    config_dict = {}
    if not args.no_config:
        config_dict = {
            'seed': metadata.get('seed'),
            'exp_name': metadata.get('exp_name')
        }
        # 可选：添加其他简单字段，如 dataset 信息等，可根据需要扩展

    # 初始化 wandb
    wandb.init(project="yunet", entity="liberoVLA", name=run_name, config=config_dict)

    # 逐行处理迭代数据（从第二行开始）
    global_step = 0
    for idx, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping line {idx+1} due to JSON error: {e}")
            continue

        # 只处理训练模式的数据（通常所有行都是 train）
        if data.get('mode') != 'train':
            continue

        # 构建要记录的指标字典
        log_dict = {}

        # 需要记录的 loss（排除 loss_kps）
        losses_to_log = ['loss_cls', 'loss_bbox', 'loss_obj', 'loss']
        for loss_name in losses_to_log:
            if loss_name in data:
                log_dict[loss_name] = data[loss_name]

        # 其他指标：time, memory, lr （data_time 可选，这里一并记录）
        for key in ['time', 'memory', 'lr', 'data_time']:
            if key in data:
                log_dict[key] = data[key]

        # 可选：保留 epoch 和 iter 以便在 wandb 中查看
        if 'epoch' in data:
            log_dict['epoch'] = data['epoch']
        if 'iter' in data:
            log_dict['iter'] = data['iter']

        global_step += 1
        wandb.log(log_dict, step=global_step)

    wandb.finish()
    print("Upload completed successfully.")


if __name__ == '__main__':
    main()

