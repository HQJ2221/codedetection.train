import os
import argparse
import numpy as np
import scipy.io as sio
from collections import defaultdict

from tools.code_evaluation import CODE_TYPES

def parse_args():
    parser = argparse.ArgumentParser(description="Convert txt annotations to .mat format")
    parser.add_argument('--input', type=str, default='labelv2.txt', help='Path to input txt file')
    parser.add_argument('--output', type=str, default='output.mat', help='Path to output .mat file')
    parser.add_argument('--test_type', type=int, default=-1, help='Code type to evaluate (0-5), -1 for all')
    return parser.parse_args()

def main():
    args = parse_args()

    # ==============================
    # 读取 txt 文件
    # ==============================
    filename = args.input

    with open(filename, 'r') as f:
        lines = f.readlines()

    # ==============================
    # 按 event 分类存储
    # ==============================
    event_dict = defaultdict(list)
    bbox_dict = defaultdict(list)
    current_event = None
    current_file = None
    current_bboxes = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # --------------------------
        # 图片路径行
        # --------------------------
        if line.startswith('#'):
            # 去掉 "# " 前缀
            line = line.replace('# ', '')

            # 示例： barcode/test_barcode_0001.jpg
            parts = line.split(' ')
            file_path = parts[0]
            fname_parts = file_path.split('/')

            if current_event is not None:
                if args.test_type == -1 or ftype == int(args.test_type):
                    event_dict[current_event].append(current_file)
                    bbox_dict[current_event].append(
                        np.array(current_bboxes, dtype=np.float32)
                        if len(current_bboxes) > 0
                        else np.zeros((0, 4), dtype=np.float32)
                    )

            ftype = int(parts[-1])
            event_name = fname_parts[0]
            file_name = fname_parts[-1].replace('.jpg', '')

            current_event = event_name
            current_file = file_name
            current_bboxes = []

        else:
            parts = line.split()
            bbox = [float(x) for x in parts[:4]]
            current_bboxes.append(bbox)

    if current_event is not None:
        if args.test_type == -1 or ftype == int(args.test_type):
            event_dict[current_event].append(current_file)
            bbox_dict[current_event].append(
                np.array(current_bboxes, dtype=np.float32)
                if len(current_bboxes) > 0
                else np.zeros((0, 4), dtype=np.float32)
            )

    # ==============================
    # 构造 MATLAB 兼容结构
    # ==============================

    events = sorted(event_dict.keys())
    n_events = len(events)

    event_list = np.empty((n_events, 1), dtype=object)
    file_list = np.empty((n_events, 1), dtype=object)
    face_bbx_list = np.empty((n_events, 1), dtype=object)

    for i, event in enumerate(events):
        event_list[i, 0] = event

        files = event_dict[event]
        bboxes = bbox_dict[event]
        n_files = len(files)

        file_cell = np.empty((n_files, 1), dtype=object)
        bbox_cell = np.empty((n_files, 1), dtype=object)

        for j in range(n_files):
            file_cell[j, 0] = files[j]
            bbox_cell[j, 0] = bboxes[j]

        file_list[i, 0] = file_cell
        face_bbx_list[i, 0] = bbox_cell


    # ==============================
    # 保存 .mat
    # ==============================

    sio.savemat(
        args.output,
        {
            'event_list': event_list,
            'file_list': file_list,
            'face_bbx_list': face_bbx_list
        },
        do_compression=True
    )

    print(f"Save to {args.output}")

if __name__ == "__main__":
    main()