import os
import cv2
import math
import numpy as np

# ------- Train 配置路径 -------
input_label_file = './labelv2/train/labelv2.txt'
image_output_dir = './WIDER_train/images/images'
label_output_file = './train.txt'
ignore_file = './ignore.txt'

# 创建输出目录
os.makedirs(image_output_dir, exist_ok=True)

def apply_affine_to_point(x, y, M):
    """应用仿射矩阵到关键点"""
    pt = np.dot(M, np.array([x, y, 1]))
    return pt[0], pt[1]

# 读取全部行
with open(input_label_file, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# 找出所有图像标注块的起始行（以 # 开头）
image_indices = [i for i, line in enumerate(lines) if line.startswith('#')]

with open(label_output_file, 'w') as f_out, open(ignore_file, 'w') as f_ignore:
    for i in image_indices:
        # 如果该图像不是单框（下一行为边框，且再下一行是新图片或结束），跳过
        if i + 1 >= len(lines) or (i + 2 < len(lines) and not lines[i + 2].startswith('#')):
            f_ignore.write(lines[i] + '\n')
            if i + 1 < len(lines):
                f_ignore.write(lines[i + 1] + '\n')
            continue

        parts = lines[i].split()
        img_path = parts[1]
        try:
            width, height, cls = map(int, parts[2:5])
        except Exception as e:
            print(f"跳过无法解析的图片大小/类别: {e}")
            continue

        full_img_path = os.path.join(image_output_dir, os.path.basename(img_path))
        print(f"处理图像: {img_path}")
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"无法读取图片 {full_img_path}")
            continue

        h, w = img.shape[:2]
        box_line = lines[i + 1].strip()
        try:
            box_info = box_line.strip().split()
            x1, y1, x2, y2 = map(int, box_info[:4])
            kps = list(map(float, box_info[4:]))
        except Exception as e:
            print(f"解析标注失败：{e}，跳过该图片")
            continue

        for angle in [90, 180, 270]:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)

            # 修正仿射矩阵偏移
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

            # 旋转边框四角
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            rotated_corners = [apply_affine_to_point(x, y, M) for x, y in corners]
            xs, ys = zip(*rotated_corners)
            new_x1, new_y1, new_x2, new_y2 = min(xs), min(ys), max(xs), max(ys)

            # 旋转关键点
            rotated_kps = []
            for j in range(0, len(kps), 3):
                kp_x, kp_y, v = kps[j:j+3]
                new_kx, new_ky = apply_affine_to_point(kp_x, kp_y, M)
                rotated_kps.extend([new_kx, new_ky, v])

            # 保存图像
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_img_name = f"{base_name}_rot{angle}.jpg"
            out_img_path = os.path.join(image_output_dir, new_img_name)
            cv2.imwrite(out_img_path, rotated_img)

            # 保存标注
            f_out.write(f"# ./images/{new_img_name} {new_w} {new_h} {cls}\n")
            f_out.write(f"{int(new_x1)} {int(new_y1)} {int(new_x2)} {int(new_y2)} " +
                        " ".join(f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in zip(rotated_kps[::3], rotated_kps[1::3], rotated_kps[2::3])) +
                        "\n")
