import os
import shutil

# 配置路径
txt_path = './wrong_predictions.txt'        # 包含图片名的txt文件
source_dir = './tmp/kuihua'      # 原始图片所在文件夹
target_dir = './wrong/kuihua'      # 你希望复制到的目标文件夹

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 读取图片名并复制
with open(txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts:  # 避免空行
            filename = parts[0]
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(target_dir, filename)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                print(f'Copied: {filename}')
            else:
                print(f'File not found: {filename}')
