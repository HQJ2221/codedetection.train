import argparse
import os
import re
from time import time
from tqdm import tqdm
import cv2
import scipy.io
import numpy as np
from code_evaluation import wider_evaluation
from qbar_model import *


class WIDERFace:
    def __init__(self, root, split='test'):
        self.root = root
        self.split = split
        assert self.root is not None

        self.widerface_img_paths = {
            'val': os.path.join(self.root, 'WIDER_val', 'images'),
            'test': os.path.join(self.root, 'WIDER_test', 'images')
        }

        self.widerface_split_fpaths = {
            'val': os.path.join(self.root, 'wider_face_split', 'wider_face_val.mat'),
            'test': os.path.join(self.root, 'labelv2/val/gt', 'output.mat')
        }

        self.img_list, self.num_img = self.load_list()

    def load_list(self):
        n_imgs = 0
        flist = []
        split_fpath = self.widerface_split_fpaths[self.split]
        img_path = self.widerface_img_paths[self.split]
        anno_data = scipy.io.loadmat(split_fpath)
        event_list = anno_data.get('event_list')
        file_list = anno_data.get('file_list')

        for event_idx, event in enumerate(event_list):
            event_name = event[0][0]
            for f_idx, f in enumerate(file_list[event_idx][0]):
                f_name = f[0][0]
                f_path = os.path.join(img_path, event_name, f_name + '.jpg')
                flist.append(f_path)
                n_imgs += 1

        return flist, n_imgs

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        event, name = re.split(r'\\|/', self.img_list[index])[-2:]
        return img, event, name

    def __len__(self):
        return self.num_img


def onnx_eval(qbar_model, eval=False, score_thresh=0.3, image=None, out_path=None):
    if eval:
        widerface_root = './data/widerface/'
        testloader = WIDERFace(split='test', root=widerface_root)
        results = {}
        time_in_model = 0

        for idx in tqdm(range(len(testloader))):
            img, event_name, img_name = testloader[idx]
            result_list, inf_time = qbar_model.process_qbar_detect(img)  # calc runtime of model only
            time_in_model += inf_time  # add all time in model

            bboxes = []
            for r in result_list:
                x1, y1, x2, y2 = r["bbox"]
                score = r["score"]
                bboxes.append([x1, y1, x2, y2, score])
            bboxes = np.array(bboxes)

            if event_name not in results:
                results[event_name] = {}
            results[event_name][img_name.rstrip('.jpg')] = bboxes

        # print(f'Total Images: {len(testloader)}')
        ap = wider_evaluation(
            pred=results,
            gt_path=os.path.join(widerface_root, 'labelv2', 'val', 'gt'),
        )

        time_in_model_per_img = time_in_model / len(testloader)
        print(f'\033[32mAvg time for each img in model: {time_in_model_per_img:.6f} sec')
        print(f'FPS: {1 / time_in_model_per_img:.6f}')
        print(f'AP: {ap}\033[0m')

    else:
        assert image is not None
        img = cv2.imread(image)
        print(f'The origin shape is: {img.shape[:-1]}')

        warm_epochs = 1
        for _ in range(warm_epochs):
            _ = qbar_model.process_qbar_detect(img)

        run_epochs = 1
        t0 = time()
        for _ in range(run_epochs):
            result_list = qbar_model.process_qbar_detect(img)
        t1 = time() - t0

        print(f'Total Time: {t1:.4f} sec')
        print(f'FPS: {run_epochs / t1:.2f}')

        for r in result_list:
            x1, y1, x2, y2 = r["bbox"]
            cls = r["class"]
            score = r["score"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'{cls} {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        os.makedirs(out_path, exist_ok=True)
        output_path = os.path.join(out_path, 'qbar_result.jpg')
        cv2.imwrite(output_path, img)
        print(f'Saved to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='WeChat QBar model inference')
    parser.add_argument('--eval', action='store_true', help='eval on widerface')
    parser.add_argument('--image', type=str, default=None, help='image to detect')
    parser.add_argument('--score_thresh', type=float, default=0.2, help='score threshold')
    parser.add_argument('--out_path', type=str, default='./output', help='output path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    qbar_model = QBar_MODEL("models")
    onnx_eval(
        qbar_model,
        eval=args.eval,
        score_thresh=args.score_thresh,
        image=args.image,
        out_path=args.out_path
    )
