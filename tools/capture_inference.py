import cv2
import numpy as np
import time
from compare_inference import YUNET
import argparse
import os

class_map = {
    0: 'ArUco',
    1: 'Bar',
    2: 'Kuihua',
    3: 'QR',
    4: 'pdf417',
    5: 'datamatrix',
}

class QRCodeDetector:
    def __init__(self, model_path, fps):
        self.detector = YUNET(model_file=model_path, nms_thresh=0.45)
        # self.cap = cv2.VideoCapture(0)

        self.cap = None
        for index in [0, 1, 2, 3, "/dev/video0", "/dev/video1", "/dev/video2"]:
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                print(f"成功打开摄像头：{index}")
                break
        else:
            raise RuntimeError("无法找到可用摄像头")
        self.frame_interval = 1 / fps
        self.last_time = time.time()
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

    def draw_realtime(self, frame, bboxes):
        for bbox in bboxes:
            x1, y1, x2, y2, score, cls_id = bbox
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示置信度
            label = f"{class_map[cls_id]}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        return frame

    def run(self):
        while True:
            # 控制帧率
            current_time = time.time()
            if current_time - self.last_time < self.frame_interval:
                continue
            self.last_time = current_time

            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 水平翻转
            # frame = cv2.resize(frame, (640, 640))  # 调整大小

            # 执行检测
            bboxes, _, _ = self.detector.detect(
                frame, 
                score_thresh=0.2, 
                mode='640,640'
            )

            # 绘制结果
            if len(bboxes) > 0:
                frame = self.draw_realtime(frame, bboxes)

            # 显示画面
            cv2.imshow("Code Detection", frame)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='inference by ONNX')
    parser.add_argument('model', help='onnx model file path')
    parser.add_argument('--fps', type=int, default=120, help='FPS of the video')  # default: 120 fps
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} does not exist.")
    if args.fps <= 0:
        raise ValueError("FPS must be a positive integer.")
    detector = QRCodeDetector(args.model, args.fps)
    detector.run()

# Try to run: python tools/capture_inference.py onnx/yunet_n_640_640_4type.onnx