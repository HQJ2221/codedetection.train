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
}

class QRCodeDetector:
    def __init__(self, model_path):
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
        self.frame_interval = 1/30  # 最大30FPS
        self.last_time = time.time()

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

            # 执行检测
            bboxes, _ = self.detector.detect(
                frame, 
                score_thresh=0.2, 
                mode='640,640'
            )

            # 绘制结果
            if len(bboxes) > 0:
                frame = self.draw_realtime(frame, bboxes)

            # 显示画面
            cv2.imshow("QR Code Detection", frame)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='inference by ONNX')
    parser.add_argument('model', help='onnx model file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    detector = QRCodeDetector(args.model)
    detector.run()

# Try to run: python tools/capture_inference.py onnx/yunet_n_640_640_4type.onnx