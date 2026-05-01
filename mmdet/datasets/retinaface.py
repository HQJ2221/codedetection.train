import numpy as np

from .builder import DATASETS
from .custom import CustomDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class RetinaFaceDataset(CustomDataset):

    CLASSES = ('aruco', 'barcode', 'kuihua', 'qrcode', 'pdf417', 'datamatrix')

    def __init__(self, min_size=None, **kwargs):
        self.NK = 4
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.gt_path = kwargs.get('gt_path')
        super(RetinaFaceDataset, self).__init__(**kwargs)

    def _parse_ann_line(self, line):
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)

        if len(values) < 5:
            raise ValueError(f"标注行必须包含类别信息: {line}")
        cat_idx = int(values[4])
        cat = self.CLASSES[cat_idx]  # 转换为类别名称

        # 关键点初始化为全零（可根据需要扩展）
        kps = np.zeros((self.NK, 3), dtype=np.float32)

        ignore = False
        if self.min_size is not None and not self.test_mode:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True

        return dict(bbox=bbox, kps=kps, ignore=ignore, cat=cat)

    def load_annotations(self, ann_file):
        """加载新格式的标注文件"""
        name = None
        bbox_map = {}
        for line in open(ann_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                # 头部： # <img_path> <width> <height>
                parts = line[1:].strip().split()
                name = parts[0]
                width = int(parts[1])
                height = int(parts[2])
                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue
            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)

        print('origin image size', len(bbox_map))
        data_infos = []
        for name, item in bbox_map.items():
            width = item['width']
            height = item['height']
            objs = []
            cats = []
            for line in item['objs']:
                data = self._parse_ann_line(line)
                if data is not None:
                    objs.append(data)
                    cats.append(data['cat'])
            if len(objs) == 0 and not self.test_mode:
                continue
            if len(set(cats)) == 1:
                img_type = self.cat2label[cats[0]]
            else:
                img_type = len(self.CLASSES)
            data_infos.append(
                dict(filename=name, width=width, height=height, objs=objs, img_type=img_type))
        
        # For sampler use (see mmdet/datasets/samplers/weighted_sampler.py)
        self.type_indices = {i: [] for i in range(7)}
        for idx, info in enumerate(data_infos):
            self.type_indices[info['img_type']].append(idx)
        
        return data_infos

    def get_ann_info(self, idx):
        """生成与MMDetection兼容的标注字典"""
        data_info = self.data_infos[idx]

        bboxes = []
        keypointss = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in data_info['objs']:
            label = self.cat2label[obj['cat']]
            bbox = obj['bbox']
            keypoints = obj['kps']
            ignore = obj['ignore']
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
                keypointss.append(keypoints)

        # 转换为numpy数组，保证维度一致
        bboxes = np.array(bboxes, ndmin=2) if bboxes else np.zeros((0, 4))
        labels = np.array(labels) if labels else np.zeros((0,))
        keypointss = np.array(keypointss, ndmin=3) if keypointss else np.zeros((0, self.NK, 3))
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) if bboxes_ignore else np.zeros((0, 4))
        labels_ignore = np.array(labels_ignore) if labels_ignore else np.zeros((0,))

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            keypointss=keypointss.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann