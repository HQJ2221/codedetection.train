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

    CLASSES = ('qrcode', 'barcode', 'aruco')

    def __init__(self, min_size=None, **kwargs):
        self.NK = 4
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.gt_path = kwargs.get('gt_path')
        super(RetinaFaceDataset, self).__init__(**kwargs)

    def _parse_ann_line(self, line):
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)
        kps = np.zeros((self.NK, 3), dtype=np.float32)
        ignore = False
        if self.min_size is not None:
            assert not self.test_mode
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
        if len(values) > 4:
            if len(values) > 5:
                kps = np.array(
                    values[4:16], dtype=np.float32).reshape((self.NK, 3))
                for li in range(kps.shape[0]):
                    if (kps[li, :] == -1).all():
                        kps[li][2] = 0.0  # weight = 0, ignore
                    else:
                        assert kps[li][2] >= 0
                        kps[li][2] = 1.0  # weight

            else:
                if not ignore:
                    ignore = (values[4] == 1)
        else:
            assert self.test_mode

        return dict(bbox=bbox, kps=kps, ignore=ignore)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        name = None
        bbox_map = {}
        for line in open(ann_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])
                # read image lebel
                img_label = int(value[3])
                bbox_map[name] = dict(width=width, height=height, label=img_label, objs=[])
                continue
            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)
        print('origin image size', len(bbox_map))
        data_infos = []
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            # get images label
            img_label = item['label']
            vals = item['objs']
            objs = []
            for line in vals:
                data = self._parse_ann_line(line)
                if data is None:
                    continue
                # read label, turn to type name
                data['cat'] = self.CLASSES[img_label]
                objs.append(data)  # data is (bbox, kps, cat)
            if len(objs) == 0 and not self.test_mode:
                continue
            data_infos.append(
                dict(filename=name, width=width, height=height, objs=objs))
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
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
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            keypointss = np.zeros((0, self.NK, 3))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
            keypointss = np.array(keypointss, ndmin=3)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            keypointss=keypointss.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
