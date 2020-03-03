from tqdm import tqdm
import pickle
import numpy as np
import torch
import os
import json
from copy import deepcopy
import os.path as osp
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner, AnchorGenerator
from .kitti_utils import read_calib, read_label, read_velo
from .augmentation import ChainedAugmentation
from .database_sampler import DatabaseBuilder


def read_label(label_filename):
    with open(label_path, encoding='utf-8') as f:
            res = f.read()
    result = json.loads(res)
    boxes = result["elem"]
    boxes_list = []
    for box in boxes:
        box_id = box["id"]
        box_loc = box["position"]
        box_size = box["size"]
        box_yaw = box["yaw"]
        box = np.array([box_id, box_loc["x"], box_loc["y"], box_loc["z"],
                        box_size["width"], box_size["depth"], box_size["height"],
                        box_yaw], dtype=np.float)
        boxes_list.append(box)
    return boxes_list

def read_velo(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


class UDIDataset(Dataset):

    def __init__(self, cfg, split='val'):
        super(UDIDataset, self).__init__()
        self.cfg = cfg
        # self.split = split
        self.load_annotations(cfg)

    def __len__(self):
        return len(self.inds)

    def read_inds(self, cfg):
        inds = []
        lidar_root_path = osp.join(cfg.DATA.ROOTDIR, "lidar")
        filenames = os.listdir(lidar_root_path)
        for filename in tqdm(filenames):
            index = filename.split(".")[0]
            inds.append(int(index))
        self.inds = inds.sort()

    def read_cached_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, 'infos_udi_train.pkl')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        print(f'Found cached annotations: {fpath}')

    def cache_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, 'infos_udi_train.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def load_annotations(self, cfg):
        self.read_inds(cfg)
        try:
            self.read_cached_annotations(cfg)
        except FileNotFoundError:
            os.makedirs(cfg.DATA.CACHEDIR, exist_ok=True)
            self.create_annotations()
            self.cache_annotations(cfg)

    def _path_helper(self, folder, idx, suffix):
        return osp.join(self.cfg.DATA.ROOTDIR, folder, f'{idx}.{suffix}')

    def create_annotations(self):
        self.annotations = dict()
        for idx in tqdm(self.inds, desc='Generating annotations'):
            item = dict(
                velo_path=self._path_helper('lidar', idx, 'bin'),
                # calib=read_calib(self._path_helper('calib', idx, 'txt')),
                objects=read_label(osp.join(self.cfg.DATA.ROOTDIR, 'label', f'{idx}_bin.json')),
                idx= idx
            self.annotations[idx] = self.make_simple_objects(item)

    def make_simple_object(self, obj):
        ## obj [class_id, x, y, z, w, l, h, yaw]
        box = obj[1:]
        obj = dict(box=box, class_idx=obj[0])
        return obj

    def make_simple_objects(self, item):
        objects = [self.make_simple_object(obj) for obj in item['objects']]
        item['boxes'] = np.stack([obj['box'] for obj in objects])
        item['class_idx'] = np.r_[[obj['class_idx'] for obj in objects]]
        return item

    def filter_bad_objects(self, item):
        class_idx = item['class_idx'][:, None]
        _, wlh, _ = np.split(item['boxes'], [3, 6], 1)
        keep = ((class_idx != -1) & (wlh > 0)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def filter_out_of_bounds(self, item):
        xyz, _, _ = np.split(item['boxes'], [3, 6], 1)
        lower, upper = np.split(self.cfg.GRID_BOUNDS, [3])
        keep = ((xyz >= lower) & (xyz <= upper)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def to_torch(self, item):
        item['points'] = np.float32(item['points'])
        item['boxes'] = torch.FloatTensor(item['boxes'])
        item['class_idx'] = torch.LongTensor(item['class_idx'])

    def drop_keys(self, item):
        for key in ['velo_path', 'objects', 'calib']:
            item.pop(key)

    def preprocessing(self, item):
        self.to_torch(item)

    def __getitem__(self, idx):
        item = deepcopy(self.annotations[self.inds[idx]])
        item['points'] = read_velo(item['velo_path'])
        self.preprocessing(item)
        self.drop_keys(item)
        return item


class UDIDatasetTrain(UDIDataset):
    """TODO: This class should certainly not need access to
        anchors. Find better place to instantiate target assigner."""

    def __init__(self, cfg):
        super(UDIDatasetTrain, self).__init__(cfg, split='train')
        anchors = AnchorGenerator(cfg).anchors
        DatabaseBuilder(cfg, self.annotations)
        self.target_assigner = ProposalTargetAssigner(cfg, anchors)
        self.augmentation = ChainedAugmentation(cfg)

    def preprocessing(self, item):
        """Applies augmentation and assigns targets."""
        self.filter_bad_objects(item)
        points, boxes, class_idx = self.augmentation(
            item['points'], item['boxes'], item['class_idx'])
        item.update(dict(points=points, boxes=boxes, class_idx=class_idx))
        self.filter_out_of_bounds(item)
        self.to_torch(item)
        self.target_assigner(item)
