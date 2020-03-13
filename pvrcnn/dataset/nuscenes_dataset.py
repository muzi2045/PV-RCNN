from tqdm import tqdm
import pickle
import json
import numpy as np
import torch
import os
from copy import deepcopy
import os.path as osp
from pathlib import Path
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner, AnchorGenerator
from .augmentation import ChainedAugmentation
from .database_sampler import DatabaseBuilder

def udi_class_name_to_idx(class_name):
    CLASS_NAME_TO_IDX = {
        "car": 0,
        "bicycle": 1,
        "bus": 2,
        "construction_vehicle": 3,
        "motorcycle": 4,
        "pedestrian": 5,
        "traffic_cone": 6,
        "trailer": 7,
        "truck": 8,
        "barrier": 9,
    }
    if class_name not in CLASS_NAME_TO_IDX.keys():
        return -1
    return CLASS_NAME_TO_IDX[class_name]

class NuscenesDataset(Dataset):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    def __init__(self, cfg, split='v1.0-trainval'):
        super(NuscenesDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.load_annotations(cfg)
        self.max_sweeps = 10

    def __len__(self):
        return len(self.inds)

    # def read_splitfile(self, cfg):
    #     fpath = osp.join(cfg.DATA.SPLITDIR, f'{self.split}.txt')
    #     self.inds = np.loadtxt(fpath, dtype=np.int32).tolist()

    def read_cached_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.json')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        print(f'Found cached annotations: {fpath}')

    def cache_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.json')
        with open(fpath, 'wb') as f:
            json.dump(self.annotations, f)
            # pickle.dump(self.annotations, f)

    def load_annotations(self, cfg):
        self.read_splitfile(cfg)
        try:
            self.read_cached_annotations(cfg)
        except FileNotFoundError:
            os.makedirs(cfg.DATA.CACHEDIR, exist_ok=True)
            self.create_annotations(self.split,self.max_sweeps)
            self.cache_annotations(cfg)

    def _path_helper(self, folder, idx, suffix):
        return osp.join(self.cfg.DATA.ROOTDIR, folder, f'{idx:06d}.{suffix}')

    def _get_available_scenes(nusc):
        available_scenes = []
        print("total scene num:", len(nusc.scene))
        for scene in nusc.scene:
            scene_token = scene["token"]
            scene_rec = nusc.get('scene', scene_token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break
                if not sd_rec['next'] == "":
                    sd_rec = nusc.get('sample_data', sd_rec['next'])
                else:
                    has_more_frames = False
            if scene_not_exist:
                continue
            available_scenes.append(scene)
        print("exist scene num:", len(available_scenes))
        return available_scenes

    def create_annotations(self, version, max_sweeps):
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=cfg.DATA.ROOTDIR, verbose=True)
        from nuscenes.utils import splits
        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == "v1.0-test":
            train_scenes = splits.test
            val_scenes = []
        elif version == "v1.0-mini":
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError("unknown")
        root_path = Path(cfg.DATA.ROOTDIR)
        available_scenes = _get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        



        self.annotations = dict()
        for idx in tqdm(self.inds, desc='Generating annotations'):
            item = dict(
                velo_path=self._path_helper('velodyne_reduced', idx, 'bin'),
                calib=read_calib(self._path_helper('calib', idx, 'txt')),
                objects=read_label(self._path_helper('label_2', idx, 'txt')), idx=idx,
            )
            self.annotations[idx] = self.make_simple_objects(item)

    def make_simple_object(self, obj, calib):
        """Converts from camera to velodyne frame."""
        xyz = calib.C2V @ np.r_[calib.R0 @ obj.t, 1]
        box = np.r_[xyz, obj.w, obj.l, obj.h, -obj.ry]
        obj = dict(box=box, class_idx=obj.class_idx)
        return obj

    def make_simple_objects(self, item):
        objects = [self.make_simple_object(
            obj, item['calib']) for obj in item['objects']]
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


class NuscenesDatasetTrain(NuscenesDataset):
    """TODO: This class should certainly not need access to
        anchors. Find better place to instantiate target assigner."""

    def __init__(self, cfg):
        super(NuscenesDatasetTrain, self).__init__(cfg, split='train')
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

