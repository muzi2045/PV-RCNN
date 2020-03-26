from tqdm import tqdm
import pickle
import json
import numpy as np
import torch
import os
from copy import deepcopy
import os.path as osp
from pathlib import Path
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner
from .augmentation import ChainedAugmentation, DatabaseBuilder
# from .database_sampler import DatabaseBuilder

def nuscenes_class_name_to_idx(class_name):
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
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.emergency.police': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.personal_mobility': 'pedestrian',
        'human.pedestrian.stroller': 'pedestrian',
        'human.pedestrian.wheelchair': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.barrier': 'barrier'
    }
    Nuscenes_classes = ['car', 
                        'bicycle', 
                        'bus', 
                        'construction_vehicle', 
                        'motorcycle',
                        'pedestrian', 
                        'traffic_cone', 
                        'trailer',
                        'truck',
                        'barrier']
    def __init__(self, cfg, split='v1.0-trainval', max_sweeps=10):
        super(NuscenesDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.max_sweeps = max_sweeps
        self.load_annotations(cfg)
        

    def __len__(self):
        # only for training
        return len(self.train_infos)

    def read_cached_annotations(self, cfg):
        train_fpath = osp.join(cfg.DATA.CACHEDIR, 'train_infos.pkl')
        val_fpath = osp.join(cfg.DATA.CACHEDIR, 'val_infos.pkl')
        with open(train_fpath, 'rb') as f:
            self.train_infos = pickle.load(f)
        with open(val_fpath, 'rb') as f:
            self.val_infos = pickle.load(f)
        print(f'Found cached train infos: {train_fpath} {val_fpath}')

    def cache_annotations(self, cfg):
        train_path = osp.join(cfg.DATA.CACHEDIR, 'train_infos.pkl')
        val_path = osp.join(cfg.DATA.CACHEDIR, 'val_infos.pkl')
        with open(train_path, 'wb') as f:
            pickle.dump(self.train_infos, f)
        with open(val_path, 'wb') as f:
            pickle.dump(self.val_infos, f)

    def load_annotations(self, cfg):
        # self.read_splitfile(cfg)
        try:
            self.read_cached_annotations(cfg)
        except FileNotFoundError:
            os.makedirs(cfg.DATA.CACHEDIR, exist_ok=True)
            self.create_annotations(self.split, self.max_sweeps)
            self.cache_annotations(cfg)

    def _path_helper(self, folder, idx, suffix):
        return osp.join(self.cfg.DATA.ROOTDIR, folder, f'{idx:06d}.{suffix}')

    def _get_available_scenes(self, nusc):
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
        nusc = NuScenes(version=version, dataroot=self.cfg.DATA.ROOTDIR, verbose=True)
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
        root_path = Path(self.cfg.DATA.ROOTDIR)
        available_scenes = self._get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ])
        val_scenes = set([
            available_scenes[available_scene_names.index(s)]["token"]
            for s in val_scenes
        ])
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
        self.train_infos = dict()
        self.val_infos = dict()
        index = 0
        
        for sample in tqdm(nusc.sample, desc="Generating train infos..."):
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get('sample_data', lidar_token)
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
            assert Path(lidar_path).exists(), ("some lidar file miss...+_+")
            item = {
                "lidar_path": lidar_path,
                "token": sample["token"],
                "sweeps": [],
                "calib": cs_record,
                "ego_pose": pose_record,
                "timestamp": sample["timestamp"]
            }
            l2e_t = cs_record['translation']
            l2e_r = cs_record['rotation']
            e2g_t = pose_record['translation']
            e2g_r = pose_record['rotation']
            
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec['prev'] == "":
                    sd_rec = nusc.get('sample_data', sd_rec['prev'])
                    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                    lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                    sweep = {
                        "lidar_path": lidar_path,
                        "sample_data_token": sd_rec['token'],
                        "timestamp": sd_rec["timestamp"]
                    }
                    l2e_r_s = cs_record["rotation"]
                    l2e_t_s = cs_record["translation"]
                    e2g_r_s = pose_record["rotation"]
                    e2g_t_s = pose_record["translation"]
                    ## sweep->ego->global->ego'->lidar
                    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    )
                    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    )
                    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T 
                    sweep["sweep2lidar_rotation"] = R.T 
                    sweep["sweep2lidar_translation"] = T
                    sweeps.append(sweep)
                else:
                    break
            item["sweeps"] = sweeps

            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] 
                            for b in boxes]).reshape(-1, 1)
            names = [b.name for b in boxes]
            class_idx = []
            for i in range(len(names)):
                if names[i] in NuscenesDataset.NameMapping:
                    names[i] = NuscenesDataset.NameMapping[names[i]]
                    class_idx.append(nuscenes_class_name_to_idx(names[i]))
            names = np.array(names)
            class_idx = np.array(class_idx)

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            ## In Nuscenes Dataset, need to filter some rare class
            mask = np.array([NuscenesDataset.Nuscenes_classes.count(s) > 0 for s in names], dtype=np.bool_)
            gt_boxes = gt_boxes[mask]
            names = names[mask]
            
            # assert len(gt_boxes) == len(
            #     annotations), f"{len(gt_boxes)}, {len(annotations)}."
            assert len(gt_boxes) == len(names), f"{len(gt_boxes)}, {len(names)}"
            
            assert len(gt_boxes) == len(class_idx), f"{len(gt_boxes)}, {len(class_idx)}"

            item["boxes"] = gt_boxes
            item["class_idx"] = class_idx
            item["names"] = names
            item["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]
            )
            item["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]
            )
            if sample["scene_token"] in train_scenes:
                self.train_infos[index] = item
            else:
                self.val_infos[index] = item
            index += 1
        

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
        item['box_ignore'] = np.full(keep.sum(), False)

    def to_torch(self, item):
        item['points'] = np.float32(item['points'])
        item['boxes'] = torch.FloatTensor(item['boxes'])
        item['class_idx'] = torch.LongTensor(item['class_idx'])
        item['box_ignore'] = torch.BoolTensor(item['box_ignore'])

    def drop_keys(self, item):
        for key in ['velo_path', 'objects', 'calib']:
            item.pop(key)

    def preprocessing(self, item):
        self.to_torch(item)

    def read_nu_lidar(self, item):
        lidar_path = Path(item["lidar_path"])
        points = np.fromfile(
            str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = item["timestamp"] / 1e6
        for sweep in item["sweeps"]:
            points_sweep = np.fromfile(
                str(sweep["lidar_path"]), dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                "sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)
        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        return points

    def __getitem__(self, idx):
        item = deepcopy(self.train_infos[idx])
        item['points'] = self.read_nu_lidar(item)
        self.preprocessing(item)
        # self.drop_keys(item)
        return item


class NuscenesDatasetTrain(NuscenesDataset):
    """TODO: This class should certainly not need access to
        anchors. Find better place to instantiate target assigner."""

    def __init__(self, cfg):
        super(NuscenesDatasetTrain, self).__init__(cfg, split='v1.0-trainval', max_sweeps=10)
        DatabaseBuilder(cfg, self.train_infos)
        DatabaseBuilder(cfg, self.val_infos)
        self.augmentation = ChainedAugmentation(cfg)
        self.target_assigner = ProposalTargetAssigner(cfg)

    def preprocessing(self, item):
        """Applies augmentation and assigns targets."""
        # self.filter_bad_objects(item)
        points, boxes, class_idx = self.augmentation(
            item['points'], item['boxes'], item['class_idx'])
        item.update(dict(points=points, boxes=boxes, class_idx=class_idx))
        self.filter_out_of_bounds(item)
        self.to_torch(item)
        self.target_assigner(item)

