import sys
import pickle
import json
import random
import operator
import numpy as np
import os
import random

from functools import reduce
from pathlib import Path
from copy import deepcopy

from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from det3d.datasets.registry import DATASETS


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return np.array([yaw])


@DATASETS.register_module
class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index

    def __init__(
        self,
        info_path,
        root_path,
        det_path=None,
        labels_path=None,
        frame_info_path=None,
        cls_info_path=None,
        det_type=None,
        max_objects=500,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="v1.0-trainval",
        load_interval=1,
        fp_ratio=1,
        dead_trk_ratio=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names, det_type=det_type,
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        self._det_path = det_path
        self._labels_path = labels_path
        self._frame_info_path = frame_info_path
        self._cls_info_path = cls_info_path
        self._class_names = class_names
        self._det_type = det_type
        self._max_objects = max_objects
        self._fp_ratio = fp_ratio
        self._dead_trk_ratio = dead_trk_ratio

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        if not hasattr(self, "_frame_info") and self._frame_info_path is not None:
            f = open(self._frame_info_path)
            self._frame_info = json.load(f)

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16 

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):
        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        _nusc_infos_all = _nusc_infos_all[::self.load_interval]

        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            self._nusc_infos = []
            
            for class_name, cls_infos in zip(list(_cls_infos.keys()), list(_cls_infos.values())):
                if class_name in self._det_type:
                    self._nusc_infos += cls_infos
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

        self._map_frame_token_idx = dict()
        for idx, info in enumerate(self._nusc_infos):
            self._map_frame_token_idx[info['token']] = idx
    
    def get_frame_idx(self, frame_token):
        if frame_token in self._map_frame_token_idx.keys():
            return self._map_frame_token_idx[frame_token]
        else:
            return None

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx):
        info = self._nusc_infos[idx]
        curr_token = info['token']
        prev_token = self._frame_info[curr_token]['prev']
        idx_prev = self.get_frame_idx(prev_token)
        if idx_prev is None:
            prev_token = ''

        # dummies
        dummies = np.zeros((self._max_objects, 11))
        info['prev_det_boxes'] = dummies
        prev_keep = [i for i in range(self._max_objects)]
        info['num_prev_det_boxes'] = 0
        info['prev_cls_det_boxes'] = []

        if prev_token != '':
            f = open(os.path.join(self._det_path, prev_token+'.json'))
            prev_det_boxes = json.load(f)
            f.close()

            f = open(os.path.join(self._cls_info_path, prev_token+'.json'))
            prev_cls_info = json.load(f)
            f.close()

            if len(prev_det_boxes) > 0:
                prev_keep = []
                det_boxes = []
                time_diff = 1e-6*self._frame_info[curr_token]['timestamp'] - 1e-6*self._frame_info[curr_token]['prev_timestamp']
                for i, (b, cls_info) in enumerate(zip(prev_det_boxes, prev_cls_info)):
                    if self._det_type is not None and cls_info['detection_name'] not in self._det_type:
                        continue
                    score = cls_info['detection_score']
                    translation, size, rotation, velocity = np.array(b[:3]), np.array(b[3:6]), np.array(b[6:10]), np.array(b[10:12])
                    curr_det = np.concatenate((translation, size, quaternion_yaw(Quaternion(rotation)), 
                                                velocity, np.array([time_diff]), np.array([score])))
                    det_boxes.append(curr_det)
                    info['prev_cls_det_boxes'].append(cls_info)
                    prev_keep.append(i)

                if len(det_boxes) > 0:
                    if len(det_boxes) > self._max_objects:
                        rand_idx_prev_det_boxes = random.sample(range(len(det_boxes)), self._max_objects)
                        rand_idx_prev_det_boxes.sort()
                        det_boxes = [det_boxes[i] for i in rand_idx_prev_det_boxes]
                        info['prev_cls_det_boxes'] = [info['prev_cls_det_boxes'][i] for i in rand_idx_prev_det_boxes]
                        prev_keep = [prev_keep[i] for i in rand_idx_prev_det_boxes]
                    info['num_prev_det_boxes'] = len(det_boxes)
                    det_boxes = np.array(det_boxes)
                    info['prev_det_boxes'][:det_boxes.shape[0], :] = det_boxes

        # dummies
        dummies = np.zeros((self._max_objects, 11))
        info['det_boxes'] = dummies
        keep = [i for i in range(self._max_objects)]
        info['num_det_boxes'] = 0
        info['cls_det_boxes'] = []


        if self._det_path is not None:
            f = open(os.path.join(self._det_path, curr_token+'.json'))
            curr_det_boxes = json.load(f)
            f.close()

            f = open(os.path.join(self._cls_info_path, curr_token+'.json'))
            curr_cls_info = json.load(f)
            f.close()

            if len(curr_det_boxes) > 0:
                det_boxes = []
                keep = []
                time_diff = 1e-6*self._frame_info[curr_token]['timestamp'] - 1e-6*self._frame_info[curr_token]['prev_timestamp']
                for i, (b, cls_info) in enumerate(zip(curr_det_boxes, curr_cls_info)):
                    if self._det_type is not None and cls_info['detection_name'] not in self._det_type:
                        continue
                    score = cls_info['detection_score']
                    translation, size, rotation, velocity = np.array(b[:3]), np.array(b[3:6]), np.array(b[6:10]), np.array(b[10:12])
                    curr_det = np.concatenate((translation, size, quaternion_yaw(Quaternion(rotation)), 
                                                velocity, np.array([time_diff]), np.array([score])))
                    det_boxes.append(curr_det)
                    info['cls_det_boxes'].append(cls_info)
                    keep.append(i)

                if len(det_boxes) > 0:
                    if len(det_boxes) > self._max_objects:
                        rand_idx_det_boxes = random.sample(range(len(det_boxes)), self._max_objects)
                        rand_idx_det_boxes.sort()
                        det_boxes = [det_boxes[i] for i in rand_idx_det_boxes]
                        info['cls_det_boxes'] = [info['cls_det_boxes'][i] for i in rand_idx_det_boxes]
                        keep = [keep[i] for i in rand_idx_det_boxes]
                    # if self.test_mode:
                    #     for i, cls_info in enumerate(info['cls_det_boxes']):
                    #         cls_info['ego_box'] = det_boxes[i]
                    #         info['cls_det_boxes'][i] = cls_info
                    info['num_det_boxes'] = len(det_boxes)
                    det_boxes = np.array(det_boxes)
                    info['det_boxes'][:det_boxes.shape[0], :] = det_boxes
        

        # if self._labels_path is not None:
        if not self.test_mode:  # if training
            curr_labels = np.load(os.path.join(self._labels_path, curr_token+'.npz'), allow_pickle=True)
            info['gt'] = np.zeros((self._max_objects+2, self._max_objects+2))
            # info['gt'][-2:, :] = 0
            # info['gt'][:, :len(keep)] = 0
            if prev_token != '':
                info['gt'][:len(prev_keep), :] = 0
                # add matching matrix
                temp = curr_labels['matched'][prev_keep]
                temp = temp[:,keep]
                info['gt'][:len(prev_keep), :len(keep)] = temp
                # dead tracks
                info['gt'][:len(prev_keep), -2] = curr_labels['matched'][prev_keep, -2]
                # FNs
                info['gt'][:len(prev_keep), -1] = 1 - info['gt'][:len(prev_keep), :].sum(axis=1)

                dead_trk = info['gt'][:len(prev_keep), -2]
                fn = info['gt'][:len(prev_keep), -1]
                prev_tp = info['gt'][:len(prev_keep), :-2].sum(axis=1) + fn
                prev_tp_idx = list(np.nonzero(prev_tp==1)[0])
                dead_trk_idx = list(np.nonzero(dead_trk==1)[0])

                # print(len(prev_tp_idx), len(dead_trk_idx))
                random.shuffle(dead_trk_idx)
                num_keep_dead_trk = int(self._dead_trk_ratio*prev_tp.sum())
                keep_dead_trk_idx = dead_trk_idx[:num_keep_dead_trk]
                temp_prev_keep = keep_dead_trk_idx+prev_tp_idx
                temp_prev_keep.sort()
                
                info['num_prev_det_boxes'] = len(temp_prev_keep)
                info['gt'][:len(temp_prev_keep), :] = info['gt'][temp_prev_keep,:]
                info['gt'][len(temp_prev_keep):-2, :] = np.zeros((self._max_objects-len(temp_prev_keep), self._max_objects+2))
            
            # newborns
            newborn = curr_labels['newborn'][keep]
            info['gt'][-2, :len(keep)] = newborn
            # FPs
            fp = 1-info['gt'][:,:len(keep)].sum(axis=0)
            info['gt'][-1, :len(keep)] = fp 

            tp = info['gt'][:-1,:len(keep)].sum(axis=0)
            tp_idx = list(np.nonzero(tp==1)[0])
            fp_idx = list(np.nonzero(fp==1)[0])
            # print(len(tp_idx), len(fp_idx))
            random.shuffle(fp_idx)
            num_keep_fp = int(self._fp_ratio*tp.sum())
            keep_fp_idx = fp_idx[:num_keep_fp]
            temp_keep = keep_fp_idx+tp_idx
            temp_keep.sort()

            info['num_det_boxes'] = len(temp_keep)
            info['gt'][:, :len(temp_keep)] = info['gt'][:, temp_keep]
            info['gt'][:, len(temp_keep):-2] = np.zeros((self._max_objects+2, self._max_objects-len(temp_keep)))



        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": curr_token,
                # "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": self.virtual 
        }

        data, _ = self.pipeline(res, info)

        if prev_token != '':
            res_prev = {
                "lidar": {
                    "type": "lidar",
                    "points": None,
                    "nsweeps": self.nsweeps,
                    "annotations": None,
                },
                "metadata": {
                    "image_prefix": self._root_path,
                    "num_point_features": self._num_point_features,
                    "token": prev_token,
                    # "token": info["token"],
                },
                "calib": None,
                "cam": {},
                "mode": "val" if self.test_mode else "train",
                "virtual": self.virtual 
            }

            info_prev = self._nusc_infos[idx_prev]
            data_temp, _ = self.pipeline(res_prev, info_prev)
            data_prev = {}
            data_prev["metadata"] = data_temp["metadata"]
            for key, value in data_temp.items():
                if key in ["points", "voxels", "shape", "num_points", "num_voxels", "coordinates",]:
                    data_prev[key] = value
        else:
            info_temp = self._nusc_infos[idx]
            data_temp, _ = self.pipeline(res, info_temp)
            data_prev = {}
            data_prev["metadata"] = data_temp["metadata"]
            for key, value in data_temp.items():
                if key in ["points", "voxels", "shape", "num_points", "num_voxels", "coordinates",]:
                    data_prev[key] = value

        return (data, data_prev)

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res, None
