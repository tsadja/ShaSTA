import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint
import json

from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='/home/ubuntu/workspace/tsadja/3d-mot/data/nuScenes')
parser.add_argument('--output_folder', type=str, default='/home/ubuntu/workspace/tsadja/3d-mot/data/nusc_detections')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def instance_info2bbox_array(info):
    ## CHANGED TO WORK FOR TRAINING SPLIT
    translation = info['translation']
    size = info['size']
    rotation = info['rotation']
    return translation + size + rotation


def _second_det_to_nusc_box(box3d):
    # box3d = detection["box3d_lidar"].detach().cpu().numpy()
    # scores = detection["scores"].detach().cpu().numpy()
    # labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            # label=labels[i],
            # score=1,
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list


def main(nusc, scene_names, gt_data, gt_folder):
    indiv_frames_path = os.path.join(gt_folder, 'correct_individual_frames')
    if not os.path.exists(indiv_frames_path):
        os.makedirs(indiv_frames_path)


    pbar = tqdm(total=len(scene_names))
    # overall_idx = 0
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']
        if scene_name not in scene_names:
            continue

        first_sample_token = scene_info['first_sample_token']
        last_sample_token = scene_info['last_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        if args.mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP']
        elif args.mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        
        IDS, inst_types, bboxes = list(), list(), list()
        while True:
            for db_idx, curr in enumerate(gt_data): 
                if cur_sample_token == curr['token']:
                    break
            db_info = gt_data[db_idx]
            # for gt_box in db_info['gt_boxes']:
            boxes = _second_det_to_nusc_box(db_info['gt_boxes'])
            boxes = _lidar_nusc_box_to_global(nusc, boxes, cur_sample_token)

            db_boxes = list()
            for box in boxes:
                db_boxes.append(box.center.tolist() + box.wlh.tolist() + box.orientation.elements.tolist())


            frame_ids, frame_types, frame_bboxes = list(), list(), list()
            gt_dict =  {'ids':frame_ids, 'types':frame_types, 'bboxes':frame_bboxes}
            if args.mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token)
                ann_tokens = frame_data['anns']
                for ann in ann_tokens:
                    instance = nusc.get('sample_annotation', ann)
                    if (instance['num_lidar_pts'] + instance['num_radar_pts'])>0:
                        frame_ids.append(instance['instance_token'])
                        frame_types.append(instance['category_name'])
                        frame_bboxes.append(instance_info2bbox_array(instance))
                
                # overlap = 0
                # for frame_bbox in frame_bboxes:
                #     if frame_bbox in db_boxes:
                #         overlap += 1
                # # print(len(db_boxes), overlap)
                print(len(db_boxes), len(frame_bboxes))
                print(db_boxes[0], frame_bboxes[0])

                with open(os.path.join(indiv_frames_path, cur_sample_token+'.json'), "w") as f:
                    gt_dict = {'frame_ids':frame_ids, 'frame_types':frame_types, 'frame_bboxes':frame_bboxes}
                    json.dump(gt_dict, f)
            elif args.mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                lidar_data = nusc.get('sample_data', cur_sample_token)
                instances = nusc.get_boxes(lidar_data['token'])
                for inst in instances:
                    frame_ids.append(inst.token)
                    frame_types.append(inst.name)
                    frame_bboxes.append(instance_info2bbox_array(inst))
            
            IDS.append(frame_ids)
            inst_types.append(frame_types)
            bboxes.append(frame_bboxes)

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            # overall_idx += 1
            if cur_sample_token == '':
                break

        np.savez_compressed(os.path.join(gt_folder, '{:}.npz'.format(scene_name)), 
            ids=IDS, types=inst_types, bboxes=bboxes)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    if args.mode == '20hz':
        output_folder = os.path.join(args.output_folder, 'validation_20hz')
    elif args.mode == '2hz':
        output_folder = os.path.join(args.output_folder, args.split+'_2hz')

    gt_folder = os.path.join(output_folder, 'correct_gt_info')
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    val_scene_names = splits.create_splits_scenes()[args.split]
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)

    if args.split == 'train':
        db = os.path.join(args.output_folder, 'infos_train_10sweeps_withvelo_filter_True.pkl')
    elif args.split == 'val':
        db = os.path.join(args.output_folder, 'infos_val_10sweeps_withvelo_filter_True.pkl')

    with open(db, 'rb') as f:
        data = pickle.load(f)
    
    main(nusc, val_scene_names, data, gt_folder)