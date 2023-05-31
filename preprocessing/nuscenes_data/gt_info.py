import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint
import json


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='data/nuScenes')
parser.add_argument('--output_folder', type=str, default='data/nusc_preprocessed')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
args = parser.parse_args()


def instance_info2bbox_array(nusc, info, sample_token):
    ## CHANGED TO WORK FOR TRAINING SPLIT
    velocity = list(nusc.box_velocity(sample_token))
    translation = info['translation']
    size = info['size']
    rotation = info['rotation']
    return translation + size + rotation + velocity[:2]


def main(nusc, scene_names, root_path, gt_folder):
    indiv_frames_path = os.path.join(gt_folder, 'individual_frames')
    if not os.path.exists(indiv_frames_path):
        os.makedirs(indiv_frames_path)


    pbar = tqdm(total=len(scene_names))
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
        
        frame_index = 0
        IDS, inst_types, bboxes = list(), list(), list()
        while True:
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
                        frame_bboxes.append(instance_info2bbox_array(nusc, instance, ann))
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
            if cur_sample_token == '':
                break

        np.savez_compressed(os.path.join(gt_folder, '{:}.npz'.format(scene_name)), 
            ids=IDS, types=inst_types, bboxes=bboxes)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    # if args.mode == '20hz':
    #     output_folder = os.path.join(args.output_folder, 'validation_20hz')
    # elif args.mode == '2hz':
    #     output_folder = os.path.join(args.output_folder, args.split+'_2hz')

    output_folder = os.path.join(args.output_folder, args.split+'_'+args.mode)

    gt_folder = os.path.join(output_folder, 'gt_info')
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    val_scene_names = splits.create_splits_scenes()[args.split]
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, gt_folder)