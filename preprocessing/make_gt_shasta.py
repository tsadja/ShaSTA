import os, argparse, numpy as np, multiprocessing, nuscenes, json
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from detection_nms import load_dets, nu_array2mot_bbox
from gt_association.associate import associate

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='gt_shasta')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--dist_thres', type=float, default=2.0)
parser.add_argument('--dist_type', type=str, default='l2')
parser.add_argument('--raw_data_folder', type=str, default='data/nuScenes')
parser.add_argument('--data_folder', type=str, default='data/nusc_preprocessed/train_2hz')
parser.add_argument('--gt_folder', type=str, default='data/nusc_preprocessed/train_2hz/gt_info')
parser.add_argument('--det_folder', type=str, default='data/nusc_preprocessed/train_2hz/detections')
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--dataset', type=str, default='nuscenes', choices=['waymo', 'nuscenes'])
args = parser.parse_args()

def load_gt_bboxes(gt_folder, data_folder, segment_name):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)),
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
        allow_pickle=True)

    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']

    frame_num = len(bboxes)
    for i in range(frame_num):
        for j in range(len(bboxes[i])):
            if args.dataset == 'nuscenes':
                bboxes[i][j] = nu_array2mot_bbox(bboxes[i][j])
            else:
                bboxes[i][j] = BBox.array2bbox(bboxes[i][j])

    return bboxes, ids, inst_types

def main(nusc, scene_names, det_name, thres, data_folder, gt_folder, det_folder, name):

    npz_path = os.path.join(data_folder, name, det_name, 'individual_frames')
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    f = open('data/nusc_preprocssed/train_frame_info.json')
    sample_keys = json.load(f)
    sample_keys = list(sample_keys.keys())
    
    gt_trans = dict()

    pbar = tqdm(total=len(scene_names))
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']

        if scene_name not in scene_names:
            continue
        
        dets, inst_types = load_dets(os.path.join(det_folder, det_name), data_folder, scene_name)
        gt_bboxes, gt_ids, gt_inst_types = load_gt_bboxes(gt_folder, data_folder, scene_name)

        first_sample_token = scene_info['first_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        cur_sample_token = deepcopy(first_sample_token)
        frame_index = 0
        while True:
            if cur_sample_token not in sample_keys:
                # clean up and prepare for the next
                cur_sample_token = frame_data['next']
                frame_index += 1
                if cur_sample_token == '':
                    break
                continue

            frame_data = nusc.get('sample', cur_sample_token)
            frame_dets = dets[frame_index]
            frame_types = inst_types[frame_index]
            frame_gt = gt_bboxes[frame_index]
            frame_gt_types = gt_inst_types[frame_index]
            frame_gt_ids = gt_ids[frame_index]
            K = len(frame_dets)

            _, _, _, _, _, _, _, tp_ind_pairs, _, fn_inds = associate(frame_gt, frame_gt_types, 
                                                                frame_dets, frame_types, threshold=thres)


            prev_sample_token = frame_data['prev']

            if prev_sample_token == '':
                matched = None
                newborn = np.zeros((K,))
                for k in range(K):
                    if k in tp_ind_pairs.keys():
                        newborn[k] = 1
            else:
                prev_dets = dets[frame_index-1]
                prev_types = inst_types[frame_index-1]
                N = len(prev_dets)
                prev_gt = gt_bboxes[frame_index-1]
                prev_gt_types = gt_inst_types[frame_index-1]
                prev_gt_ids = gt_ids[frame_index-1]

                _, _, _, _, _, _, _, prev_tp_ind_pairs, prev_fp_inds, _ = associate(prev_gt, prev_gt_types, 
                                                                        prev_dets, prev_types, threshold=thres)

                
                matched = np.zeros((N,K+2))
                newborn = np.zeros((K,))

                prev_tp_ids = list()
                prev_tp_dets = list()
                prev_tp_types = list()

                for prev_idx, prev_gt_idx in prev_tp_ind_pairs.items():
                    prev_tp_ids.append(prev_gt_ids[prev_gt_idx])
                    prev_tp_dets.append(prev_dets[prev_idx])
                    prev_tp_types.append(prev_types[prev_idx])


                prev_tp_idx = list(prev_tp_ind_pairs.keys())
                matched_prev_tp_ids = list()

                # Look at current frame TP pairs: curr det idx --> curr frame gt idx
                for curr_idx, gt_idx in tp_ind_pairs.items():
                    # get curr frame GT id associated with TP det
                    gt_id = frame_gt_ids[gt_idx]
                    # if curr TP GT id was in previous TP id, then match
                    if gt_id in prev_tp_ids:
                        matched_prev_tp_ids.append(gt_id)
                        prev_tp_id_idx = prev_tp_ids.index(gt_id)
                        prev_idx = prev_tp_idx[prev_tp_id_idx]
                        matched[prev_idx,curr_idx] = 1
                    else:
                        newborn[curr_idx] = 1
                
                for prev_tp_id_idx, prev_tp_id in enumerate(prev_tp_ids):
                    if prev_tp_id not in matched_prev_tp_ids:
                        prev_idx = prev_tp_idx[prev_tp_id_idx]
                        # previous TPs that are unmatched and have ID in current frame
                        if prev_tp_id in frame_gt_ids:
                            gt_id_idx = frame_gt_ids.index(prev_tp_id)
                            # check if unmatched previous TP ID matches a FN GT ID for current frame
                            if gt_id_idx in fn_inds:
                                matched[prev_idx, -1] = 1 # FN track
                
                # dead tracks: all previous detections that are FPs or previous TPs with no ID match in current frame
                matched[:, -2] = 1 - matched.sum(axis=1)

            matched_list = [] if matched is None else matched.tolist()
            newborn_list = [] if newborn is None else newborn.tolist()
            gt_trans[cur_sample_token] = {'matched': matched_list, 'newborn': newborn_list}
            np.savez_compressed(os.path.join(npz_path, cur_sample_token + '.npz'), matched=matched, newborn=newborn)

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            frame_index += 1
            if cur_sample_token == '':
                break
        pbar.update(1)
    pbar.close()

    return



if __name__ == '__main__':
    scene_names = splits.create_splits_scenes()[args.split]
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, scene_names, args.det_name, args.dist_thres, args.data_folder, args.gt_folder, args.det_folder, args.name)