import os, argparse, numpy as np, json
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='/home/ubuntu/workspace/tsadja/3d-mot/data/nuScenes')
parser.add_argument('--data_folder', type=str, default='/home/ubuntu/workspace/tsadja/3d-mot/data/nusc_detections')
parser.add_argument('--detection_folder', type=str, default='/home/ubuntu/workspace/tsadja/3d-mot/data/nusc_detections/mini_train/detection')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--file_name', type=str, default='cp-train.json')
parser.add_argument('--velo', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--split', type=str, default='mini_train')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()


def sample_result2bbox_array(sample):
    trans, size, rot, velocity, score = \
        sample['translation'], sample['size'],sample['rotation'], sample['velocity'], sample['detection_score']
    return trans + size + rot + velocity + [score]


def _second_det_to_nusc_box(box3d):
    box_list = []
    for i in range(box3d.shape[0]):
        velocity = (*box3d[i, 10:12], 0.0)
        quat = Quaternion(box3d[i, 6:10])
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            score = box3d[i, 12],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def _global_nusc_box_to_sensor(nusc, boxes, sample_token):
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
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)
        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)
        box_list.append(box)
    return box_list


def main(nusc, det_name, file_name, detection_folder, data_folder, mode, split):
    if split == 'train':
        scene_names = splits.train
    elif split == 'mini_train':
        scene_names = splits.mini_train
    elif split == 'val':
        scene_names = splits.val
    elif split == 'mini_val':
        scene_names = splits.mini_val
    elif split == 'test':
        scene_names = splits.test

    # dealing with the paths
    raw_file_path = os.path.join(data_folder, det_name, file_name)
    raw_file_path2 = os.path.join(data_folder, det_name, 'cp-val.json')
    indiv_output_folder = os.path.join(detection_folder, det_name, 'filtered_sensor_individual_frames')
    if not os.path.exists(indiv_output_folder):
        os.makedirs(indiv_output_folder)

    cls_output_folder = os.path.join(detection_folder, det_name, 'filtered_cls_individual_frames')
    if not os.path.exists(cls_output_folder):
        os.makedirs(cls_output_folder)
    
    # load the detection file
    print('LOADING RAW FILE')
    f = open(raw_file_path, 'r')
    det_data = json.load(f)['results']
    f.close()

    f = open(raw_file_path2, 'r')
    det_data2 = json.load(f)['results']
    f.close()

    for key, value in det_data2.items():
        det_data[key] = value

    # enumerate through all the frames
    # sample_keys = list(det_data.keys())
    f = open('/home/ubuntu/workspace/tsadja/3d-mot/data/nusc_detections/cp/mini_train_frame_info.json')
    sample_keys = json.load(f)
    sample_keys = list(sample_keys.keys())
    print(len(sample_keys))
    print('PROCESSING...')
    pbar = tqdm(total=len(sample_keys))
    for sample_key in sample_keys:
        # extract the bboxes and types
        sample_results = det_data[sample_key]

        bboxes = []
        filtered_sample_results = list()
        for sample in sample_results:
            if sample['detection_name'] not in NUSCENES_TRACKING_NAMES:
                continue
            filtered_sample_results.append(sample)
            bbox = sample_result2bbox_array(sample)
            bboxes.append(bbox)

        with open(os.path.join(cls_output_folder, sample_key+'.json'), "w") as f:
            json.dump(filtered_sample_results, f)

        bboxes = np.array(bboxes)
        bboxes = _second_det_to_nusc_box(bboxes)
        bboxes = _global_nusc_box_to_sensor(nusc, bboxes, sample_key)

        sensor_bboxes = []
        for box in bboxes:
            trans = list(box.center)
            size = list(box.wlh)
            rot = list(box.orientation)
            score = box.score
            velocity = list(box.velocity)
            curr_sensor_bbox = trans + size + rot + velocity[:2] + [score]
            sensor_bboxes.append(curr_sensor_bbox)

        with open(os.path.join(indiv_output_folder, sample_key+'.json'), "w") as f:
            json.dump(sensor_bboxes, f)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    # data_folder = os.path.join(args.data_folder, args.split + '_{:}'.format(args.mode))
    data_folder = args.data_folder
    nusc = NuScenes(version='v1.0-mini', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, args.det_name, args.file_name, args.detection_folder, data_folder, args.mode, args.split)