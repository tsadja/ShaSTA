from nuscenes import NuScenes
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.render import TrackingRenderer
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.data_classes import TrackingBox
from temp_nusc import TempNuScenes
from nuscenes.utils import splits
import os, argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='data/nuScenes')
parser.add_argument('--save_path', type=str, default='work_dir/visualize')
parser.add_argument('--scene_name', type=str, default='scene-0270')
parser.add_argument('--render_class', type=str, default='car')
parser.add_argument('--track_result_path', type=str, default='results/val_tracking_result.json')
parser.add_argument('--split', type=str, default='val')
args = parser.parse_args()


def main(nusc, lidar_save_dir, cam_save_dir, scene_name, render_class):
    cfg = track_configs("tracking_nips_2019")
    pred_boxes, meta = load_prediction(args.track_result_path, cfg.max_boxes_per_sample, TrackingBox, verbose=True)
    pred_boxes = add_center_dist(nusc, pred_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=True)
    tracks_pred = create_tracks(pred_boxes, nusc, split, gt=False)

    for scene_info in nusc.scene:
        if scene_info['name'] != scene_name:
            continue

        scene_token = scene_info['token']
        scene_tracks_pred = tracks_pred[scene_token]

        first_sample_token = scene_info['first_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        cur_sample_token = deepcopy(first_sample_token)
        while True:
            if cur_sample_token == '':
                break
            
            frame_data = nusc.get('sample', cur_sample_token)
            timestamp = frame_data['timestamp']

            lidar_out_path = os.path.join(lidar_save_dir, str(timestamp))
            cam_out_path = os.path.join(cam_save_dir, str(timestamp))
            tracks = scene_tracks_pred[timestamp]
            tracks = [t for t in tracks if t.tracking_name==render_class]

            nusc.render_sample_data(frame_data['data']['LIDAR_TOP'], nsweeps=10, underlay_map=True, out_path=lidar_out_path, tracks=tracks)
            nusc.render_sample_data(frame_data['data']['CAM_FRONT'], out_path=cam_out_path, tracks=tracks)


            cur_sample_token = frame_data['next']

if __name__ == '__main__':
    lidar_save_dir = os.path.join(args.save_path, 'lidar', args.scene_name)
    cam_save_dir = os.path.join(args.save_path, 'front-camera', args.scene_name)
    if not os.path.exists(lidar_save_dir):
        os.makedirs(lidar_save_dir)
    if not os.path.exists(cam_save_dir):
        os.makedirs(cam_save_dir)
    
    nusc = TempNuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)

    main(nusc, lidar_save_dir, cam_save_dir, args.scene_name, args.render_class)
