import json
import os
import sys
import argparse
from nuscenes import NuScenes
from nuscenes.eval.tracking.evaluate import TrackingEval 
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.utils import splits

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save merged results", 
                        default="work_dir/final_val_results")
    parser.add_argument("--checkpoint_name", help="name merged checkpoint", 
                        default="merged_cp_val.json")
    parser.add_argument("--split", type=str, default='val')

    args = parser.parse_args()

    return args


args = parse_args()
output_path = args.work_dir
paths = [os.path.join(output_path, trk_name, 'cp_'+ args.split + '.json') for trk_name in NUSCENES_TRACKING_NAMES]

res_path = os.path.join(output_path, args.checkpoint_name)
output_data = {
    'meta': {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False
    }, 
    'results': {}
}

for path in paths:
    print(path)
    with open(path) as f:
        data = json.load(f)
    for sample_token in data['results']:
        if sample_token not in output_data['results']:
            output_data['results'][sample_token] = []

        output_data['results'][sample_token].extend(data['results'][sample_token])

with open(res_path, 'w') as outfile:
    json.dump(output_data, outfile)
