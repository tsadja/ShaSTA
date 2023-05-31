from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Save scene info")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results", 
                        default="data/nusc_detections/val_2hz")
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--split", type=str, default='mini_val')

    args = parser.parse_args()

    return args


def save_scene():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    if args.split == 'train':
        scene_names = splits.train
    elif args.split == 'mini_train':
        scene_names = splits.mini_train
    elif args.split == 'val':
        scene_names = splits.val
    elif args.split == 'mini_val':
        scene_names = splits.mini_val
    elif args.split == 'test':
        scene_names = splits.test

    scenes = {scene_name: list() for scene_name in scene_names}
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        scenes[scene_name].append(frame)

    del nusc

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    for scene_name in scene_names:
        if len(scenes[scene_name]) == 0:
            scenes.pop(scene_name)
    
    with open(os.path.join(args.work_dir, args.split+'_scenes_meta.json'), "w") as f:
        json.dump(scenes, f)

if __name__ == '__main__':
    save_scene()
