import os, argparse, numpy as np, multiprocessing, nuscenes, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='data/nuScenes')
parser.add_argument('--data_folder', type=str, default='data/nusc_preprocessed')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()


def main(nusc, scene_names, save_path, split):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    frame_info = dict()

    pbar = tqdm(total=len(scene_names))
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']

        if scene_name not in scene_names:
            continue

        first_sample_token = scene_info['first_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        cur_sample_token = deepcopy(first_sample_token)
        while True:
            frame_data = nusc.get('sample', cur_sample_token)
            cur_timestamp = frame_data['timestamp']
            
            prev_sample_token = frame_data['prev']
            next_sample_token = frame_data['next']


            prev_timestamp = cur_timestamp if prev_sample_token == '' else nusc.get('sample', prev_sample_token)['timestamp']
            next_timestamp = cur_timestamp if next_sample_token == '' else nusc.get('sample', next_sample_token)['timestamp']

            frame_info[cur_sample_token] = {'prev': prev_sample_token, 'next': next_sample_token, 'timestamp': cur_timestamp, 
                                            'prev_timestamp': prev_timestamp, 'next_timestamp': next_timestamp}

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            if cur_sample_token == '':
                break

        pbar.update(1)
    pbar.close()
    
    with open(os.path.join(save_path, split+'_frame_info.json'), "w") as f:
        json.dump(frame_info, f)

    return


if __name__ == '__main__':
    if args.test:
        scene_names = splits.create_splits_scenes()['test']
        nusc = NuScenes(version='v1.0-test', dataroot=args.raw_data_folder, verbose=True)
        args.split = 'test'
    else:
        scene_names = splits.create_splits_scenes()[args.split]
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    main(nusc, scene_names, args.data_folder, args.split)
