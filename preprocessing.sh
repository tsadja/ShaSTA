#!/bin/sh
python preprocessing/nuscenes_data/token_info.py --split train --raw_data_folder data/nuScenes
python preprocessing/nuscenes_data/ego_pose.py --split train --raw_data_folder data/nuScenes
python preprocessing/nuscenes_data/gt_info.py --split train --raw_data_folder data/nuScenes
python preprocessing/nuscenes_data/detection.py --split train --det_name cp --file_name train.json
python preprocessing/get_det_info.py --data_folder data/nusc_preprocessed --det_name cp --file_name train.json --detection_folder data/nusc_preprocessed/train_2hz/detections --split train
python preprocessing/get_det_sensor_info.py --split train --raw_data_folder data/nuScenes --data_folder data/nusc_preprocessed --detection_folder data/nusc_preprocessed/train_2hz/detections --det_name cp --file_name train.json
python preprocessing/get_frame_info.py --split train --raw_data_folder data/nuScenes --data_folder data/nusc_preprocessed
python preprocessing/make_gt_shasta.py --name gt_shasta --det_name cp --raw_data_folder data/nuScenes --data_folder data/nusc_preprocessed/train_2hz --gt_folder data/nusc_preprocessed/train_2hz/gt_info --det_folder data/nusc_preprocessed/train_2hz/detections --split train

python preprocessing/nuscenes_data/token_info.py --split val --raw_data_folder data/nuScenes
python preprocessing/nuscenes_data/ego_pose.py --split val --raw_data_folder data/nuScenes
python preprocessing/nuscenes_data/detection.py --split val --det_name cp --file_name val.json
python preprocessing/get_det_info.py --data_folder data/nusc_preprocessed --det_name cp --file_name val.json --detection_folder data/nusc_preprocessed/val_2hz/detections --split val
python preprocessing/get_det_sensor_info.py --split val --raw_data_folder data/nuScenes --data_folder data/nusc_preprocessed --detection_folder data/nusc_preprocessed/val_2hz/detections --det_name cp --file_name val.json
python preprocessing/get_frame_info.py --split val --raw_data_folder data/nuScenes --data_folder data/nusc_preprocessed

python preprocessing/nuscenes_data/token_info.py --split test --test
python preprocessing/nuscenes_data/ego_pose.py --split test --test
python preprocessing/nuscenes_data/detection.py --split test --test --det_name cp --file_name test.json
python preprocessing/get_det_info.py --split test --test
python preprocessing/get_det_sensor_info.py --split test --test
python preprocessing/get_frame_info.py --split test --test
