import argparse
import logging
import json
import os

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import wandb

import collections
from collections import defaultdict

from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits

import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from det3d import torchie
from det3d.torchie.trainer import load_state_dict
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_simp_track
from det3d.torchie.apis import track_batch_processor, set_random_seed, get_root_logger
from det3d.torchie import Config
from pub_tracker import PubTracker as Tracker
from scipy.optimize import linear_sum_assignment

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a detector")
    parser.add_argument("--config", help="val config file path",
                        default='configs/nusc/car.py')
    parser.add_argument("--root", required=False, help="the dir with nuscenes data",
                        default='data/nuScenes')
    parser.add_argument("--work_dir", required=False, help="the dir to save logs and models",
                        default='work_dir/car_eval')
    parser.add_argument("--split", required=False, help="which data split to use",
                        default='val')
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from",
        default='models/car.pth'
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument('--local_rank', default=1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for confidence refinement")
    parser.add_argument("--beta", type=float, default=0.5, help="beta for confidence refinement")

    args = parser.parse_args()

    return args


def main(run=None):
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    logger = get_root_logger(cfg.log_level)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_simp_track(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = model.cuda()

    checkpoint = torch.load(args.checkpoint)
    load_state_dict(model, checkpoint)
    print("init weight from {}".format(args.checkpoint))

    ## Set up dataset and dataloader
    if args.split == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.split == 'test':
        print('Loading test data...')
        dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
    )

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    dead_tracker = {}

    model.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            (matched1, matched2, processed_batch) = track_batch_processor(model, data_batch, train_mode=False, local_rank=cfg.local_rank)
            
            token = processed_batch["metadata"][0]["token"]
            if token not in dead_tracker.keys():
                dead_tracker.update({token: {'dead_idx': [], 'keep_idx': []}})
            
            cls_det_boxes = processed_batch["cls_det_boxes"][0]
            prev_cls_det_boxes = processed_batch["prev_cls_det_boxes"][0]

            annos = []

            fn_annos = []

            num_prev_det_boxes = len(prev_cls_det_boxes)
            if num_prev_det_boxes > 0:
                keep_prev_dets = []
                prev_token = processed_batch["prev_metadata"][0]["token"]
                if prev_token not in dead_tracker.keys():
                    dead_tracker.update({prev_token: {'dead_idx': [], 'keep_idx': []}})
                matched_dets = torch.cat((matched1[0, :num_prev_det_boxes, :len(cls_det_boxes)], matched1[0,:num_prev_det_boxes,-2:]), dim=1)
                max_vals, max_idx = torch.max(matched_dets, dim=1)
                for n, (val, k) in enumerate(zip(max_vals, max_idx)):
                    val, k = val.item(), k.item()
                    # dead track
                    if val > 0.5 and k == matched_dets.shape[1]-2:
                        dead_tracker[prev_token]['dead_idx'].append(n)
                        continue
                    # FN
                    if val > 0.5 and k == matched_dets.shape[1]-1:
                        time_lag = processed_batch["prev_det_boxes"][0,0,9].item()
                        translation = [t+time_lag*v for t,v in zip(prev_cls_det_boxes[n]["translation"][:2], prev_cls_det_boxes[n]["velocity"])]
                        prev_cls_det_boxes[n]["translation"][:2] = translation
                        prev_cls_det_boxes[n]["FN"] = True
                        prev_cls_det_boxes[n]["token"] = token
                        prev_cls_det_boxes[n]["ref_detection_score"] = 1 - matched_dets[n,-2].item()
                        fn_annos.append(prev_cls_det_boxes[n])
                        continue
                    keep_prev_dets.append(n)
            
                matched_dets = torch.cat((matched2[0, keep_prev_dets, :len(cls_det_boxes)], matched2[0,-2:,:len(cls_det_boxes)]), dim=0)
            else:
                matched_dets = matched2[0,-2:,:len(cls_det_boxes)]

            keep_dets = []
            if len(cls_det_boxes) > 0:
                max_vals, max_idx = torch.max(matched_dets, dim=0)
                for k, (val, n) in enumerate(zip(max_vals, max_idx)):
                    val, n = val.item(), n.item()
                    if val > 0.7 and n == matched_dets.shape[0]-1:
                        continue
                    if val > 0.5 and n == matched_dets.shape[0]-2:
                        cls_det_boxes[k]['newborn'] = True
                    cls_det_boxes[k]['ref_detection_score'] = 1 - matched_dets[-1,k].item()
                    keep_dets.append(k)
                    annos.append(cls_det_boxes[k])
                dead_tracker[token]['keep_idx'] = keep_dets
            
            for curr_anno in fn_annos:
                annos.append(curr_anno)

            nusc_annos["results"].update({token: annos})
    
    for token in nusc_annos['results'].keys():
        dead_idx = dead_tracker[token]['dead_idx']
        keep_idx = dead_tracker[token]['keep_idx']
        for i in dead_idx:
            if i in keep_idx:
                local_i = keep_idx.index(i)
                nusc_annos['results'][token][local_i]['dead'] = True

    
    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    with open(os.path.join(args.work_dir, 'cp_' + args.split +'.json'), "w") as f:
        json.dump(nusc_annos, f)

    return

def save_first_frame(save_path):
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuScenes", verbose=False)
    scenes = splits.val

    frames = []
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
        frames.append(frame)

    del nusc
    
    with open(os.path.join(save_path, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def track(save_path, max_age, refine_confidence=False, alpha=0.5, beta=0.5):
    tracker = Tracker(max_age=max_age, hungarian=False, refine_confidence=refine_confidence, alpha=alpha, beta=beta)

    with open(os.path.join(save_path, 'cp_val.json'), 'rb') as f:
        predictions=json.load(f)['results']

    with open(os.path.join(save_path, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    # print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']
        # print(token)

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        time_lag = (frames[i]['timestamp'] - last_time_stamp) 
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token]

        outputs = tracker.step_centertrack(preds, time_lag)
        annos = []

        for item in outputs:
            if item['active'] == 0:
                continue
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
                "attribute_name": item['attribute_name'],
            }
            if refine_confidence:
                nusc_anno['tracking_score'] = item['ref_detection_score']

            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    
    end = time.time()

    second = (end-start) 

    speed=size / second
    # print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    if refine_confidence:
        file_name = 'tracking_result_ref_conf5_norm_alpha' + str(alpha) + '_beta' + str(beta)  +'.json'
    else:
        file_name = 'tracking_result.json'
        
    with open(os.path.join(save_path, file_name), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking(save_path, refine_confidence=False, alpha=0.5, beta=0.5):
    if refine_confidence:
        file_name = 'tracking_result_ref_conf5_norm_alpha' + str(alpha) + '_beta' + str(beta)  +'.json'
        output_path = save_path + '_conf_ref5_norm_alpha' + str(alpha) + '_beta' + str(beta) 
    else:
        file_name = 'tracking_result.json'
        output_path = save_path

    return eval(os.path.join(save_path, file_name),
                'v1.0-trainval',
                'val',
                output_path,
                "data/nuScenes"
                )

def eval(res_path, nusc_version, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs


    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=nusc_version,
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()

    return metrics_summary

if __name__ == "__main__":
    main()

    args = parse_args()
    cfg = Config.fromfile(args.config)

    save_first_frame(args.work_dir, args.split)
    track(args.work_dir, cfg.max_age)
    metrics_summary = eval_tracking(args.work_dir)

    track(args.work_dir, cfg.max_age, refine_confidence=True, alpha=args.alpha, beta=args.beta)
    ref_metrics_summary = eval_tracking(args.work_dir, refine_confidence=True, alpha=args.alpha, beta=args.beta)
