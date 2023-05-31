import argparse
import logging
import json
import os
import sys
import subprocess
import pdb
import time

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel
except:
    print("No APEX!")

import torch.distributed as dist
import subprocess

import wandb

import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel
from det3d import torchie
from det3d.torchie.trainer import load_state_dict
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_track, build_simp_track
from det3d.torchie.apis import track_batch_processor, set_random_seed, get_root_logger
from det3d.torchie import Config
from tools.nusc_shasta.validate import validate
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path",
                        default='configs/nusc/car.py')
    parser.add_argument("--work_dir", required=False, help="the dir to save logs and models",
                        default='work_dirs/car_train')
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--project_name", default=None, help="project name for wandb")
    parser.add_argument("--group_name", default=None, help="group name for wandb project")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="slurm",
        help="job launcher",
    )
    parser.add_argument('--local_rank', default=0, type=int, 
                        help='local rank for distributed training')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(run):
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    logger = get_root_logger(cfg.log_level)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        if args.launcher == 'pytorch':
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            cfg.local_rank = args.local_rank
        elif args.launcher == "slurm":
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            cfg.gpus = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                # 29500 is torch.distributed default port
                os.environ["MASTER_PORT"] = "29501"
            # use MASTER_ADDR in the environment variable if it already exists
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            cfg.local_rank = int(os.environ["LOCAL_RANK"])

        cfg.gpus = dist.get_world_size()
    else:
        cfg.local_rank = args.local_rank

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    if not os.path.exists(args.work_dir) and cfg.local_rank == 1:
        os.makedirs(args.work_dir)

    if cfg.local_rank == 1:
        logger.info("Distributed training: {}".format(distributed))
        logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_simp_track(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

    ## Set up loss and optimizer
    epochs = cfg.total_epochs
    learning_rate = cfg.learning_rate
    weight_decay = cfg.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

    # put model on gpus
    if cfg.local_rank == 0:
        print('is distributed = ', distributed)
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(model)
    else:
        model = model.cuda()

    ## Set up dataset and dataloader
    dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
    )

    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(data_loader), epochs=epochs, max_lr=cfg.max_lr, 
                                                pct_start=cfg.pct_start, anneal_strategy=cfg.anneal_strategy, div_factor=cfg.div_factor,
                                                base_momentum=cfg.base_momentum, max_momentum=cfg.max_momentum)

    if cfg.local_rank == 1 and run is not None:
        run.watch(model)

    print(len(data_loader))

    dist_criterion = nn.L1Loss()
    # dist_criterion = nn.MSELoss()

    if cfg.freeze_bev:
        print('freeze CP bev weights')
        for i, child in enumerate(model.children()):
            if i > 3:
                break
            elif i == 1 or i == 2:
                for param in child.parameters():
                    param.requires_grad = False

    model.train()
    with torch.enable_grad():
        for epoch in tqdm(range(epochs)):
            batch_loss = []
            for i, data_batch in tqdm(enumerate(data_loader), leave=False):
                matched1, matched2, gt = track_batch_processor(model, data_batch, local_rank=cfg.local_rank)

                # get GT info
                gt1 = gt[:,:-2,:]
                gt2 = gt[:,:,:-2]

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # calculate loss
                eps=1e-10
                loss_f = torch.mul(gt1, -torch.log(matched1+eps)).sum() / gt1.sum() if gt1.sum() > 0 else torch.mul(gt1, -torch.log(matched1+eps)).sum()
                loss_b = torch.mul(gt2, -torch.log(matched2+eps)).sum() / gt2.sum() if gt2.sum() > 0 else torch.mul(gt2, -torch.log(matched2+eps)).sum()
                loss = (loss_f + loss_b)/2.

                # backprop and update
                loss.backward()
                optimizer.step()

                if cfg.use_scheduler:
                    scheduler.step()

                curr_loss = loss.item()
                batch_loss.append(curr_loss)

                if run is not None:
                    run.log({"batch_loss": curr_loss})
            
            if run is not None:
                run.log({"epoch": epoch, "loss": np.mean(batch_loss)})
            if cfg.local_rank == 1:
                model_path = os.path.join(args.work_dir, 'epoch'+ str(epoch+1)+'.pth')
                if distributed:
                    torch.save(model.module.state_dict(), model_path)
                    save_path = os.path.join(args.work_dir, 'validate_epoch' + str(epoch+1))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    validate(model_path, cfg.model, val_dataset, cfg.data.workers_per_gpu, 
                            save_path, cfg.local_rank, run, cfg.det_type, cfg.train_cfg, cfg.test_cfg, cfg.max_age, cfg.alpha, cfg.beta)
                else:
                    torch.save(model.state_dict(), model_path)
    
    if run is not None:
        run.finish()

    return

if __name__ == "__main__":
    ## Set up wandb
    args = parse_args()
    if args.project_name is None:
        run = None
    else:
        run = wandb.init(
            project=args.project_name,
            group=args.group_name,  # all runs for the experiment in one group
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )
    main(run)
