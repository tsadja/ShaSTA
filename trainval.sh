#!/bin/sh
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=8 ./tools/nusc_shasta/train.py --launcher pytorch --seed 1 \
--config configs/nusc/car.py --work_dir work_dir/car --project_name shasta --group_name car
