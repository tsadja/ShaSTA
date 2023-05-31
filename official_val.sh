python tools/nusc_shasta/eval.py --config configs/nusc/bicycle.py \
--work_dir work_dir/final_val_results/bicycle --split 'val' --root data/nuScenes \
--checkpoint models/bicycle.pth
python tools/nusc_shasta/eval.py --config configs/nusc/bus.py \
--work_dir work_dir/final_val_results/bus --split 'val' --root data/nuScenes \
--checkpoint models/bus.pth
python tools/nusc_shasta/eval.py --config configs/nusc/car.py \
--work_dir work_dir/final_val_results/car --split 'val' --root data/nuScenes \
--checkpoint models/car.pth
python tools/nusc_shasta/eval.py --config configs/nusc/motorcycle.py \
--work_dir work_dir/final_val_results/motorcycle --split 'val' --root data/nuScenes \
--checkpoint models/motorcycle.pth
python tools/nusc_shasta/eval.py --config configs/nusc/ped.py \
--work_dir work_dir/final_val_results/pedestrian --split 'val' --root data/nuScenes \
--checkpoint models/pedestrian.pth
python tools/nusc_shasta/eval.py --config configs/nusc/trailer.py \
--work_dir work_dir/final_val_results/trailer --split 'val' --root data/nuScenes \
--checkpoint models/trailer.pth
python tools/nusc_shasta/eval.py --config configs/nusc/truck.py \
--work_dir work_dir/final_val_results/truck --split 'val' --root data/nuScenes \
--checkpoint models/truck.pth
python tools/nusc_shasta/merge_results.py --work_dir work_dir/final_val_results --checkpoint_name merged_cp_val.json
python tools/nusc_shasta/pub_test.py --work_dir work_dir/final_val_results --root data/nuScenes --checkpoint work_dir/final_val_results/merged_cp_val.json \
--split 'val' --version 'v1.0-trainval'
