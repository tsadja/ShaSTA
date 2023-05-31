python tools/nusc_shasta/eval.py --config configs/nusc/bicycle.py \
--work_dir work_dir/final_test_results/bicycle --split 'test' --root data/nuScenes \
--checkpoint models/bicycle.pth
python tools/nusc_shasta/eval.py --config configs/nusc/bus.py \
--work_dir work_dir/final_test_results/bus --split 'test' --root data/nuScenes \
--checkpoint models/bus.pth
python tools/nusc_shasta/eval.py --config configs/nusc/car.py \
--work_dir work_dir/final_test_results/car --split 'test' --root data/nuScenes \
--checkpoint models/car.pth
python tools/nusc_shasta/eval.py --config configs/nusc/motorcycle.py \
--work_dir work_dir/final_test_results/motorcycle --split 'test' --root data/nuScenes \
--checkpoint models/motorcycle.pth
python tools/nusc_shasta/eval.py --config configs/nusc/ped.py \
--work_dir work_dir/final_test_results/pedestrian --split 'test' --root data/nuScenes \
--checkpoint models/pedestrian.pth
python tools/nusc_shasta/eval.py --config configs/nusc/trailer.py \
--work_dir work_dir/final_test_results/trailer --split 'test' --root data/nuScenes \
--checkpoint models/trailer.pth
python tools/nusc_shasta/eval.py --config configs/nusc/truck.py \
--work_dir work_dir/final_test_results/truck --split 'test' --root data/nuScenes \
--checkpoint models/truck.pth
python tools/nusc_shasta/merge_results.py --work_dir work_dir/final_test_results --checkpoint_name merged_cp_test.json
python tools/nusc_shasta/pub_test.py --work_dir work_dir/final_test_results --root data/nuScenes --checkpoint work_dir/final_test_results/merged_cp_test.json \
--split 'test' --version 'v1.0-test'
