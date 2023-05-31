import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

pretrained_bev_map = "data/nusc_preprocessed/bev_map.pth"
freeze_bev = True
max_age = 4

max_objects = 90
num_feats = 3
det_type = ["pedestrian"]
max_objects *= len(det_type)
iou3d_nms_thresh=0.3
fp_ratio = 1/3
dead_trk_ratio = 1/3

#fp_elim = 0.6
fp_elim = 0.7

alpha = 0.5
beta = 0.5

refine_confidence = False

# model settings
model = dict(
    type="Shasta",
    pretrained=pretrained_bev_map,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bev_extractor=dict(
        type="BEVFeatureExtractor",
        pc_start=[-54, -54],
        voxel_size=[0.075, 0.075],
        out_stride=8
    ),
    max_obj=max_objects,
    num_feats=num_feats,
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075]
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuScenes"

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=None,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/nusc_preprocessed/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nusc_preprocessed/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = "data/nusc_preprocessed/infos_test_10sweeps_withvelo.pkl"

train_det_path = "data/nusc_preprocessed/train_2hz/detections/cp/sensor_individual_frames"
val_det_path = "data/nusc_preprocessed/val_2hz/detections/cp/sensor_individual_frames"
test_det_path = "data/nusc_preprocessed/test_2hz/detections/cp/sensor_individual_frames"

train_labels_path = "data/nusc_preprocessed/train_2hz/gt_shasta/cp/individual_frames"

train_frames_path = "data/nusc_preprocessed/train_frame_info.json"
train_cls_path = "data/nusc_preprocessed/train_2hz/detections/cp/cls_individual_frames"

val_frames_path = "data/nusc_preprocessed/val_frame_info.json"
val_cls_path = "data/nusc_preprocessed/val_2hz/detections/cp/cls_individual_frames"

test_frames_path = "data/nusc_preprocessed/test_frame_info.json"
test_cls_path = "data/nusc_preprocessed/test_2hz/detections/cp/cls_individual_frames"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=48,  
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        det_path = train_det_path,
        labels_path = train_labels_path,
        frame_info_path = train_frames_path,
        cls_info_path = train_cls_path,
        ann_file=train_anno,
        det_type=det_type,
        max_objects=max_objects,
        fp_ratio=fp_ratio,
        dead_trk_ratio=dead_trk_ratio,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        det_path = val_det_path,
        frame_info_path = val_frames_path,
        cls_info_path = val_cls_path,
        test_mode=True,
        ann_file=val_anno,
        det_type=det_type,
        max_objects=max_objects,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        det_path = test_det_path,
        frame_info_path = test_frames_path,
        cls_info_path = test_cls_path,
        ann_file=test_anno, 
        test_mode=True,
        det_type=det_type,
        max_objects=max_objects,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
learning_rate = 2.5e-4
weight_decay = 1e-2

use_scheduler = False
max_lr = 1e-3
pct_start=0.4
anneal_strategy='cos'
div_factor = 10.0
base_momentum=0.85
max_momentum=0.95

optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 4
device_ids = range(4)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]
