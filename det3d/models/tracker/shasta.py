from ..registry import TRACK
from .base import BaseTrack
from det3d.torchie.trainer import load_state_dict
from .. import builder
import torch
import torch.nn as nn
from det3d.core import box_torch_ops

@TRACK.register_module
class Shasta(BaseTrack):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bev_extractor,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        max_obj=100,
        num_feats=7,
        in_channels=512,
        share_conv_channel=64,
        num_point=5,
    ):
        super(Shasta, self).__init__()
        
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.bev_extractor = builder.build_second_stage_module(bev_extractor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        self.num_feats = num_feats
        self.max_obj = max_obj
        self.num_point = num_point

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.aug_shape = nn.ModuleList()
        self.aug_shape_input = max_obj*share_conv_channel*num_point
        self.aug_shape_output = share_conv_channel*num_point
        for i in range(4):
            self.aug_shape.append(nn.Sequential(
                nn.Linear(in_features=self.aug_shape_input, out_features=self.aug_shape_input//64),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.aug_shape_input//64, out_features=self.aug_shape_output),
            ))

        self.fuse_shape = nn.Sequential(
            nn.Linear(in_features=2*self.aug_shape_output, out_features=self.aug_shape_output//8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.aug_shape_output//8, out_features=self.aug_shape_output//16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.aug_shape_output//16, out_features=self.aug_shape_output//32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.aug_shape_output//32, out_features=1),
        )

        self.aug_input = max_obj*7
        self.aug_dets = nn.ModuleList()
        for i in range(4):
            self.aug_dets.append(nn.Sequential(
                nn.Linear(in_features=self.aug_input, out_features=self.aug_input//32),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.aug_input//32, out_features=7),
            ))

        self.fuse_det = nn.Sequential(
            nn.Linear(in_features=self.num_feats*2, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8, out_features=1),
        )

        self.res_coeff = nn.Sequential(
            nn.Linear(in_features=self.num_feats*2 + self.aug_shape_output*2, out_features=32 + self.aug_shape_output//8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32 + self.aug_shape_output//8, out_features=8 + self.aug_shape_output//32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8 + self.aug_shape_output//32, out_features=3),
        )

        self.aff = nn.Sequential(
            nn.Linear(in_features=max_obj+2, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=max_obj+2),
        )

        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            checkpoint = torch.load(pretrained)
            load_state_dict(self, checkpoint, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
    
    def get_box_center(self, boxes):
        centers = [] 
        for box in boxes: 
            if self.num_point==1:
                centers.append(box[:, :3])
            elif self.num_point==4:        
                center2d = box[:, :2]
                height = box[:, 2:3]
                dim2d = box[:, 3:5]
                rotation_y = box[:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([front_middle, back_middle, left_middle, \
                    right_middle], dim=0) # (max_obj*num_points) X 3

                centers.append(points)
            elif self.num_point==5:
                center2d = box[:, :2]
                height = box[:, 2:3]
                dim2d = box[:, 3:5]
                rotation_y = box[:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box[:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0) # (max_obj*num_points) X 3

                centers.append(points)

        return centers


    def extract_feat(self, data):
        if 'voxels' not in data or 'prev_voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            prev_data = dict(
                features=data['prev_voxels'],
                num_voxels=data["prev_num_points"],
                coors=data["prev_coordinates"],
                batch_size=len(data['prev_points']),
                input_shape=data["prev_shape"][0],
            )

            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )

            input_features = self.reader(data["features"], data['num_voxels'])
            prev_input_features = self.reader(prev_data["features"], prev_data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        prev_x, prev_voxel_feature = self.backbone(
                prev_input_features, prev_data["coors"], prev_data["batch_size"], prev_data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)
            prev_x = self.neck(prev_x)


        return x, voxel_feature, prev_x, prev_voxel_feature


    def forward(self, example, train_mode=True, **kwargs):
        # Load inputs
        prev_det_boxes = example["prev_det_boxes"][:,:,:7]
        det_boxes = example["det_boxes"][:,:,:7]
        vel_det_boxes = example["det_boxes"][:,:,7:9]
        time_diffs = example["det_boxes"][:,:,9].unsqueeze(-1)

        # Get BEV maps
        bev_map, _, prev_bev_map, _ = self.extract_feat(example)

        bev_map = self.shared_conv(bev_map)
        example['bev_feature'] = bev_map.permute(0, 2, 3, 1).contiguous()

        prev_bev_map = self.shared_conv(prev_bev_map)
        prev_example = {}
        prev_example['bev_feature'] = prev_bev_map.permute(0, 2, 3, 1).contiguous()

        # Get geometric features for current frame detections --> ALL GEOMETRIC FEATURES ARE NON-NEGATIVE!!
        centers_vehicle_frame = self.get_box_center(det_boxes) 
        feature = self.bev_extractor(example, centers_vehicle_frame, self.num_point)
        feature = torch.stack(feature) # B x MAX_OBJ X 320 --> 320/64 = 5 (num_point)

        # Get geometric features for previous frame detections
        prev_centers_vehicle_frame = self.get_box_center(prev_det_boxes) 
        prev_feature = self.bev_extractor(prev_example, prev_centers_vehicle_frame, self.num_point)
        prev_feature = torch.stack(prev_feature) # B x MAX_OBJ X 320 --> 320/64 = 5 (num_point)

        # Add newborn, fp, and dead track anchor geometric representations
        newborn_geom = torch.abs(self.aug_shape[0].forward(feature.view(feature.shape[0], -1)).reshape(feature.shape[0], 1, -1))
        fp_geom = torch.abs(self.aug_shape[1].forward(feature.view(feature.shape[0], -1)).reshape(feature.shape[0], 1, -1))
        dead_trk_geom = torch.abs(self.aug_shape[2].forward(prev_feature.view(prev_feature.shape[0], -1)).reshape(prev_feature.shape[0], 1, -1))
        fn_geom = torch.abs(self.aug_shape[3].forward(prev_feature.view(prev_feature.shape[0], -1)).reshape(prev_feature.shape[0], 1, -1))

        feature = torch.cat((feature, dead_trk_geom, fn_geom), dim=1)
        prev_feature = torch.cat((prev_feature, newborn_geom, fp_geom), dim=1)

        # Get shape residual
        feature = feature.unsqueeze(1)
        prev_feature = prev_feature.unsqueeze(2)

        B, D, T, F = prev_feature.shape[0], feature.shape[2], prev_feature.shape[1], self.aug_shape_output

        prev_feature = prev_feature.expand(B, T, D, F) # shape: (B, T, D, 320)
        feature = feature.expand(B, T, D, F) # shape: (B, T, D, 320)
            

        # Add newborn, fp, and dead track anchor bbox representations
        newborn = self.aug_dets[0].forward(det_boxes.reshape(det_boxes.shape[0], -1)).reshape(det_boxes.shape[0], 1, -1)
        self.newborn = torch.cat((newborn[:,:,:3], torch.abs(newborn[:,:,3:6]), newborn[:,:,6:]), dim=-1)
        fp =self.aug_dets[1].forward(det_boxes.reshape(det_boxes.shape[0], -1)).reshape(det_boxes.shape[0], 1, -1)
        self.fp = torch.cat((fp[:,:,:3], torch.abs(fp[:,:,3:6]), fp[:,:,6:]), dim=-1)
        dead_trk = self.aug_dets[2].forward(prev_det_boxes.reshape(prev_det_boxes.shape[0], -1)).reshape(prev_det_boxes.shape[0], 1, -1)
        self.dead_trk = torch.cat((dead_trk[:,:,:3], torch.abs(dead_trk[:,:,3:6]), dead_trk[:,:,6:]), dim=-1)
        fn = self.aug_dets[3].forward(prev_det_boxes.reshape(prev_det_boxes.shape[0], -1)).reshape(prev_det_boxes.shape[0], 1, -1)
        self.fn = torch.cat((fn[:,:,:3], torch.abs(fn[:,:,3:6]), fn[:,:,6:]), dim=-1)

        # Backproject curr detections using velocity
        det_boxes[:,:,:2] = det_boxes[:,:,:2] - vel_det_boxes * time_diffs
        
        # Augment with newborn, fp, and dead tracks
        prev_det_boxes = torch.cat((prev_det_boxes, self.newborn, self.fp), dim=1) # shape: (B, N+2, 3)
        det_boxes = torch.cat((det_boxes, self.dead_trk, self.fn), dim=1) # shape: (B, K+2, 3)

        # Get hand-designed residuals
        eps = 1e-10
        residual_dist = ((prev_det_boxes[:,:,:self.num_feats].unsqueeze(2)- det_boxes[:,:,:self.num_feats].unsqueeze(1))**2).sum(dim=-1)
        residual_dist = nn.functional.normalize(residual_dist)
        residual_dim = torch.abs(torch.log(prev_det_boxes[:,:,3:6].unsqueeze(2)+eps) - torch.log(det_boxes[:,:,3:6].unsqueeze(1)+eps)).sum(dim=-1)
        residual_dist += residual_dim
        residual_rot = torch.sqrt((torch.cos(prev_det_boxes[:,:,6].unsqueeze(2)) - torch.cos(det_boxes[:,:,6].unsqueeze(1)))**2 + (torch.sin(prev_det_boxes[:,:,6].unsqueeze(2)) - torch.sin(det_boxes[:,:,6].unsqueeze(1)))**2).squeeze(-1)
        residual_dist += residual_rot

        # Shape residual
        fused_shape = torch.cat([prev_feature, feature], dim=3) # shape: (B, T, D, 640)
        fused_shape = fused_shape.view(B, T*D, F*2) # shape: (B, T*D, 640)
        residual_shape = self.fuse_shape(fused_shape) # shape: (B, T*D, 1)
        residual_shape = residual_shape.view(B, T, D, -1) # shape: (B, T, D, 1)
        residual_shape = residual_shape[:,:,:,0] # shape: (B, T, D)

        # Create desired input for voxelnet residual
        det_boxes = det_boxes[:,:,:self.num_feats]
        prev_det_boxes = prev_det_boxes[:,:,:self.num_feats]
        prev_det_boxes = prev_det_boxes.clone().unsqueeze(2) # shape: (B, N+2, 1, 3)
        det_boxes = det_boxes.clone().unsqueeze(1)

        B, D, T, F = prev_det_boxes.shape[0], det_boxes.shape[2], prev_det_boxes.shape[1], self.num_feats

        prev_det_boxes = prev_det_boxes.expand(B, T, D, F) # shape: (B, T, D, 3)
        det_boxes = det_boxes.expand(B, T, D, F) # shape: (B, T, D, 3)

        fused_boxes = torch.cat([prev_det_boxes, det_boxes], dim=3) # shape: (B, T, D, 6)
        fused_boxes = fused_boxes.view(B, T*D, F*2) # shape: (B, T*D, 6)
        residual_fused = self.fuse_det(fused_boxes) # shape: (B, T*D, 1)
        residual_fused = residual_fused.view(B, T, D, -1) # shape: (B, T, D, 1)
        residual_fused = residual_fused[:,:,:,0] # shape: (B, T, D)

        # Find learned weights for summing individual residuals
        fused_prev = torch.cat([prev_feature, prev_det_boxes], dim=-1) # shape: (B, T, D, 320 + 3)
        fused_curr = torch.cat([feature, det_boxes], dim=-1) # shape: (B, T, D, 320 + 3)
        fused_all = torch.cat([fused_prev, fused_curr], dim=-1) # shape: (B, T, D, 2*(320 + 3))
        fused_all = fused_all.view(B, T*D, -1) # shape: (B, T*D, 2*(320 + 3))
        residual_coeff= self.res_coeff(fused_all) # shape: (B, T*D, 3)
        residual_coeff = residual_coeff.view(B, T, D, -1) # shape: (B, T, D, 3)
        alpha, beta, omega = residual_coeff[:,:,:,0], residual_coeff[:,:,:,1], residual_coeff[:,:,:,2] 

        # Get overall residual (weighted sum)
        residual = torch.mul(alpha, residual_fused) + torch.mul(beta, residual_dist) + torch.mul(omega, residual_shape)


        # Find affinity
        matched = self.aff(residual)
        matched1 = self.softmax1(matched[:,:-2,:])
        matched2 = self.softmax2(matched[:,:,:-2])

        return matched1, matched2, example

