from ..registry import BEV
from .single_stage import SingleStageBEV
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@BEV.register_module
class BEVMap(SingleStageBEV):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(BEVMap, self).__init__(
            reader, backbone, neck, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        if 'voxels' not in data:
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
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        bev_feature = x
        return bev_feature