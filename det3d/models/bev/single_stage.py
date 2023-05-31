import torch.nn as nn

from .. import builder
from ..registry import BEV
from .base import BaseBEV
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint


@BEV.register_module
class SingleStageBEV(BaseBEV):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(SingleStageBEV, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self