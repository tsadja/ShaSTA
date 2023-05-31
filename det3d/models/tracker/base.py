import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
from det3d import torchie
from det3d.torchie.trainer import load_state_dict


class BaseTrack(nn.Module):
    """Base class for Track"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseTrack, self).__init__()
        self.fp16_enabled = False

    @property
    def with_reader(self):
        # Whether input data need to be processed by Input Feature Extractor
        return hasattr(self, "reader") and self.reader is not None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_pred(self):
        return hasattr(self, "pred") and self.pred is not None

    @property
    def with_matcher(self):
        return hasattr(self, "matcher") and self.matcher is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            checkpoint = torch.load(pretrained)
            load_state_dict(self, checkpoint['state_dict'], strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def forward_test(self, imgs, **kwargs):
        pass

    def forward(self, example, return_loss=True, **kwargs):
        pass
