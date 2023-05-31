from det3d.utils import Registry

READERS = Registry("reader")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
BEV = Registry("bev")
PRED = Registry("pred")
TRACK = Registry("track")
MATCHER = Registry("matcher")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")
