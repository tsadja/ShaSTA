""" input form of the data in each frame
"""
from .data_protos import BBox
import numpy as np, mot_3d.utils as utils


class FrameData:
    def __init__(self, dets, ego, gt_dets=None, time_stamp=None, pc=None, det_types=None, 
                gt_det_types=None, aux_info=None):
        self.dets = dets         # detections for each frame
        self.ego = ego           # ego matrix information
        self.gt_dets = gt_dets
        self.pc = pc
        self.det_types = det_types
        self.gt_det_types = gt_det_types
        self.time_stamp = time_stamp
        self.aux_info = aux_info

        for i, det in enumerate(self.dets):
            self.dets[i] = BBox.array2bbox(det)
        
        if self.gt_dets is not None:
            for i, gt_det in enumerate(self.gt_dets):
                self.gt_dets[i] = BBox.array2bbox(gt_det)
        
        # if not aux_info['is_key_frame']:
        #     self.dets = [d for d in self.dets if d.s >= 0.5]