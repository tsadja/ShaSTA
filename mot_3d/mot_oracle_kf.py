import sys
from copy import deepcopy
import numpy as np, mot_3d.tracklet as tracklet, mot_3d.utils as utils
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .association import associate_dets_to_tracks
from . import visualization as visualization
from mot_3d import redundancy
import pdb, os
import matplotlib.pyplot as plt
from collections import OrderedDict
sys.path.append('./../')
from preprocessing.gt_association.associate import associate


def frame_visualization(bboxes, ids, save_dir, gt_bboxes=None, gt_ids=None, pc=None, dets=None, name=''):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    if pc is not None:
        visualizer.handler_pc(pc)

    if gt_bboxes is not None:
        for _, (bbox, id) in enumerate(zip(gt_bboxes, ids)):
            visualizer.handler_box(bbox, message='%s'%(id), color='black', label='GT')
    if dets is not None:
        dets = [d for d in dets if d.s >= 0.01]
        for det in dets:
            visualizer.handler_box(det, message='%.2f' % det.s, color='gray', linestyle='dashed')
    for _, (bbox, id) in enumerate(zip(bboxes, ids)):
        visualizer.handler_box(bbox, message='%s'%(id), color='red', label='TP KF Pred')
    # remove duplicates in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # save visualization
    save_path = os.path.join(save_dir, '{:}.png'.format(name))
    visualizer.save(save_path)
    visualizer.close()


class MOTModel:
    def __init__(self, configs):
        self.trackers = list()         # tracker for each single tracklet
        self.frame_count = 0           # record for the frames
        self.count = 0                 # record the obj number to assign ids
        self.time_stamp = None         # the previous time stamp
        self.redundancy = RedundancyModule(configs) # module for no detection cases

        #non_key_redundancy_config = deepcopy(configs)
        #non_key_redundancy_config['redundancy'] = {
        #    'mode': 'mm',
        #    'det_score_threshold': {'giou': 0.1, 'iou': 0.1, 'euler': 0.1},
        #    'det_dist_threshold': {'giou': -0.5, 'iou': 0.1, 'euler': 4}
        #}
        #self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

        self.configs = configs
        self.visualize = configs['running']['visualize']
        self.scene_name = configs['scene_name']
        self.match_type = configs['running']['match_type']
        self.score_threshold = configs['running']['score_threshold']
        self.asso = configs['running']['asso']
        self.asso_thres = configs['running']['asso_thres'][self.asso]
        self.motion_model = configs['running']['motion_model']

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']

    @property
    def has_velo(self):
        return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')
    
    def frame_mot(self, input_data: FrameData, obj_type):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp

        if not input_data.aux_info['is_key_frame']:
            result = self.non_key_frame_mot(input_data, obj_type)
            return result
    
        if self.motion_model == 'velo':
            matched, unmatched_dets, unmatched_trks = self.back_step_det(input_data)
        elif 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks, tp_ind_pairs = self.forward_step_trk(input_data, obj_type)
        elif 'ma' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data, obj_type)
        
        time_lag = input_data.time_stamp - self.time_stamp
        for t, trk in enumerate(self.trackers):
            gt_bbox = None
            # if t in tp_ind_pairs.keys():
            #     gt_idx = tp_ind_pairs[t]
            #     gt_bbox = input_data.gt_dets[gt_idx]

            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info, gt_bbox=gt_bbox)
            else:
                result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag) # does not actually use input_data for default mode we use
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info, gt_bbox=gt_bbox)

        for index in unmatched_dets:
            if self.has_velo:
                aux_info = {
                    'velo': list(input_data.aux_info['velos'][index]), 
                    'is_key_frame': input_data.aux_info['is_key_frame']}
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}

            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1
        
        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))
        
        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result
    
    def forward_step_trk(self, input_data: FrameData, obj_type):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        dets = [dets[i] for i in det_indexes]

        # prediction
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))

        # get TP kalman filter predictions
        gt_boxes, gt_types = list(), list()
        pred_boxes, pred_types = list(), list()
        tp_ind_pairs, fp_inds, fn_inds = dict(), list(), list()
        if self.trackers:
            gt_boxes = input_data.gt_dets
            gt_types = input_data.gt_det_types

            for trk_pred in trk_preds:
                pred_boxes.append(trk_pred) # trk_pred same as trk.motion_model.history[-1]
            pred_types = [obj_type]*len(pred_boxes)
        else:
            pred_boxes = input_data.dets
            pred_types = [obj_type]*len(pred_boxes)

        _, _, _, _, _, _, _, tp_ind_pairs, fp_inds, fn_inds = associate(gt_boxes, gt_types, pred_boxes, pred_types, 
                                                            threshold=2.0, distance_type="l2")

        # visualize the TP pairs between GT boxes and kalman filter predictions
        if self.visualize:
            if trk_preds:
                save_dir = os.path.join('/juno/u/tsadja/SimpleTrack/nu_mot_results/debug_base1_vis/imgs', obj_type)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_dir = os.path.join(save_dir, self.scene_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pred_bboxes, gt_bboxes, ids = list(), list(), list()
                for idx, t in enumerate(tp_ind_pairs.keys()):
                    gt_idx = tp_ind_pairs[t]
                    pred_bboxes.append(trk_preds[t])
                    gt_bboxes.append(gt_boxes[gt_idx])
                    ids.append(idx)
                
                frame_visualization(pred_bboxes, ids, save_dir, gt_bboxes=gt_bboxes, 
                                    gt_ids=ids, name='{:}'.format(self.frame_count))


        # replace TP kalman filter predictions with GT for data association
        for t, trk_pred in enumerate(trk_preds):
            if t in tp_ind_pairs.keys():
                gt_idx = tp_ind_pairs[t]
                gt_boxes[gt_idx].s = trk_pred.s
                trk_preds[t] = gt_boxes[gt_idx]
                # ADDED
                # print(self.trackers[t].motion_model.kf.x_prior.shape)
                # print(np.expand_dims(BBox.bbox2array(gt_boxes[gt_idx]), axis=1).shape)
        

        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]

        # association
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        
        return matched, unmatched_dets, unmatched_trks, tp_ind_pairs
    
    def non_key_forward_step_trk(self, input_data: FrameData):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks, tp_ind_pairs

    def back_step_det(self, input_data: FrameData):
        # extract the detection information
        dets, velos = input_data.dets, input_data.aux_info['velos']
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        dets, velos = [dets[i] for i in det_indexes], [velos[i] for i in det_indexes]

        # back-step the detection and association
        cur_time_stamp = input_data.time_stamp
        time_lag = cur_time_stamp - self.time_stamp
        
        # back-step prediction
        det_preds = list()
        for det, velo in zip(dets, velos):
            det_preds.append(utils.back_step_det(det, velo, time_lag))
        trk_states = [trk.get_state() for trk in self.trackers]

        # make every tracklet predict to hold the frame place
        for trk in self.trackers:
            trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame'])
        
        # for m-dis
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]
        
        # association
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(det_preds, trk_states, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks 
    
    def non_key_frame_mot(self, input_data: FrameData, obj_type):
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
        
        if self.motion_model == 'velo':
            matched, unmatched_dets, unmatched_trks = self.back_step_det(input_data)
        elif 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
        elif 'ma' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data, obj_type)
        time_lag = input_data.time_stamp - self.time_stamp

        redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t], 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result