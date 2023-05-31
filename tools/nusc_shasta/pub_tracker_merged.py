import numpy as np
import copy
from track_utils import greedy_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import copy 
import importlib
import sys 
import difflib

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':2,
  'truck':2,
  'bus':4,
  'trailer':2,
  'pedestrian':0.75,
  'motorcycle':2,
  'bicycle':1.5,
}

TRK_REF = {
        'bicycle': {'alpha':0.5, 'beta':0.4, 'ref': True},
        'bus': {'alpha':0.5, 'beta':0.7, 'ref': True},
        'car': {'alpha':0.5, 'beta':0.5, 'ref': True},
        'motorcycle': {'alpha':0.5, 'beta':0.5, 'ref': True},
        'pedestrian': {'alpha':0.5, 'beta':0.5, 'ref': True},
        'trailer':{'alpha':0.5, 'beta':0.4, 'ref': True},
        'truck':{'alpha':0.5, 'beta':0.5, 'ref': True},
}
'''
TRK_REF = {
        'bicycle': {'alpha':0.5, 'beta':0.4, 'ref': False},
        'bus': {'alpha':0.5, 'beta':0.7, 'ref': False},
        'car': {'alpha':0.5, 'beta':0.5, 'ref': False},
        'motorcycle': {'alpha':0.5, 'beta':0.5, 'ref': False},
        'pedestrian': {'alpha':0.5, 'beta':0.5, 'ref': False},
        'trailer':{'alpha':0.5, 'beta':0.4, 'ref': False},
        'truck':{'alpha':0.5, 'beta':0.5, 'ref': False},
}

'''


class PubTrackerMerged(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian
    self.max_age = max_age

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    print(self.NUSCENE_CLS_VELOCITY_ERROR)

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
        self.tracks = []
        return []
    ret = []
    for nusc_name in NUSCENES_TRACKING_NAMES:
        # print(nusc_name)
        # alpha, beta, refine_score = TRK_REF[nusc_name]['alpha'], TRK_REF[nusc_name]['beta'], TRK_REF[nusc_name]['ref']
        temp = []
        curr_ret = []
        for det in results:
            # print(det['detection_name'])
            if det['detection_name'] != nusc_name:
                # print(det['detection_name'], nusc_name)
                continue 

            det['ct'] = np.array(det['translation'][:2])
            det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
            det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
            temp.append(det)

        curr_results = temp

        N = len(curr_results)
        curr_tracks = [track for track in self.tracks if track['detection_name'] == nusc_name]
        M = len(curr_tracks)

        # print(N, M)
        if len(temp) == 0:
            continue

        # N X 2 
        if 'tracking' in curr_results[0]:
            dets = np.array(
            [ det['ct'] + det['tracking'].astype(np.float32)
            for det in curr_results], np.float32)
        else:
            dets = np.array(
                [det['ct'] for det in curr_results], np.float32) 

        item_cat = np.array([item['label_preds'] for item in curr_results], np.int32) # N
        track_cat = np.array([track['label_preds'] for track in curr_tracks], np.int32) # M

        max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in curr_results], np.float32)

        tracks = np.array(
        [pre_det['ct'] for pre_det in curr_tracks], np.float32) # M x 2

        if len(tracks) > 0:  # NOT FIRST FRAME
            dist = (((tracks.reshape(1, -1, 2) - \
                        dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist) # absolute distance in meter

            invalid = ((dist > max_diff.reshape(N, 1)) + \
            (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

            dist = dist  + invalid * 1e18
            if self.hungarian:
                dist[dist > 1e18] = 1e18
                row_ind, col_ind = linear_assignment(copy.deepcopy(dist))
                matched_indices = np.concatenate((row_ind.reshape(-1,1), col_ind.reshape(-1,1)), axis=-1)
            else:
                matched_indices = greedy_assignment(copy.deepcopy(dist))
        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) \
            if not (d in matched_indices[:, 0])]

        unmatched_tracks = [d for d in range(tracks.shape[0]) \
            if not (d in matched_indices[:, 1])]
        
        if self.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        for m in matches:
            track = curr_results[m[0]]
            track['tracking_id'] = curr_tracks[m[1]]['tracking_id']      
            
            refine_score = TRK_REF[track['detection_name']]['ref']
            # NEW IDEA
            if refine_score:
                alpha, beta = TRK_REF[track['detection_name']]['alpha'], TRK_REF[track['detection_name']]['beta']
                prev_track_conf = curr_tracks[m[1]]['ref_detection_score']
                tp_prob = track['ref_detection_score']
                det_conf = track['detection_score']
                track['ref_detection_score'] = (tp_prob > alpha) * beta * det_conf + (1-beta) * prev_track_conf
                #track['ref_detection_score'] = beta * det_conf + (-1)**(tp_prob < alpha) * (1-beta) * prev_track_conf
            else:
                track['ref_detection_score'] = track['detection_score']
            
            track['age'] = 1
            track['active'] = curr_tracks[m[1]]['active'] + 1
            ret.append(track)

        for i in unmatched_dets:
            track = curr_results[i]
            if len(tracks) > 0:
                if 'newborn' not in track.keys() and (dist[i,:] <= self.NUSCENE_CLS_VELOCITY_ERROR[track['detection_name']]).sum():
                     continue
            self.id_count += 1
            track['tracking_id'] = self.id_count
            
            # NEW IDEA
            refine_score = TRK_REF[track['detection_name']]['ref']
            if refine_score:
                beta = TRK_REF[track['detection_name']]['beta']
                track['ref_detection_score'] = beta*track['detection_score']
            else:
                track['ref_detection_score'] = track['detection_score']
            
            #track['ref_detection_score'] = track['detection_score']

            track['age'] = 1
            track['active'] =  1
            ret.append(track)

        # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
        # the object in current frame 
        for i in unmatched_tracks:
            track = curr_tracks[i]
            if 'dead' in track.keys() and (dist[:,i] <= self.NUSCENE_CLS_VELOCITY_ERROR[track['detection_name']]).sum():
                continue
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']
                
                # NEW IDEA
                refine_score = TRK_REF[track['detection_name']]['ref']
                if refine_score:
                    beta = TRK_REF[track['detection_name']]['beta']
                    track['ref_detection_score'] = (1-beta)*track['ref_detection_score']
                

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1 # move forward 
                    track['ct'] = ct + offset 
                ret.append(track)

    self.tracks = ret
    return ret
