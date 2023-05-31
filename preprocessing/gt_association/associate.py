
import numpy as np
from mot_3d.utils.data_utils import BBox
np.set_printoptions(suppress=True)

def associate(gt_boxes, gt_types, pred_boxes, pred_types, threshold, distance_type="l2"):
    """

    Args:
        gt_boxes (list<BBox>):
        pred_boxes (list<BBox>):
        threshold: we do not consider a match (1) above this threshold for l2 (2) below this threshold for IOU
        distance_type: one of ["l2", "3D-IOU"]

    Returns:
        (list<Box3D>): updated list of boxes, where information of assigned boxes is merged
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        fp_inds = [i for i in range(len(pred_boxes))]
        fn_inds = [i for i in range(len(gt_boxes))]

        return [], [], pred_boxes, gt_boxes, [], pred_types, gt_types, {}, fp_inds, fn_inds

    # Sort predictions by confidence
    scores = [box.s for box in pred_boxes]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(scores))][::-1]
    pred_boxes = [pred_boxes[i] for i in sortind]
    pred_types = [pred_types[i] for i in sortind]

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    dist_fn = iou_3d if distance_type=="3D-IOU" else l2
    dists = dist_fn(gt_boxes, pred_boxes)

    # we always compare less than for thresholding
    if distance_type == "3D-IOU":
        threshold *= -1
        dists *= -1

    # run through and assign predicted boxes to GT labels
    taken = set()  # Initially no gt bounding box is matched.
    tp_ind_pairs, fp_inds = dict(), list()
    tp, fp, tp_matches = list(), list(), list()
    tp_types, fp_types = list(), list()
    for pred_idx, box in enumerate(pred_boxes):
        pred_type = pred_types[pred_idx]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes):
            # Find closest match among ground truth boxes
            if pred_type in gt_types[gt_idx] and gt_idx not in taken:
                this_distance = dists[gt_idx, pred_idx]
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        valid_match = min_dist < threshold
        
        if valid_match:
            taken.add(match_gt_idx)
            tp.append(box)
            tp_matches.append(gt_boxes[match_gt_idx])
            tp_types.append(pred_type)
            tp_ind_pairs[sortind[pred_idx]] = match_gt_idx
        else:
            fp.append(box)
            fp_types.append(pred_type)
            fp_inds.append(sortind[pred_idx])

    # get set of false negatives
    fn_inds = list()
    fn, fn_types = list(), list()
    for k, gt_box in enumerate(gt_boxes):
        if k not in taken:
            fn.append(gt_box)
            fn_types.append(gt_types[k])
            fn_inds.append(k)

    return tp, tp_matches, fp, fn, tp_types, fp_types, fn_types, tp_ind_pairs, fp_inds, fn_inds


def iou_3d(gts, preds):
    """
    Computes a matrix of 3D IOUs between two sets of bounding boxes.
    Args:
        gt (list<Box3D>): first set of boxes
        preds (list<Box3D>): second set of boxes

    Returns:
        ious (np.array)
    """
    # todo: update me to simpletrack bbox format
    # convert boxes to tensors
    gt_corners = np.stack([gt.gt_corners() for gt in gts])  # M x 8 x 3
    gt_corners = torch.from_numpy(gt_corners)

    # get predicted corners
    pred_corners = np.stack([pred.prediction_corners() for pred in preds])  # N x 8 x 3
    pred_corners = torch.from_numpy(pred_corners)

    # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
    intersection_vol, iou_3d = box3d_overlap(gt_corners.float(), pred_corners.float())
    return iou_3d.data.numpy()


def l2(gts, preds):
    gt_locs = np.stack([BBox.bbox2array(gt)[:3] for gt in gts])
    pred_locs = np.stack([BBox.bbox2array(pred)[:3] for pred in preds])
    gt_centers = np.stack([BBox.bbox2array(gt)[:3] for gt in gts]).reshape((-1, 1, 3))  # M x 3
    pred_centers = np.stack([BBox.bbox2array(pred)[:3] for pred in preds]).reshape((1, -1, 3))
    dists = np.linalg.norm(gt_centers[:, :, :2] - pred_centers[:, :, :2], axis=2)
    return dists