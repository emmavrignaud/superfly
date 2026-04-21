"""
src/association.py

What this file does
-------------------
Every frame, the detector outputs a set of bounding boxes ("detections") and
the Kalman filter predicts where each existing tracker ended up ("predictions").
This file answers the question: which detection belongs to which tracker?

It does that in two steps:
  1. Score every detection-tracker pair with a similarity metric (IoU or a variant)
  2. Run the Hungarian algorithm to find the globally best 1-to-1 assignment

IoU — Intersection over Union
------------------------------
The core similarity metric. Given two boxes, IoU measures how much they overlap:

    IoU = area of overlap / area of union

    IoU = 1.0  →  boxes are identical
    IoU = 0.0  →  boxes don't touch at all

If a fly is where the tracker predicted, IoU is high and they get matched.
If they're far apart, IoU is low and they don't match.

IoU variants
------------
Standard IoU only measures overlap — two boxes that don't touch at all both
get IoU = 0, whether they're 1px apart or 1000px apart. The variants fix this:

  giou  — penalises boxes that don't overlap based on how far apart they are
  diou  — also penalises large centre-to-centre distance
  ciou  — diou + penalises aspect ratio mismatch
  hmiou — IoU × vertical overlap ratio. Two boxes at very different heights
           get penalised even if they overlap horizontally. Useful for vial
           tracking where a fly near the top should never match one at the bottom.
           (Added from boxmot — not in the original OC-SORT.)

The OCM direction term (in associate())
----------------------------------------
OC-SORT adds a second term on top of IoU: velocity direction consistency (OCM).

For each tracker, we know the direction it was moving (computed from its
observation delta_t=3 frames ago vs now). We check: does the detection appear
in roughly the direction the tracker was already heading?

If a tracker was moving right, a detection to its right gets a bonus.
A detection to its left gets penalised — even if IoU is the same.

This reduces ID switches when two flies cross paths: the tracker moving right
won't steal the detection that belongs to the tracker moving left.

    cost = -(iou_matrix + angle_diff_cost)

Linear assignment
-----------------
Once we have the cost matrix, linear_assignment() finds the globally cheapest
1-to-1 matching using the Hungarian algorithm (via lap or scipy). No detection
can match two trackers and no tracker can match two detections.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)  


def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0 # resize from (-1,1) to (0,1)

def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 
    iou = wh / union

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)


def hmiou_batch(bboxes1, bboxes2):
    """
    Height-modulated IoU (boxmot extension, not in original OC-SORT).

    Standard IoU multiplied by the vertical overlap ratio o:
        o = intersection_height / union_height

    Two boxes at very different y-positions get o ≈ 0 and are penalised
    even if they overlap horizontally. Useful for vial tracking where flies
    at different heights should not be matched.
    """
    bboxes1 = np.expand_dims(bboxes1, axis=1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, axis=0)  # (1, M, 4)

    # Vertical overlap ratio
    intersect_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    intersect_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    intersection_height = np.maximum(0.0, intersect_y2 - intersect_y1)
    union_y1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    union_y2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    union_height = np.maximum(1e-10, union_y2 - union_y1)
    o = intersection_height / union_height

    # Standard IoU
    inter_x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    inter_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    inter_x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    inter_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    inter_area = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    iou = inter_area / (area1 + area2 - inter_area + 1e-10)

    return iou * o


def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist # resize to (0,1)



def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = np.setdiff1d(np.arange(len(detections)), matched_indices[:,0]) if matched_indices.shape[0] > 0 else np.arange(len(detections))
    unmatched_trackers = np.setdiff1d(np.arange(len(trackers)), matched_indices[:,1]) if matched_indices.shape[0] > 0 else np.arange(len(trackers))

    #filter out matched with low IOU
    if matched_indices.shape[0] > 0:
        iou_vals = iou_matrix[matched_indices[:,0], matched_indices[:,1]]
        low_iou_mask = iou_vals < iou_threshold
        unmatched_detections = np.concatenate([unmatched_detections, matched_indices[low_iou_mask, 0]])
        unmatched_trackers = np.concatenate([unmatched_trackers, matched_indices[low_iou_mask, 1]])
        matches = matched_indices[~low_iou_mask]
    else:
        matches = np.empty((0,2),dtype=int)

    return matches, unmatched_detections.astype(int), unmatched_trackers.astype(int)


def _filter_matches(matched_indices, iou_matrix, iou_threshold, num_dets, num_trks):
    """Shared helper to split matched_indices into matches/unmatched based on IOU threshold."""
    if matched_indices.shape[0] > 0:
        unmatched_dets = np.setdiff1d(np.arange(num_dets), matched_indices[:,0])
        unmatched_trks = np.setdiff1d(np.arange(num_trks), matched_indices[:,1])
        iou_vals = iou_matrix[matched_indices[:,0], matched_indices[:,1]]
        low_iou_mask = iou_vals < iou_threshold
        unmatched_dets = np.concatenate([unmatched_dets, matched_indices[low_iou_mask, 0]])
        unmatched_trks = np.concatenate([unmatched_trks, matched_indices[low_iou_mask, 1]])
        matches = matched_indices[~low_iou_mask]
    else:
        unmatched_dets = np.arange(num_dets)
        unmatched_trks = np.arange(num_trks)
        matches = np.empty((0,2),dtype=int)
    return matches, unmatched_dets.astype(int), unmatched_trks.astype(int)


def aspect_ratio_bonus_batch(detections, trackers, weight):
    """
    One-sided aspect ratio similarity bonus.

    For each detection-tracker pair, compute how similar their aspect ratios are.
    Similar shape → bonus; dissimilar shape → 0 (never penalised).

        r = w / h  for each box
        similarity = 1 - |r_det - r_trk| / (r_det + r_trk)   in [0, 1]
        bonus = max(0, similarity) * weight

    Shape: (n_det, n_trk)
    """
    det_w = detections[:, 2] - detections[:, 0]
    det_h = detections[:, 3] - detections[:, 1] + 1e-6
    det_r = (det_w / det_h)[:, np.newaxis]          # (n_det, 1)

    trk_w = trackers[:, 2] - trackers[:, 0]
    trk_h = trackers[:, 3] - trackers[:, 1] + 1e-6
    trk_r = (trk_w / trk_h)[np.newaxis, :]          # (1, n_trk)

    similarity = 1.0 - np.abs(det_r - trk_r) / (det_r + trk_r + 1e-6)
    return np.maximum(0.0, similarity) * weight


def behavioral_consistency_batch(detections, trk_profiles, trk_last_centers, behavioral_weight):
    """
    Behavioral consistency bonus for each (detection, tracker) pair.

    Asks two questions per pair:
      1. Speed plausibility — is the detection within a plausible distance
         given the tracker's typical movement speed?
            bonus_speed = max(0, 1 - excess / (median_speed + 1))
         where excess = max(0, dist - median_speed). Falls to 0 only when
         the detection is much farther than the tracker's typical step size.

      2. Scale consistency — does the detection box area match the tracker's
         typical box area?
            bonus_scale = 1 - |area_det - median_scale| / (area_det + median_scale + 1e-6)

    Final bonus = behavioral_weight × 0.5 × (bonus_speed + bonus_scale)

    Trackers with no profile yet (< 2 observations) contribute 0 bonus so
    they don't distort the cost matrix for newly spawned tracks.

    Parameters
    ----------
    detections       : (n_det, 5) array [x1,y1,x2,y2,score]
    trk_profiles     : list of n_trk dicts (from KalmanBoxTracker.behavioral_profile)
                       or None entries for trackers with insufficient history
    trk_last_centers : (n_trk, 2) array of (cx, cy) from last_observation
    behavioral_weight: scalar weight applied to the final bonus

    Returns
    -------
    bonus : (n_det, n_trk) float array
    """
    n_det = len(detections)
    n_trk = len(trk_profiles)
    bonus = np.zeros((n_det, n_trk), dtype=float)

    if behavioral_weight == 0.0 or n_det == 0 or n_trk == 0:
        return bonus

    det_cx = (detections[:, 0] + detections[:, 2]) / 2.0  # (n_det,)
    det_cy = (detections[:, 1] + detections[:, 3]) / 2.0
    det_areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])  # (n_det,)

    for j, prof in enumerate(trk_profiles):
        if prof is None:
            continue
        tcx, tcy = trk_last_centers[j]
        dist = np.sqrt((det_cx - tcx) ** 2 + (det_cy - tcy) ** 2)  # (n_det,)

        med_speed  = prof["median_speed"]
        mean_accel = prof.get("mean_acceleration", 0.0)
        # Acceleration-adjusted expected distance: if the tracker has been
        # speeding up (positive acceleration), it may move faster than its
        # median on the next step; if slowing down, expect less distance.
        expected_dist = max(med_speed + mean_accel, 0.0)
        excess     = np.maximum(0.0, dist - expected_dist)
        b_speed    = np.maximum(0.0, 1.0 - excess / (expected_dist + 1.0))  # (n_det,)

        med_scale  = prof["median_scale"]
        b_scale    = 1.0 - np.abs(det_areas - med_scale) / (det_areas + med_scale + 1e-6)  # (n_det,)
        b_scale    = np.maximum(0.0, b_scale)

        bonus[:, j] = behavioral_weight * 0.5 * (b_speed + b_scale)

    return bonus


def simulate_position(
    cx: float, cy: float,
    vx: float, vy: float,
    speed: float,
    acceleration: float,
    gap: int,
    vial_roi: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float]:
    """
    Simulate a tracker's centre position frame-by-frame over `gap` frames,
    with wall-bounce reflection when a vial_roi is provided.

    Ported from stitching.simulate_position so the same physics model is
    available during live tracking (used in link_cost_batch below).

    Parameters
    ----------
    cx, cy       : starting centre position (px)
    vx, vy       : unit velocity direction vector (from tracker.velocity)
    speed        : initial speed (px/frame), e.g. tracker profile median_speed
    acceleration : signed speed change per frame (profile mean_acceleration)
    gap          : number of frames to simulate forward
    vial_roi     : (x0, y0, x1, y1) bounding box; None = no reflection

    Returns
    -------
    (x, y) predicted centre after `gap` frames
    """
    x, y = float(cx), float(cy)
    spd  = float(speed)
    dvx, dvy = float(vx), float(vy)   # unit direction

    for _ in range(gap):
        nx = x + dvx * spd
        ny = y + dvy * spd

        if vial_roi is not None:
            x0, y0, x1, y1 = vial_roi
            if nx < x0:
                dvx = -dvx;  nx = x0 + (x0 - nx)
            elif nx > x1:
                dvx = -dvx;  nx = x1 - (nx - x1)
            if ny < y0:
                dvy = -dvy;  ny = y0 + (y0 - ny)
            elif ny > y1:
                dvy = -dvy;  ny = y1 - (ny - y1)

        x, y  = nx, ny
        spd   = max(spd + acceleration, 0.0)

    return x, y


def link_cost_batch(
    detections:        np.ndarray,
    trackers_state:    List[Dict],
    weights:           Optional[Dict] = None,
) -> np.ndarray:
    """
    Compute an (n_det, n_trk) cost matrix combining three terms — the same
    logic as stitching.link_score but applied live, per frame, to each
    detection-tracker candidate pair.

    Term 1 — Extrapolated position error (px)
        Simulate each tracker's centre forward by `gap` frames (time since
        last observation) using its velocity, speed, and acceleration.
        Error = distance between the predicted landing spot and the detection.

    Term 2 — Direction agreement [0, 1]
        Angle between tracker's velocity direction and the gap vector
        (tracker last centre → detection centre), normalised to [0, 1].
        0 = perfectly aligned, 1 = opposite directions.

    Term 3 — Behavioral dissimilarity
        Weighted z-score distance between tracker and detection kinematic
        proxies. Detection "profile" is estimated from its bbox area and
        the implied step distance from the tracker's last centre.

    Final cost = w_extrap * term1 + w_direction * term2 + w_behavioral * term3

    Parameters
    ----------
    detections     : (n_det, 5) array [x1,y1,x2,y2,score]
    trackers_state : list of n_trk dicts, each with keys:
                       'last_cx', 'last_cy'  — last observed centre
                       'velocity'            — (vy, vx) unit vector or None
                       'profile'             — behavioral_profile dict or None
                       'gap'                 — frames since last observation
                       'vial_roi'            — (x0,y0,x1,y1) or None
    weights        : optional dict with keys 'extrap', 'direction', 'behavioral'
                     (defaults to 1.0 each if not provided)

    Returns
    -------
    cost : (n_det, n_trk) float array — lower = better match
    """
    if weights is None:
        weights = {"extrap": 1.0, "direction": 1.0, "behavioral": 1.0}
    w_ext = weights.get("extrap",     1.0)
    w_dir = weights.get("direction",  1.0)
    w_beh = weights.get("behavioral", 1.0)

    n_det = len(detections)
    n_trk = len(trackers_state)
    cost  = np.zeros((n_det, n_trk), dtype=float)

    if n_det == 0 or n_trk == 0:
        return cost

    det_cx = (detections[:, 0] + detections[:, 2]) / 2.0   # (n_det,)
    det_cy = (detections[:, 1] + detections[:, 3]) / 2.0
    det_areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

    for j, ts in enumerate(trackers_state):
        tcx   = ts["last_cx"]
        tcy   = ts["last_cy"]
        vel   = ts.get("velocity")     # (vy, vx) unit vector or None
        prof  = ts.get("profile")
        gap   = max(int(ts.get("gap", 1)), 1)
        roi   = ts.get("vial_roi")

        speed    = prof["median_speed"]    if prof else 0.0
        accel    = prof.get("mean_acceleration", 0.0) if prof else 0.0

        # --- Term 1: extrapolated position error ---
        if vel is not None and (vel[0] != 0 or vel[1] != 0):
            vy_unit, vx_unit = float(vel[0]), float(vel[1])
            ex, ey = simulate_position(tcx, tcy, vx_unit, vy_unit,
                                       speed, accel, gap, roi)
        else:
            ex, ey = tcx, tcy

        extrap_err = np.sqrt((det_cx - ex) ** 2 + (det_cy - ey) ** 2)  # (n_det,)

        # --- Term 2: direction agreement ---
        dx = det_cx - tcx
        dy = det_cy - tcy
        gap_norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        if vel is not None:
            vy_unit, vx_unit = float(vel[0]), float(vel[1])
            dot = (dx / gap_norm) * vx_unit + (dy / gap_norm) * vy_unit
            direction_term = (1.0 - np.clip(dot, -1.0, 1.0)) / 2.0     # [0,1]
        else:
            direction_term = np.zeros(n_det)

        # --- Term 3: behavioral dissimilarity ---
        if prof is not None:
            med_speed = prof["median_speed"]
            med_scale = prof["median_scale"]
            # Detection proxies: step distance from tracker last centre, bbox area
            det_step  = np.sqrt((det_cx - tcx) ** 2 + (det_cy - tcy) ** 2)
            b_speed   = np.abs(det_step - med_speed) / (med_speed + 1.0)
            b_scale   = np.abs(det_areas - med_scale) / (det_areas + med_scale + 1e-6)
            beh_term  = 0.5 * (b_speed + b_scale)
        else:
            beh_term  = np.zeros(n_det)

        cost[:, j] = w_ext * extrap_err + w_dir * direction_term + w_beh * beh_term

    return cost


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, asso_func=iou_batch, aspect_weight=0.0, vial_mask=None, trk_profiles=None, trk_last_centers=None, behavioral_weight=0.0, link_trk_states=None, link_weights=None):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = asso_func(detections, trackers)

    # Vial-aware hard constraint: a detection in vial A can never match a tracker
    # last seen in vial B. Zero out cross-vial entries in the iou_matrix so they
    # fall below iou_threshold and are filtered by _filter_matches.
    # vial_mask is (n_det, n_trk) bool; True = same vial (or unknown). None = no constraint.
    if vial_mask is not None:
        iou_matrix = iou_matrix * vial_mask.astype(float)

    # Aspect ratio bonus: same shape → small bonus, different shape → 0
    bonus = aspect_ratio_bonus_batch(detections, trackers, aspect_weight)

    # Behavioral consistency bonus: speed plausibility + scale consistency
    if trk_profiles is not None and trk_last_centers is not None:
        bonus = bonus + behavioral_consistency_batch(
            detections, trk_profiles, trk_last_centers, behavioral_weight
        )

    # OCM: velocity direction consistency term.
    # For each tracker we know which direction it was moving (velocities[:,dy,dx]).
    # For each detection we compute the direction from the tracker's last
    # observation to the detection centre. If that matches the tracker's inertia,
    # we add a bonus; if opposite, a penalty. Trackers with no valid previous
    # observation (previous_obs[:,4] < 0) are masked to zero so they don't
    # contribute noise for freshly spawned trackers.
    Y, X = speed_direction_batch(detections, previous_obs)   # (n_trk, n_det)
    inertia_Y = velocities[:, 0][:, np.newaxis]              # (n_trk, 1)
    inertia_X = velocities[:, 1][:, np.newaxis]
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1, 1)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi  # (n_trk, n_det)
    valid_mask = (previous_obs[:, 4] >= 0).astype(float)[:, np.newaxis]     # (n_trk, 1)
    scores = detections[:, -1][np.newaxis, :]                                # (1, n_det)
    angle_diff_cost = (valid_mask * diff_angle * scores * vdc_weight).T      # (n_det, n_trk)

    # Link cost: full composite cost (extrapolated position + direction + behavioral)
    # used in the jump round where IoU alone is unreliable (inflated bboxes).
    # link_trk_states is a list of dicts built in ocsort.update(); None = disabled.
    if link_trk_states is not None and len(link_trk_states) > 0:
        lc = link_cost_batch(detections, link_trk_states, weights=link_weights)
        # Normalise to [0, 1] and convert to a bonus (lower cost = higher bonus)
        lc_max = lc.max()
        if lc_max > 0:
            lc_bonus = 1.0 - lc / lc_max
        else:
            lc_bonus = np.zeros_like(lc)
        bonus = bonus + lc_bonus

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + bonus + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0,2))

    return _filter_matches(matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers))


def associate_kitti(detections, trackers, det_cates, iou_threshold,
        velocities, previous_obs, vdc_weight):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = inertia_Y[:, np.newaxis]
    inertia_X = inertia_X[:, np.newaxis]
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[previous_obs[:,4] < 0] = 0
    valid_mask = valid_mask[:, np.newaxis]

    scores = detections[:,-1][:, np.newaxis]
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    cate_matrix = np.where(det_cates[:, np.newaxis] != trackers[np.newaxis, :, 4], -1e6, 0.0)

    cost_matrix = - iou_matrix -angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    return _filter_matches(matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers))