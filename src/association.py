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


def _filter_matches(matched_indices, iou_matrix, iou_threshold, num_dets, num_trks,
                    composite_matrix=None):
    """Shared helper to split matched_indices into matches/unmatched based on IOU threshold.

    Returns (matches, unmatched_dets, unmatched_trks, match_scores) where
    match_scores[i] is the composite association score for matches[i] — used by
    OCSort to record per-frame association confidence in each tracker's observation_log.

    composite_matrix : the full cost matrix (iou + behavioural + angle bonuses) used
        for the Hungarian assignment. When provided, match_scores reflect the actual
        decision quality rather than raw IoU alone. Threshold filtering still uses
        iou_matrix so the acceptance criterion is unchanged.
    """
    score_matrix = composite_matrix if composite_matrix is not None else iou_matrix
    if matched_indices.shape[0] > 0:
        unmatched_dets = np.setdiff1d(np.arange(num_dets), matched_indices[:,0])
        unmatched_trks = np.setdiff1d(np.arange(num_trks), matched_indices[:,1])
        iou_vals = iou_matrix[matched_indices[:,0], matched_indices[:,1]]
        low_iou_mask = iou_vals < iou_threshold
        unmatched_dets = np.concatenate([unmatched_dets, matched_indices[low_iou_mask, 0]])
        unmatched_trks = np.concatenate([unmatched_trks, matched_indices[low_iou_mask, 1]])
        matches = matched_indices[~low_iou_mask]
        match_scores = score_matrix[matches[:,0], matches[:,1]].tolist()
    else:
        unmatched_dets = np.arange(num_dets)
        unmatched_trks = np.arange(num_trks)
        matches = np.empty((0,2),dtype=int)
        match_scores = []
    return matches, unmatched_dets.astype(int), unmatched_trks.astype(int), match_scores


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
    direction: float,
    speed: float,
    acceleration: float,
    gap: int,
    vial_roi: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float]:
    """
    Simulate a tracker's centre position frame-by-frame over `gap` frames,
    with left/right wall-bounce reflection when a vial_roi is provided.

    Parameters
    ----------
    cx, cy       : starting centre position (px)
    direction    : heading in degrees (0 = right, 90 = down)
    speed        : initial speed (px/frame)
    acceleration : signed speed change per frame
    gap          : number of frames to simulate forward
    vial_roi     : (x0, y0, x1, y1) bounding box; None = no reflection

    Returns
    -------
    (x, y) predicted centre after `gap` frames
    """
    x, y = float(cx), float(cy)
    spd  = float(speed)
    rad  = np.radians(direction)
    dvx  = float(np.cos(rad))
    dvy  = float(np.sin(rad))

    for _ in range(gap):
        nx = x + dvx * spd
        ny = y + dvy * spd

        if vial_roi is not None:
            x0, y0, x1, y1 = vial_roi
            if nx < x0:
                dvx = -dvx;  nx = x0 + (x0 - nx)
            elif nx > x1:
                dvx = -dvx;  nx = x1 - (nx - x1)

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
        tcx     = ts["last_cx"]
        tcy     = ts["last_cy"]
        vel     = ts.get("velocity")     # (vy, vx) unit vector — fallback only
        prof    = ts.get("profile")
        gap     = max(int(ts.get("gap", 1)), 1)
        roi     = ts.get("vial_roi")
        history = ts.get("history", [])

        # Derive final heading and median speed from the full observation history
        # (same approach as stitching.link_score). Falls back to tracker.velocity
        # + behavioral profile if history is too short.
        if len(history) >= 2:
            centers = np.array([
                ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in history
            ])
            diffs  = np.diff(centers, axis=0)          # (N-1, 2)
            speeds = np.sqrt((diffs ** 2).sum(axis=1))  # px/frame
            pos_speeds = speeds[speeds > 0]
            median_speed = float(np.median(pos_speeds)) if len(pos_speeds) > 0 else 0.0
            last_dx, last_dy = float(diffs[-1, 0]), float(diffs[-1, 1])
            final_direction  = float(np.degrees(np.arctan2(last_dy, last_dx)))
            has_direction    = True
        elif vel is not None and (vel[0] != 0 or vel[1] != 0):
            vy_unit, vx_unit = float(vel[0]), float(vel[1])
            final_direction  = float(np.degrees(np.arctan2(vy_unit, vx_unit)))
            median_speed     = prof["median_speed"] if prof else 0.0
            has_direction    = True
        else:
            final_direction  = 0.0
            median_speed     = prof["median_speed"] if prof else 0.0
            has_direction    = False

        accel = prof.get("mean_acceleration", 0.0) if prof else 0.0

        # --- Term 1: extrapolated position error ---
        if has_direction and median_speed > 0:
            ex, ey = simulate_position(tcx, tcy, final_direction,
                                       median_speed, accel, gap, roi)
        else:
            ex, ey = tcx, tcy

        extrap_err = np.sqrt((det_cx - ex) ** 2 + (det_cy - ey) ** 2)  # (n_det,)

        # --- Term 2: direction agreement ---
        dx = det_cx - tcx
        dy = det_cy - tcy
        gap_norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        if has_direction:
            rad     = np.radians(final_direction)
            vx_unit = float(np.cos(rad))
            vy_unit = float(np.sin(rad))
            dot     = (dx / gap_norm) * vx_unit + (dy / gap_norm) * vy_unit
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


def behavioral_fingerprint_bonus(
    detections: np.ndarray,
    trk_obs_windows: list,
    trk_profiles: list,
    weights: dict,
) -> np.ndarray:
    """
    Per-feature behavioral fingerprint bonus for each (detection, tracker) pair.

    Uses the tracker's recent observation window to compute implied kinematics
    for each candidate detection, then scores how well those implied kinematics
    match the tracker's running behavioral profile.

    Features (all scores in [0, 1]):
      speed         — implied step distance vs tracker median speed
      scale         — detection bbox area vs tracker median scale
      turning_angle — implied heading change vs tracker mean turning angle
                      (requires len(obs) >= 2)
      pause         — whether detection implies a pause vs tracker pause fraction
                      (requires len(obs) >= 2)
      acceleration  — implied speed change vs tracker mean acceleration
                      (requires len(obs) >= 2)

    Parameters
    ----------
    detections       : (n_det, 5) array [x1,y1,x2,y2,score]
    trk_obs_windows  : list of n_trk lists, each a list of recent bbox arrays
                       [x1,y1,x2,y2] (most recent last)
    trk_profiles     : list of n_trk profile dicts (or None for new trackers)
    weights          : dict with keys "speed", "scale", "turning_angle",
                       "pause", "acceleration"

    Returns
    -------
    bonus : (n_det, n_trk) float array
    """
    n_det = len(detections)
    n_trk = len(trk_profiles)
    bonus = np.zeros((n_det, n_trk), dtype=float)

    if n_det == 0 or n_trk == 0:
        return bonus

    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return bonus

    det_cx = (detections[:, 0] + detections[:, 2]) / 2.0  # (n_det,)
    det_cy = (detections[:, 1] + detections[:, 3]) / 2.0
    det_areas = (
        (detections[:, 2] - detections[:, 0]) *
        (detections[:, 3] - detections[:, 1])
    )  # (n_det,)

    w_speed  = weights.get("speed",         0.0)
    w_scale  = weights.get("scale",         0.0)
    w_turn   = weights.get("turning_angle", 0.0)
    w_pause  = weights.get("pause",         0.0)
    w_accel  = weights.get("acceleration",  0.0)

    for j, (obs, prof) in enumerate(zip(trk_obs_windows, trk_profiles)):
        if prof is None or len(obs) == 0:
            continue

        last_bbox = obs[-1]
        last_cx = (last_bbox[0] + last_bbox[2]) / 2.0
        last_cy = (last_bbox[1] + last_bbox[3]) / 2.0

        has_direction = len(obs) >= 2
        if has_direction:
            prev_bbox = obs[-2]
            prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
            last_dx = last_cx - prev_cx
            last_dy = last_cy - prev_cy
            last_speed = float(np.sqrt(last_dx ** 2 + last_dy ** 2))
        else:
            last_speed = 0.0

        # implied speed: distance from tracker's last centre to each detection
        implied_speed = np.sqrt(
            (det_cx - last_cx) ** 2 + (det_cy - last_cy) ** 2
        )  # (n_det,)

        col = np.zeros(n_det, dtype=float)

        # --- speed score ---
        if w_speed != 0.0:
            med_speed = prof["median_speed"]
            speed_score = np.maximum(
                0.0,
                1.0 - np.abs(implied_speed - med_speed) / (med_speed + 1.0),
            )
            col += w_speed * speed_score

        # --- scale score ---
        if w_scale != 0.0:
            med_scale = prof["median_scale"]
            scale_score = np.maximum(
                0.0,
                1.0 - np.abs(det_areas - med_scale) / (det_areas + med_scale + 1e-6),
            )
            col += w_scale * scale_score

        # --- direction-dependent features ---
        if has_direction and (w_turn != 0.0 or w_pause != 0.0 or w_accel != 0.0):
            # implied direction vector: last_centre → detection
            det_dx = det_cx - last_cx  # (n_det,)
            det_dy = det_cy - last_cy
            det_norm = np.sqrt(det_dx ** 2 + det_dy ** 2) + 1e-9

            if w_turn != 0.0:
                last_norm = np.sqrt(last_dx ** 2 + last_dy ** 2) + 1e-9
                cos_angle = np.clip(
                    (det_dx * last_dx + det_dy * last_dy) / (det_norm * last_norm),
                    -1.0, 1.0,
                )
                implied_turning = np.degrees(np.arccos(cos_angle))  # [0, 180]
                mean_turn = prof["mean_turning_angle"]
                turning_score = np.maximum(
                    0.0,
                    1.0 - np.abs(implied_turning - mean_turn) / 180.0,
                )
                col += w_turn * turning_score

            if w_pause != 0.0:
                is_pause = (implied_speed < 1.0).astype(float)
                pause_frac = prof["pause_fraction"]
                pause_score = 1.0 - np.abs(is_pause - pause_frac)
                col += w_pause * pause_score

            if w_accel != 0.0:
                implied_accel = implied_speed - last_speed
                mean_accel = prof["mean_acceleration"]
                accel_score = np.maximum(
                    0.0,
                    1.0 - np.abs(implied_accel - mean_accel) /
                         (abs(mean_accel) + 1.0),
                )
                col += w_accel * accel_score

        bonus[:, j] = col

    return bonus


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, asso_func=iou_batch, aspect_weight=0.0, vial_mask=None, trk_profiles=None, trk_last_centers=None, behavioral_weight=0.0, behavioral_weights=None, trk_obs_windows=None, link_trk_states=None, link_weights=None, overlap_det_mask=None, overlap_iou_scale=0.1):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int), []

    iou_matrix = asso_func(detections, trackers)

    # Vial-aware hard constraint: a detection in vial A can never match a tracker
    # last seen in vial B. Zero out cross-vial entries in the iou_matrix so they
    # fall below iou_threshold and are filtered by _filter_matches.
    # vial_mask is (n_det, n_trk) bool; True = same vial (or unknown). None = no constraint.
    if vial_mask is not None:
        iou_matrix = iou_matrix * vial_mask.astype(float)

    # Overlap downscale: detections that touch another detection in the same vial
    # have their IoU contribution scaled down so link_cost_batch (trajectory
    # extrapolation) dominates the assignment instead.
    if overlap_det_mask is not None and overlap_det_mask.any():
        iou_matrix = iou_matrix.copy()
        iou_matrix[overlap_det_mask] *= overlap_iou_scale

    # Aspect ratio bonus: same shape → small bonus, different shape → 0
    bonus = aspect_ratio_bonus_batch(detections, trackers, aspect_weight)

    # Behavioral fingerprint bonus: per-feature weighted consistency scores.
    # When trk_obs_windows and behavioral_weights are provided, use the richer
    # multi-feature fingerprint; otherwise fall back to the legacy scalar path.
    if trk_obs_windows is not None and behavioral_weights is not None:
        bonus = bonus + behavioral_fingerprint_bonus(
            detections, trk_obs_windows, trk_profiles or [], behavioral_weights
        )
    elif trk_profiles is not None and trk_last_centers is not None:
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
        # Skip when there are zero detections this frame: lc has shape (0, n_trk),
        # so .max() raises ValueError on the zero-size array. With no detections
        # there is nothing to score, and the matched_indices block below already
        # handles min(iou_matrix.shape) == 0 correctly.
        if lc.size > 0:
            lc_max = lc.max()
            if lc_max > 0:
                lc_bonus = 1.0 - lc / lc_max
            else:
                lc_bonus = np.zeros_like(lc)
            bonus = bonus + lc_bonus

    composite = iou_matrix + bonus + angle_diff_cost

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-composite)
    else:
        matched_indices = np.empty(shape=(0,2))

    return _filter_matches(matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers),
                           composite_matrix=composite)


# ---------------------------------------------------------------------------
# Second-round re-linking helpers
# ---------------------------------------------------------------------------

# Normalisation scales for each behavioral feature (rough empirical ranges).
# Used to bring all features onto a similar 0–1 scale before L1 distance.
_RELINK_SCALES: dict = {
    "median_speed":          40.0,   # px/s; typical fly speed range
    "pause_fraction":         1.0,
    "mean_turning_angle":   180.0,   # degrees
    "mean_angular_velocity": 5400.0, # degrees/s  (180 deg × 30 fps)
    "mean_acceleration":     80.0,   # px/s²
    "n_large_displacements":  5.0,   # count; few per video segment
    "tortuosity":             3.0,
}


def _profile_from_obs(obs_list: list, fps: float = 30.0) -> "dict | None":
    """Compute a behavioral profile dict from a raw observation log.

    Parameters
    ----------
    obs_list : list of (frame_idx, bbox_array) pairs where bbox_array is
               [x1, y1, x2, y2] in pixel coordinates.
    fps      : frames per second, used to convert frame-step velocities to px/s.

    Returns None when there are fewer than 3 observations (not enough for
    velocity + acceleration).
    """
    import numpy as np

    if len(obs_list) < 3:
        return None

    obs_list = sorted(obs_list, key=lambda t: t[0])
    frames = np.array([t[0] for t in obs_list], dtype=float)
    bboxes = np.array([t[1] for t in obs_list], dtype=float)

    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

    dt = np.diff(frames) / fps          # seconds between consecutive obs
    dt = np.where(dt == 0, 1e-6, dt)

    dx = np.diff(cx)
    dy = np.diff(cy)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speeds = dist / dt                  # px/s

    # Turning angle (degrees) between consecutive displacement vectors
    angles = []
    for i in range(1, len(dx)):
        v1 = np.array([dx[i - 1], dy[i - 1]])
        v2 = np.array([dx[i], dy[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            angles.append(0.0)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_a))))

    mean_turning = float(np.mean(angles)) if angles else 0.0
    mean_ang_vel = mean_turning * fps

    # Acceleration (px/s²)
    dt2 = (dt[:-1] + dt[1:]) / 2.0
    dt2 = np.where(dt2 == 0, 1e-6, dt2)
    accel = np.abs(np.diff(speeds)) / dt2
    mean_accel = float(np.mean(accel)) if len(accel) > 0 else 0.0

    # Tortuosity: total path length / straight-line distance
    path_len = float(dist.sum())
    straight = float(np.sqrt((cx[-1] - cx[0]) ** 2 + (cy[-1] - cy[0]) ** 2))
    tortuosity = path_len / straight if straight > 1.0 else 1.0

    pause_thresh_px_s = 1.0  # px/s
    pause_fraction = float(np.mean(speeds < pause_thresh_px_s))

    large_disp_thresh = float(np.percentile(dist, 90)) if len(dist) >= 10 else float(dist.max() + 1)
    n_large = int(np.sum(dist > large_disp_thresh))

    return {
        "median_speed":           float(np.median(speeds)),
        "pause_fraction":         pause_fraction,
        "mean_turning_angle":     mean_turning,
        "mean_angular_velocity":  mean_ang_vel,
        "mean_acceleration":      mean_accel,
        "n_large_displacements":  n_large,
        "tortuosity":             tortuosity,
    }


def behavioral_profile_distance(
    profile_a: "dict | None",
    profile_b: "dict | None",
    weights: dict,
) -> float:
    """Weighted normalised L1 distance between two behavioral profiles.

    Returns 1.0 if either profile is None (max dissimilarity).
    """
    if profile_a is None or profile_b is None:
        return 1.0

    total_w = 0.0
    total_d = 0.0
    for feat, w in weights.items():
        if w == 0.0:
            continue
        scale = _RELINK_SCALES.get(feat, 1.0)
        if scale == 0.0:
            scale = 1.0
        va = profile_a.get(feat, 0.0)
        vb = profile_b.get(feat, 0.0)
        total_d += w * abs(va - vb) / scale
        total_w += w

    if total_w == 0.0:
        return 0.0
    return min(total_d / total_w, 1.0)


def relink_tracklets(
    trackers: list,
    weights: dict,
    min_length: int = 10,
    inconsistency_threshold: float = 0.4,
    swap_threshold: float = 0.2,
    confidence_weight: float = 1.0,
    fps: float = 30.0,
) -> list:
    """Second-round re-linking via a Hungarian cost matrix at each weak frame.

    Algorithm
    ---------
    1. Per-tracker adaptive threshold: a score is "low" if it falls below that
       tracker's own median IoU. A tracker that is generally hard to follow has
       a higher baseline, so only truly anomalous frames are flagged.

    2. Weak-point frames: frames where ≥2 trackers simultaneously have a low
       score — the moments most likely to coincide with an ID swap.

    3. Each tracker's split point is its *earliest* weak-point frame (the first
       moment of uncertainty).

    4. At each split frame, group all trackers that split there. Build a cost
       matrix C where:
           C[i,i] = dist(first_half_i, second_half_i)           (keep own)
           C[i,j] = dist(first_half_i, second_half_j)
                    * (1 + confidence_weight * score_j_at_split) (swap penalty)
       High-confidence match at split frame → expensive to take that second half.

    5. Solve Hungarian on C. Accept the full group assignment only if total cost
       improves by more than swap_threshold vs the diagonal (identity) assignment.
       Record pairwise transpositions as (id_a, id_b, split_frame).

    Parameters
    ----------
    trackers            : list of KalmanBoxTracker with `.observation_log`, `.id`
                          observation_log entries are (frame_idx, bbox, score).
    weights             : per-feature behavioral distance weights
    min_length          : min observations to consider a tracker
    inconsistency_threshold : reserved (unused, kept for API compatibility)
    swap_threshold      : minimum total cost improvement to accept a group assignment
    confidence_weight   : how strongly the split-frame IoU score penalises swapping
                          that tracker's second half away (0 = ignore scores)
    fps                 : frames per second

    Returns
    -------
    List of (id_a, id_b, split_frame) tuples (1-based IDs) — one per accepted
    transposition. For group assignments with cycles > 2, each edge in the cycle
    is emitted as a separate pair (tracking.py handles them in order).
    """
    import numpy as np
    from collections import defaultdict

    # ── 1. Per-tracker: sorted observations + median IoU score ───────────────
    tracker_data = []
    for trk in trackers:
        obs = getattr(trk, "observation_log", [])
        if len(obs) < min_length:
            continue
        obs_sorted = sorted(obs, key=lambda t: t[0])
        scores = [t[2] for t in obs_sorted if t[2] is not None]
        if len(scores) < 3:
            continue
        median_score = float(np.median(scores))
        tracker_data.append({
            "trk":          trk,
            "id":           trk.id + 1,    # 1-based
            "obs":          obs_sorted,
            "median_score": median_score,
        })

    if len(tracker_data) < 2:
        return []

    # ── 2. For each tracker, flag frames where its score is below its median ──
    # frame → list of (tracker_data, score_at_frame)
    frame_to_low: dict = defaultdict(list)
    for td in tracker_data:
        for frame_idx, bbox, score in td["obs"]:
            if score is not None and score < td["median_score"]:
                frame_to_low[frame_idx].append((td, float(score)))

    # ── 3. Weak frames: ≥2 trackers simultaneously below their median ─────────
    weak_frames = {
        f: entries for f, entries in frame_to_low.items() if len(entries) >= 2
    }
    if not weak_frames:
        return []

    # ── 4. Each tracker's split point = its earliest weak frame ──────────────
    tracker_split: dict = {}   # td["id"] → split_frame
    tracker_split_score: dict = {}  # td["id"] → score at that split frame
    for frame in sorted(weak_frames):
        for td, score in weak_frames[frame]:
            if td["id"] not in tracker_split:
                tracker_split[td["id"]] = frame
                tracker_split_score[td["id"]] = score

    # ── 5. Group trackers by split frame; solve a cost matrix per group ───────
    split_groups: dict = defaultdict(list)
    for td in tracker_data:
        if td["id"] in tracker_split:
            split_groups[tracker_split[td["id"]]].append(td)

    swaps: list = []
    used: set = set()

    for split_frame, group in sorted(split_groups.items()):
        group = [td for td in group if td["id"] not in used]
        if len(group) < 2:
            continue

        # Build halves for each member
        halves = []
        for td in group:
            obs = td["obs"]
            first  = [(f, b) for f, b, s in obs if f <  split_frame]
            second = [(f, b) for f, b, s in obs if f >= split_frame]
            if len(first) < 2 or len(second) < 2:
                continue
            p_first  = _profile_from_obs(first,  fps)
            p_second = _profile_from_obs(second, fps)
            halves.append({
                "td":       td,
                "p_first":  p_first,
                "p_second": p_second,
                # IoU score this tracker had at the split frame
                "split_score": tracker_split_score.get(td["id"], 0.0),
            })

        if len(halves) < 2:
            continue

        n = len(halves)

        # Cost matrix
        # Diagonal = self-assignment (keep own second half) — no confidence penalty
        # Off-diagonal = swap penalty proportional to how confident tracker j was
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                base = behavioral_profile_distance(
                    halves[i]["p_first"], halves[j]["p_second"], weights
                )
                if i == j:
                    C[i, j] = base
                else:
                    penalty = 1.0 + confidence_weight * halves[j]["split_score"]
                    C[i, j] = base * penalty

        diagonal_cost = float(np.trace(C))

        # Solve Hungarian (minimise total cost)
        assignment = linear_assignment(C)   # → array [[i, j], ...]
        assigned_cost = float(sum(C[int(a[0]), int(a[1])] for a in assignment))

        gain = diagonal_cost - assigned_cost
        if gain <= swap_threshold:
            continue

        # Emit pairwise transpositions from the assignment permutation
        perm = {int(a[0]): int(a[1]) for a in assignment}
        visited: set = set()
        for start in range(n):
            if start in visited or perm[start] == start:
                visited.add(start)
                continue
            # Walk the cycle
            cycle = []
            cur = start
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = perm[cur]
            # Decompose cycle into adjacent transpositions
            for k in range(len(cycle) - 1):
                ia, ib = cycle[k], cycle[k + 1]
                id_a = halves[ia]["td"]["id"]
                id_b = halves[ib]["td"]["id"]
                if id_a not in used and id_b not in used:
                    swaps.append((id_a, id_b, split_frame))
                    used.add(id_a)
                    used.add(id_b)

    return swaps


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