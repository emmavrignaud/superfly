"""
src/ocsort.py

What this file does
-------------------
The main tracker class. Runs once per frame and maintains a list of active
KalmanBoxTrackers — one per fly currently being tracked.

Adopted from the OC-SORT paper (Cao et al., 2022). hmiou added from boxmot.

Two classes
-----------
KalmanBoxTracker — one instance per tracked fly. Owns:

  self.kf                 the Kalman filter (state, matrices). Holds the
                          filter's best estimate of where the fly is and how
                          fast it's moving. See kalmanfilter.py.

  self.observations       dict {age: bbox} — every real detection this tracker
                          has ever been matched to, keyed by tracker age (frames).
                          Stored separately from the Kalman state because the
                          state is smoothed; sometimes you want the raw measurement.

  self.velocity           unit direction vector [dy, dx] — NOT a speed.
                          Computed from the observation delta_t=3 frames ago vs
                          now. Used only for the OCM direction term in associate().
                          Note: the Kalman state also has vx, vy internally, but
                          those are used only for position prediction — this
                          self.velocity is what OC-SORT actually uses for association.

  self.hits               total frames ever matched to a detection (counts up forever)
  self.hit_streak         consecutive frames matched without a miss (resets on any gap)
  self.time_since_update  frames since the last real detection (resets on match)

  Together these control the tracker lifecycle:
    hit_streak >= min_hits       → tracker is confident enough to appear in the output CSV
    time_since_update > max_age  → tracker has been lost too long, remove it

  Note: min_hits=10 means a fly must be detected in 10 consecutive frames before
  it ever appears in the CSV. One missed frame resets the clock. This is conservative
  — a key parameter to tune for flies, and a candidate for redesign (see below).

OCSort — the main tracker. update() runs once per frame:

  Step 1 — predict()
    Every active tracker's Kalman filter advances one frame forward.
    Trackers that produce NaN positions (can happen if scale drifts negative
    under fast motion) are deleted immediately.

  Step 2 — round 1 association
    associate() matches high-confidence detections to tracker predictions
    using IoU + OCM. Matched detections update their tracker.

  Step 3 — round 2 association
    For trackers that didn't match in round 1: try again using last_observation
    (the last real bbox) instead of the Kalman prediction. Useful when the
    Kalman has drifted away from reality during a gap.

  Step 4 — BYTE (disabled, use_byte=False)
    Would try low-confidence detections against still-unmatched trackers.
    Not used in this pipeline.

  Step 5 — spawn and cull
    Unmatched detections become new trackers.
    Trackers with no detection for > max_age frames are removed.

  Step 6 — output
    Only trackers matched THIS frame (time_since_update < 1) AND with enough
    consecutive hits (hit_streak >= min_hits) are returned.

Design note
-----------
Currently the tracker aggressively prunes: short tracks are suppressed by
min_hits, lost tracks are permanently deleted after max_age frames. This means
stitching only sees a pre-filtered subset of what was detected.

A planned redesign: lower min_hits to 1 (emit everything), and instead of
deleting lost trackers, move them to a dead_trackers store. Stitching then
works from the full picture — linking short fragments and bridging long gaps
that the tracker currently throws away.
"""
# This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
from __future__ import print_function

import numpy as np
from .association import *
from .association import link_cost_batch
from .kalmanfilter import KalmanFilterNew


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    # Pre-computed constant matrices shared across all trackers
    _F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                        0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
    _H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)

    def __init__(self, bbox, delta_t=3, orig=False, brownian_pos_noise=1.0, vial_roi=None, fps=30.0):
        """
        Initialises a tracker using initial bounding box.

        brownian_pos_noise : scale factor applied to Q[cx] and Q[cy].
            1.0 = original behaviour.
            Higher values let the filter tolerate larger positional deviations
            per frame (e.g. saccades), reducing spurious ID switches.
        """
        # define constant velocity model
        if not orig:
          self.kf = KalmanFilterNew(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = KalmanBoxTracker._F
        self.kf.H = KalmanBoxTracker._H

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        # Q is the process noise covariance matrix — it describes how much each
        # state variable [cx, cy, s, r, vx, vy, vs] is expected to deviate from
        # the model's prediction in a single frame due to unpredictable motion.
        # Each diagonal entry Q[i,i] is the variance for one variable:
        #   Q[0,0] = uncertainty in cx (x position)
        #   Q[1,1] = uncertainty in cy (y position)
        #   Q[2,2] = uncertainty in s  (scale/area)
        #   Q[3,3] = uncertainty in r  (aspect ratio)
        #   Q[4,4] = uncertainty in vx, Q[5,5] = vy, Q[6,6] = vs (velocities)
        # Off-diagonal entries would capture correlated noise between variables;
        # we leave them at zero (independent noise per dimension).
        #
        # Velocities are set very low (0.01) — the filter trusts its velocity
        # estimate and assumes the fly mostly maintains constant speed/direction.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # brownian_pos_noise scales the x and y position uncertainty.
        # At 1.0 the filter expects ~1px of unexplained deviation per frame.
        # Higher values (e.g. 5.0) widen the tolerance for sudden direction
        # changes (saccades): the prediction uncertainty grows faster, so a
        # detection further from the prediction can still be matched to the
        # same track instead of spawning a new ID.
        # Only cx and cy are scaled — association happens in pixel space and
        # saccades are positional failures, not size or velocity failures.
        self.kf.Q[0, 0] *= brownian_pos_noise  # cx
        self.kf.Q[1, 1] *= brownian_pos_noise  # cy

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.fps = fps             # used to compute mean_angular_velocity in behavioral_profile
        self.observation_log: list = []  # [(frame_idx, bbox), ...] — real detections only
        self.frame_born: int = -1  # absolute frame index when this tracker was spawned
        self.vial_roi = vial_roi  # (x0, y0, x1, y1) or None — used for wall-bounce prediction

    def update(self, bbox, frame_idx: int = -1, score: float | None = None):
        """
        Updates the state vector with observed bbox.
        frame_idx : absolute frame number — stored in observation_log for real detections.
        score     : IoU association score for this match, recorded for re-linking.
        """
        if bbox is not None:
            self.observation_log.append((frame_idx, bbox[:4].copy(), score))
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.

        If a vial_roi was set at construction, the predicted centre is reflected
        back inside the vial when it would land outside (wall bounce). The Kalman
        state itself is not modified — only the box returned for association is
        corrected. This means the IoU between the prediction and a real post-bounce
        detection is much higher, preventing spurious track breaks at vial walls.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        box = convert_x_to_bbox(self.kf.x)[0]  # [x1, y1, x2, y2]

        if self.vial_roi is not None:
            vx0, vy0, vx1, vy1 = self.vial_roi
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            w  = box[2] - box[0]
            h  = box[3] - box[1]
            # Only reflect off left/right walls — flies move freely vertically
            if cx < vx0:
                cx = 2 * vx0 - cx
            elif cx > vx1:
                cx = 2 * vx1 - cx
            box = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        self.history.append(box.reshape(1, 4))
        return self.history[-1]

    def predict_jump(self, jump_factor: float) -> np.ndarray:
        """
        Return a jump-round prediction bbox WITHOUT modifying the Kalman state.

        Two effects controlled by jump_factor:
          1. Position extrapolation — the predicted centre is pushed further along
             the tracker's current velocity:
               cx_jump = cx + vx * jump_factor
               cy_jump = cy + vy * jump_factor
          2. Bbox size inflation — the area (scale) is multiplied by jump_factor²,
             making the box jump_factor times wider and taller. This increases IoU
             overlap with detections that are near but not perfectly aligned.

        Wall bounce (left/right only) is applied when a vial_roi is set, same as
        the regular predict(). The Kalman state is never touched.
        """
        x  = self.kf.x
        cx = x[0].item() + x[4].item() * jump_factor
        cy = x[1].item() + x[5].item() * jump_factor
        s  = max(x[2].item() * (jump_factor ** 2), 1.0)   # inflate area
        r  = x[3].item()

        w = np.sqrt(s * r)
        h = s / (w + 1e-6)

        if self.vial_roi is not None:
            vx0, vy0, vx1, vy1 = self.vial_roi
            if cx < vx0:
                cx = 2 * vx0 - cx
            elif cx > vx1:
                cx = 2 * vx1 - cx

        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    @property
    def behavioral_profile(self):
        """
        Rolling kinematic summary computed from all raw observations so far.

        Returns a dict with:
          median_speed           — median centre-to-centre distance between consecutive
                                   observations (px/frame). Proxy for how fast this fly moves.
          median_scale           — median bounding-box area (px²). Proxy for detected size.
          pause_fraction         — fraction of steps where speed < 1 px/frame.
          mean_turning_angle     — mean absolute heading change between consecutive steps
                                   (degrees). High = erratic; low = straight-line mover.
          mean_acceleration      — mean signed speed change per step (px/frame²). Positive =
                                   speeding up on average; negative = slowing down.
          n_large_displacements  — number of steps where speed > 2× median speed (burst count).
          tortuosity             — path length / straight-line displacement. 1.0 = perfectly
                                   straight; higher = more winding path.
          area_covered           — convex hull area of all visited centre positions (px²).
                                   Proxy for how much of the vial the fly has explored.

        Returns None when fewer than 2 observations are available (can't compute
        kinematics from a single point).
        """
        obs = self.history_observations
        if len(obs) < 2:
            return None

        centers = np.array([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in obs])
        diffs   = np.diff(centers, axis=0)                        # (N-1, 2)
        speeds  = np.sqrt((diffs ** 2).sum(axis=1))               # px/frame

        headings = np.arctan2(diffs[:, 1], diffs[:, 0])           # radians
        if len(headings) >= 2:
            delta_h = np.diff(headings)
            # wrap to [-pi, pi]
            delta_h = (delta_h + np.pi) % (2 * np.pi) - np.pi
            mean_turning_angle = float(np.degrees(np.abs(delta_h).mean()))
        else:
            mean_turning_angle = 0.0

        scales = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in obs])

        median_speed = float(np.median(speeds))

        # n_large_displacements: burst steps where speed exceeds 2× median
        n_large_displacements = int((speeds > 2 * median_speed).sum())

        # mean_acceleration: mean signed change in speed between consecutive steps
        mean_acceleration = float(np.diff(speeds).mean()) if len(speeds) >= 2 else 0.0

        # tortuosity: total path length divided by straight-line displacement
        path_length   = float(speeds.sum())
        dx = centers[-1, 0] - centers[0, 0]
        dy = centers[-1, 1] - centers[0, 1]
        straight_line = float(np.sqrt(dx ** 2 + dy ** 2))
        tortuosity    = path_length / (straight_line + 1e-6)

        # area_covered: convex hull area of all visited centre positions
        if len(centers) >= 3:
            try:
                from scipy.spatial import ConvexHull
                area_covered = float(ConvexHull(centers).volume)  # .volume = area in 2D
            except Exception:
                # Fallback for collinear points or missing scipy
                area_covered = float(
                    (centers[:, 0].max() - centers[:, 0].min()) *
                    (centers[:, 1].max() - centers[:, 1].min())
                )
        else:
            area_covered = 0.0

        return {
            "median_speed":          median_speed,
            "median_scale":          float(np.median(scales)),
            "pause_fraction":        float((speeds < 1.0).mean()),
            "mean_turning_angle":    mean_turning_angle,
            "mean_angular_velocity": mean_turning_angle * self.fps,  # degrees/second
            "mean_acceleration":     mean_acceleration,
            "n_large_displacements": n_large_displacements,
            "tortuosity":            tortuosity,
            "area_covered":          area_covered,
        }


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
def _select_prefix(costs, n_active, expected, w_under, w_over):
    """
    Accept the cheapest prefix of `costs` that minimises

        Σ costs[:k]  +  w_under · max(n_active+k − expected, 0)
                      +  w_over  · max(expected − (n_active+k), 0)

    Used to decide how many unmatched detections to spawn as new trackers.
    Cost of spawning detection i = (1 − confidence_i), so high-confidence
    detections are preferred. w_under >> w_over means we prefer to over-spawn
    slightly rather than leave real flies untracked.

    Parameters
    ----------
    costs    : list of floats, sorted ascending (cheapest spawn first)
    n_active : number of live trackers before spawning
    expected : target number of active trackers
    w_under  : penalty per tracker below expected (fragmentation cost)
    w_over   : penalty per tracker above expected (false-positive cost)

    Returns
    -------
    k : int — number of detections to spawn (0 = spawn nothing)
    """
    def _penalty(n):
        dev = n - expected
        return w_under * max(dev, 0) + w_over * max(-dev, 0)

    best_k     = 0
    best_total = _penalty(n_active)   # k=0 baseline
    running    = 0.0
    for k, c in enumerate(costs, start=1):
        running += c
        total = running + _penalty(n_active + k)
        if total < best_total:
            best_total = total
            best_k     = k
    return best_k


def _scale_weights(weights: dict, scale: float) -> dict:
    """Return a copy of weights with every value multiplied by scale."""
    return {k: v * scale for k, v in weights.items()}


ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist,
                "hmiou": hmiou_batch}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False,
        brownian_pos_noise=1.0, vial_rois=None, aspect_weight=0.0,
        behavioral_weights=None,
        overlap_weight_scale=6.0,
        jump_factor=2.0, jump_iou_threshold=0.05, jump_inertia=0.05,
        expected_count=None, w_under=15.0, w_over=2.0,
        overlap_iou_scale=0.1, edge_fraction=0.1,
        fps=30.0,
        relink_behavioral_weights=None, relink_min_length=10,
        relink_inconsistency_threshold=0.4, relink_swap_threshold=0.2,
        relink_confidence_weight=1.0):
        """
        Sets key parameters for SORT

        expected_count      : total expected number of flies across all vials. When set,
            _select_prefix is applied when spawning new trackers so the active
            tracker count is steered toward this target. None = disabled (spawn all).
        w_under             : penalty per tracker below expected_count (fragmentation cost).
        w_over              : penalty per tracker above expected_count (false-positive cost).
        overlap_iou_scale   : when two detections in the same vial overlap (IoU > 0),
            their IoU contribution to the association cost matrix is multiplied by
            this factor. 0.1 = trust IoU 10x less, letting link_cost_batch
            (trajectory extrapolation) dominate instead.
        edge_fraction       : detections within this fraction of the vial width/height
            from a wall are excluded from overlap handling (wall-adjacent flies).
        behavioral_weights  : dict of per-feature weights for the behavioral fingerprint
            bonus in the live cost matrix. Keys: "speed", "scale", "turning_angle",
            "pause", "acceleration". None uses a sensible default.
        overlap_weight_scale: multiplier applied to all behavioral_weights when
            detections overlap. Boosts behavioral signal when IoU is unreliable.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.brownian_pos_noise = brownian_pos_noise
        self.vial_rois = vial_rois  # {vial_id: (x0,y0,x1,y1)} or None
        self.aspect_weight = aspect_weight
        # Per-feature weights for the behavioral fingerprint bonus in the live
        # cost matrix. Defaults provide a gentle speed + scale signal equivalent
        # to the legacy behavioral_weight=0.05 (0.5 × 0.05 per feature).
        self.behavioral_weights = behavioral_weights if behavioral_weights is not None else {
            "speed":         0.025,
            "scale":         0.025,
            "turning_angle": 0.0,
            "pause":         0.0,
            "acceleration":  0.0,
        }
        # Multiplier applied to all behavioral_weights when detections overlap.
        self.overlap_weight_scale = overlap_weight_scale
        self.jump_factor = jump_factor
        self.jump_iou_threshold = jump_iou_threshold
        self.jump_inertia = jump_inertia
        self.expected_count = expected_count
        self.w_under = w_under
        self.w_over  = w_over
        self.overlap_iou_scale = overlap_iou_scale
        self.edge_fraction = edge_fraction
        self.fps = fps
        self.relink_behavioral_weights = relink_behavioral_weights or {
            "median_speed":          1.0,
            "pause_fraction":        1.0,
            "mean_turning_angle":    1.0,
            "mean_angular_velocity": 1.0,
            "mean_acceleration":     1.0,
            "n_large_displacements": 1.0,
            "tortuosity":            1.0,
        }
        self.relink_min_length              = relink_min_length
        self.relink_inconsistency_threshold = relink_inconsistency_threshold
        self.relink_swap_threshold          = relink_swap_threshold
        self.relink_confidence_weight       = relink_confidence_weight
        KalmanBoxTracker.count = 0

        # --- Diagnostics ---
        # detection_log: one entry per frame — (frame_idx, n_detections, n_emitted)
        #   n_detections = raw boxes above det_thresh passed to the tracker
        #   n_emitted    = tracks returned to the caller (survived min_hits gate)
        #   gap between the two = signal suppressed by min_hits or lost in association
        self.detection_log = []

        # suppressed_tracks: trackers that died (max_age exceeded) before ever reaching min_hits.
        #   These are real detections the tracker saw but silently discarded —
        #   short fragments that never made it into the CSV.
        #   Each entry contains the full (frame, cx, cy) trajectory so they can be
        #   plotted alongside emitted tracks in diagnostics.
        self.suppressed_tracks = []

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        n_dets = len(dets)  # raw detection count this frame, logged at the end

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        _zero_vel = np.array((0, 0))
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else _zero_vel for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        _dt = self.frame_count if self.delta_t < 0 else self.delta_t
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, _dt) for trk in self.trackers])

        # Behavioral fingerprinting active? When all weights are zero the
        # bonus term multiplies out to nothing, so skip the expensive profile
        # and obs-window builds (~65% of per-frame work).
        _bw_active = self.behavioral_weights is not None and any(
            abs(v) > 0 for v in self.behavioral_weights.values()
        )

        # Behavioral profiles for each active tracker (None if < 2 observations)
        if _bw_active:
            trk_profiles = [trk.behavioral_profile for trk in self.trackers]
        else:
            trk_profiles = [None] * len(self.trackers)
        trk_last_centers = np.array([
            ((trk.last_observation[0] + trk.last_observation[2]) / 2.0,
             (trk.last_observation[1] + trk.last_observation[3]) / 2.0)
            if trk.last_observation.sum() >= 0 else (0.0, 0.0)
            for trk in self.trackers
        ])

        # Recent observation windows for behavioral fingerprint matching.
        # Each entry is a list of raw bbox arrays [x1,y1,x2,y2], most recent last,
        # limited to the last _dt observations so cost stays proportional to history used.
        if _bw_active:
            trk_obs_windows = []
            for trk in self.trackers:
                obs = [b for _, b, *_ in sorted(trk.observation_log, key=lambda t: t[0])]
                trk_obs_windows.append(obs[-_dt:] if _dt > 0 else obs)
        else:
            trk_obs_windows = [[] for _ in self.trackers]

        # Build tracker state dicts for link_cost_batch — built once here and
        # reused in both round 1 and the jump round (jump round subsets by index).
        all_trk_states = []
        for trk in self.trackers:
            lo  = trk.last_observation
            tcx = (lo[0] + lo[2]) / 2.0 if lo.sum() >= 0 else 0.0
            tcy = (lo[1] + lo[3]) / 2.0 if lo.sum() >= 0 else 0.0
            all_trk_states.append({
                "last_cx":  tcx,
                "last_cy":  tcy,
                "velocity": trk.velocity,
                "profile":  trk.behavioral_profile if _bw_active else None,
                "gap":      trk.time_since_update,
                "vial_roi": trk.vial_roi,
                "history":  trk.history_observations,
            })

        """
            First round of association
        """
        # Vial membership for detections — computed once, reused for vial_mask
        # and overlap detection below.
        def _vial_of(cx, cy):
            if self.vial_rois is None:
                return None
            for vid, (x0, y0, x1, y1) in self.vial_rois.items():
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return vid
            return None

        det_vials = [
            _vial_of((d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0)
            for d in dets
        ] if len(dets) > 0 else []

        # Overlap detection: flag detections that touch another detection in the
        # same vial (IoU > 0), excluding detections near the vial wall.
        # For flagged detections, IoU is scaled down in associate() so that
        # link_cost_batch (trajectory extrapolation) dominates instead.
        overlap_det_mask = np.zeros(len(dets), dtype=bool)
        if self.vial_rois is not None and len(dets) > 1:
            det_iou_matrix = iou_batch(dets[:, :4], dets[:, :4])
            np.fill_diagonal(det_iou_matrix, 0.0)
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    if det_iou_matrix[i, j] <= 0:
                        continue
                    if det_vials[i] is None or det_vials[i] != det_vials[j]:
                        continue
                    # Exclude wall-adjacent detections
                    x0, y0, x1, y1 = self.vial_rois[det_vials[i]]
                    ef = self.edge_fraction
                    w, h = x1 - x0, y1 - y0
                    def _near_edge(d):
                        cx = (d[0] + d[2]) / 2.0
                        cy = (d[1] + d[3]) / 2.0
                        return (cx - x0 < ef * w or x1 - cx < ef * w or
                                cy - y0 < ef * h or y1 - cy < ef * h)
                    if not _near_edge(dets[i]) and not _near_edge(dets[j]):
                        overlap_det_mask[i] = True
                        overlap_det_mask[j] = True

        # Vial-aware mask: True where detection i and tracker j are in the same vial
        # (or either is outside all vials — we don't constrain those).
        vial_mask = None
        if self.vial_rois is not None and len(dets) > 0 and len(self.trackers) > 0:
            trk_vials = [
                _vial_of(
                    (trk.last_observation[0] + trk.last_observation[2]) / 2.0,
                    (trk.last_observation[1] + trk.last_observation[3]) / 2.0,
                ) if trk.last_observation.sum() >= 0 else None
                for trk in self.trackers
            ]
            vial_mask = np.array([
                [(dv is None or tv is None or dv == tv) for tv in trk_vials]
                for dv in det_vials
            ], dtype=bool)

        matched, unmatched_dets, unmatched_trks, match_scores = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia, self.asso_func, self.aspect_weight,
            vial_mask=vial_mask,
            trk_profiles=trk_profiles, trk_last_centers=trk_last_centers,
            behavioral_weights=_scale_weights(
                self.behavioral_weights,
                self.overlap_weight_scale if overlap_det_mask.any() else 1.0,
            ) if _bw_active else None,
            trk_obs_windows=trk_obs_windows if _bw_active else None,
            link_trk_states=all_trk_states,
            overlap_det_mask=overlap_det_mask,
            overlap_iou_scale=self.overlap_iou_scale)
        for m, sc in zip(matched, match_scores):
            self.trackers[m[1]].update(dets[m[0], :], frame_idx=self.frame_count, score=sc)

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :], frame_idx=self.frame_count)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Jump round: for still-unmatched pairs, use inflated predictions.
        # predict_jump() pushes the predicted centre further along velocity and
        # inflates the bbox, giving a wider search radius at lower confidence.
        # OCM weight is reduced (jump_inertia) since direction is less reliable
        # for a fly that may have changed course during the gap.
        if self.jump_factor > 0 and unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]

            jump_boxes = np.array([
                np.append(self.trackers[t].predict_jump(self.jump_factor), 0)
                for t in unmatched_trks
            ])

            jump_vial_mask = None
            if vial_mask is not None:
                jump_vial_mask = vial_mask[np.ix_(unmatched_dets, unmatched_trks)]

            jump_overlap_mask = overlap_det_mask[unmatched_dets] if overlap_det_mask.any() else None

            jump_profiles     = [trk_profiles[t]     for t in unmatched_trks]
            jump_last_centers = trk_last_centers[unmatched_trks]
            jump_velocities   = velocities[unmatched_trks]
            jump_k_obs        = k_observations[unmatched_trks]

            # Subset all_trk_states and obs_windows for the unmatched trackers only.
            jump_trk_states  = [all_trk_states[t]   for t in unmatched_trks]
            jump_obs_windows = [trk_obs_windows[t]  for t in unmatched_trks]

            jump_matched, jump_ud, jump_ut, jump_scores = associate(
                left_dets, jump_boxes,
                self.jump_iou_threshold,
                jump_velocities, jump_k_obs, self.jump_inertia,
                self.asso_func, self.aspect_weight,
                vial_mask=jump_vial_mask,
                trk_profiles=jump_profiles,
                trk_last_centers=jump_last_centers,
                behavioral_weights=_scale_weights(
                    self.behavioral_weights,
                    self.overlap_weight_scale if (jump_overlap_mask is not None and jump_overlap_mask.any()) else 1.0,
                ) if _bw_active else None,
                trk_obs_windows=jump_obs_windows if _bw_active else None,
                link_trk_states=jump_trk_states,
                overlap_det_mask=jump_overlap_mask,
                overlap_iou_scale=self.overlap_iou_scale,
            )

            for m, sc in zip(jump_matched, jump_scores):
                det_ind = unmatched_dets[m[0]]
                trk_ind = unmatched_trks[m[1]]
                self.trackers[trk_ind].update(dets[det_ind, :], frame_idx=self.frame_count, score=sc)

            unmatched_dets = unmatched_dets[jump_ud]
            unmatched_trks = unmatched_trks[jump_ut]

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Count-aware spawning: use _select_prefix to decide how many of the
        # unmatched detections to turn into new trackers. Detections are sorted
        # by descending confidence so the cheapest (most reliable) are tried first.
        # If expected_count is None, all unmatched detections are spawned as before.
        if self.expected_count is not None and len(unmatched_dets) > 0:
            n_active    = len(self.trackers)
            sort_order  = sorted(range(len(unmatched_dets)),
                                 key=lambda k: -float(dets[unmatched_dets[k], 4]))
            sorted_udets = [unmatched_dets[k] for k in sort_order]
            costs        = [1.0 - float(dets[i, 4]) for i in sorted_udets]
            k_spawn      = _select_prefix(costs, n_active, self.expected_count,
                                          self.w_under, self.w_over)
            unmatched_dets = sorted_udets[:k_spawn]

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            cx = (dets[i, 0] + dets[i, 2]) / 2.0
            cy = (dets[i, 1] + dets[i, 3]) / 2.0
            vial_roi = None
            if self.vial_rois is not None:
                for roi in self.vial_rois.values():
                    x0, y0, x1, y1 = roi
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        vial_roi = roi
                        break
            trk = KalmanBoxTracker(dets[i, :], delta_t=max(self.delta_t, 1),
                                   brownian_pos_noise=self.brownian_pos_noise,
                                   vial_roi=vial_roi, fps=self.fps)
            trk.frame_born = self.frame_count
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                # if this tracker never reached min_hits it was never emitted —
                # save it to the graveyard so diagnostics can see what was lost
                if trk.hits < self.min_hits:
                    frame_born = self.frame_count - trk.age
                    # compute (frame, cx, cy) for each observation in order
                    # history_observations is a list of raw bboxes [x1,y1,x2,y2,score]
                    # we pair them with frame numbers using the known birth frame
                    xy = []
                    for obs_i, bbox in enumerate(trk.history_observations):
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        xy.append((frame_born + obs_i, cx, cy))
                    self.suppressed_tracks.append({ #adding the detections that were too short to be a track
                        "id":         trk.id,
                        "hits":       trk.hits,
                        "frame_born": frame_born,
                        "frame_died": self.frame_count,
                        "xy":         xy,   # [(frame, cx, cy), ...]
                    })
                self.trackers.pop(i)

        self.detection_log.append((self.frame_count, n_dets, len(ret))) #now we are adding the number of detections

        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def relink(self) -> list:
        """
        Post-tracking re-link pass using behavioral split-and-compare.
        Call after all frames have been processed.

        Finds tracks whose behavioral profile is inconsistent with itself
        (first half vs second half differ more than relink_inconsistency_threshold),
        then tries swapping second segments between pairs of suspect tracks to
        improve consistency. The split point is the frame with the largest
        speed/acceleration change — a proxy for where an overlap-induced swap
        may have occurred.

        Returns
        -------
        list of (id_a, id_b, swap_frame) tuples.
        Each entry means: after `swap_frame`, observations in tracker id_a
        should be labelled id_b and vice versa.
        id_a, id_b are 1-based tracker IDs.
        """
        from .association import relink_tracklets
        return relink_tracklets(
            self.trackers,
            weights=self.relink_behavioral_weights,
            min_length=self.relink_min_length,
            inconsistency_threshold=self.relink_inconsistency_threshold,
            swap_threshold=self.relink_swap_threshold,
            confidence_weight=self.relink_confidence_weight,
            fps=self.fps,
        )

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        _zero_vel = np.array((0, 0))
        velocities = np.array([trk.velocity if trk.velocity is not None else _zero_vel for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        _dt = self.frame_count if self.delta_t < 0 else self.delta_t
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, _dt) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            cate_matrix = np.where(det_cates_left[:, np.newaxis] != trk_cates_left[np.newaxis, :], -1e6, 0.0)
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


