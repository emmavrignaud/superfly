"""
stitching.py

Tracklet stitching pipeline for fly tracking.

Background
----------
The tracker (OC-SORT) assigns a temporary numeric ID to each fly it detects.
When a fly is occluded, exits the frame, or is briefly missed by the detector,
the tracker loses it and assigns a new ID when it reappears. This produces
many short "tracklets" that actually belong to the same fly.

Stitching is the process of deciding which tracklets belong to the same fly
and merging them under a single identity.

Pipeline (called by stitch())
-------------------------------------
1. wide_to_long()         — reshape tracker output from wide CSV to long format
2. build_tracklets()      — summarise each ID's detections into a Tracklet object
3. build_cost_matrix()    — score every candidate A→B pair with link_score()
4. _solve_hungarian()     — find the globally cheapest 1-to-1 assignment
5. _map_chains_to_roots() — walk matched chains and assign each tracklet a root ID
6. Repeat 3-5 per vial, iteratively relaxing the gap limit, until the target
   fly count is reached or no further merges are possible.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
from src.features import add_kinematics, add_area_covered, add_path_tortuosity



# ---------------------------------------------------------------------------
# Converting wide format to long format
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

cfg_stitching = cfg['stitching']

def wide_to_long(df_wide, out_csv: Optional[str] = None):
    """
    Reshape a wide tracker DataFrame into long format.

    Input (wide):  one row per frame, one column per tracked ID,
                   each cell holds '(x, y)' or NaN.
    Output (long): columns [frame, id, x, y],
                   one row per (frame, id) observation.

    Parameters
    ----------
    out_csv : optional path to save the long-format CSV (e.g. OUTPUT_PATH/ocsort_tracks_long.csv)
    """
    id_cols = [c for c in df_wide.columns if c != "frame"]

    if not id_cols:
        return pd.DataFrame(columns=["frame", "orig_id", "x", "y"])

    # Set frame as the index so that stack() melts only the ID columns.
    # Before: columns are [frame, id1, id2, ...] with frame as index
    # After stack(): each (frame, id) combination becomes its own row,
    # and NaN cells are automatically dropped.
    stacked = df_wide.set_index("frame")[id_cols].stack().reset_index()
    stacked.columns = ["frame", "orig_id", "xy_raw"]
    # stacked now looks like:
    #   frame   id   xy_raw
    #   0       id1  (18.39, 311.59)
    #   0       id2  (283.23, 321.32)
    #   ...

    # Split the '(18.39, 311.59)' strings into two float columns.
    # 1. Strip parentheses:  '(18.39, 311.59)' → '18.39, 311.59'
    # 2. Split on comma:     '18.39, 311.59'   → ['18.39', '311.59']
    # 3. Cast to float
    coords = (
        stacked["xy_raw"]
        .str.strip("()")
        .str.split(",", expand=True)
        .astype(float)
    )
    stacked["x"] = coords[0]
    stacked["y"] = coords[1]

    # Drop the raw string column, sort for readability
    df_long = (
        stacked
        .drop(columns="xy_raw")
        .sort_values(["orig_id", "frame"])
        .reset_index(drop=True)
    )
    if out_csv is not None:
        df_long.to_csv(out_csv, index=False)
    return df_long


# ---------------------------------------------------------------------------
# Tracklet summaries
# ---------------------------------------------------------------------------



# Important Variable Explanations  
#
#   n_points              = len(group)
#   start_xy / end_xy     = (x, y) of first / last row
#   start_frame/end_frame = frame of first / last row
#
#   step_distance         = sqrt(dx² + dy²)
#   dt                    = frame.diff() / fps
#   velocity              = step_distance / dt
#   heading               = arctan2(dy, dx)
#   acceleration          = velocity.diff() / dt
#   turning_angle         = arctan2(sin(heading.diff()), cos(heading.diff()))
#   angular_velocity      = turning_angle / dt
#
#   velocities            = [velocity] per row (excluding first diff baseline)
#   directions            = [degrees(heading)] per row (excluding first)
#
#   n_large_displacements = count(velocity > 2 * median(velocity))
#   distance_traveled     = distance_traveled[-1]  (cumsum of step_distance)
#   mean_velocity         = mean(velocity)
#   median_velocity       = median(velocity)
#   mean_acceleration     = mean(|acceleration|)
#   mean_turning_angle    = mean(|turning_angle|)
#   mean_angular_velocity = mean(|angular_velocity|)
#   pause_fraction        = mean(velocity < pause_threshold)
#   tortuosity            = distance_traveled[-1] / ||end_xy - start_xy||
#   area_covered          = convex hull area of all (x, y) in tracklet
#   overall_direction     = degrees(arctan2(y[-1]-y[0], x[-1]-x[0]))
#   final_direction       = directions[-1]
#


@dataclass
class Tracklet:
    """
    A summary of one continuous detection sequence under a single tracker ID.

    The tracker assigns a temporary ID to each fly. When the fly is lost and
    redetected, a new ID is issued. A Tracklet captures everything we know about
    one such ID: where it started and ended, how it moved, and behavioural
    statistics that describe its movement style.

    These summaries are the inputs to link_score(), which decides whether two
    Tracklets plausibly belong to the same fly.
    """
    orig_id:               str                      # tracker-assigned ID (before stitching)
    start_frame:           int                      # first frame this ID was observed
    end_frame:             int                      # last frame this ID was observed
    start_xy:              Tuple[float, float]      # (x, y) position at start_frame
    end_xy:                Tuple[float, float]      # (x, y) position at end_frame
    num_frames:            int                      # total number of detections
    trajectory:            List[Tuple]              # full path: [(frame, x, y), ...]

    # Per-step profiles: one value per consecutive frame pair (first row excluded
    # because kinematics are undefined for a single detection).
    velocities:            List[Optional[float]]    # speed at each step (px/s)
    directions:            List[Optional[float]]    # heading at each step (degrees)

    # Motion summaries
    n_large_displacements: int                      # steps where velocity > 2× median velocity (burst detection)
    distance_traveled:     float                    # cumulative path length (px)
    mean_velocity:         float                    # mean speed across all steps (px/s)
    median_velocity:       float                    # median speed — more robust to bursts (px/s)
    mean_acceleration:     float                    # mean signed acceleration (px/s²)
    mean_turning_angle:    float                    # mean absolute heading change per step (degrees)
    mean_angular_velocity: float                    # mean absolute turning rate (degrees/s)

    # Shape of the journey
    tortuosity:            float                    # path length / straight-line displacement; 1.0 = perfectly straight
    area_covered:          float                    # convex hull area of all visited positions (px²)
    pause_fraction:        float                    # fraction of steps where velocity < pause_threshold

    # Direction
    overall_direction:     float                    # angle of the straight line from start to end (degrees)
    final_direction:       Optional[float]          # heading at the last recorded step (degrees); None if single-frame


def build_tracklets(long_df: pd.DataFrame) -> List[Tracklet]:
    """
    Convert a long-format DataFrame of tracker detections into a list of
    Tracklet objects — one per unique tracker ID.

    Runs kinematic feature extraction (via features.py) on the full dataframe,
    then groups by orig_id and computes per-tracklet summaries used later by
    link_score() to decide whether two tracklets belong to the same fly.

    Parameters
    ----------
    long_df : pd.DataFrame
        One row per detection. Required columns:
            frame    — frame number (int)
            orig_id  — tracker-assigned fly ID (str)
            x        — horizontal position in pixels (float)
            y        — vertical position in pixels (float)

    Returns
    -------
    List[Tracklet]
        One Tracklet per unique orig_id, sorted by orig_id.
        See the Tracklet dataclass for a full description of each field.
    """
    fps = cfg_stitching['fps']
    pause_threshold = cfg_stitching['pause_threshold']
    
    # Run feature extraction on the long dataframe    
    df = long_df.copy()
    df = add_kinematics(df, group_col="orig_id")
    df = add_area_covered(df, group_col="orig_id")
    df = add_path_tortuosity(df, group_col="orig_id")

    tracklets = []
    for fly_id, fly_group in df.groupby("orig_id", sort=False):
        # fly_id  — the tracker-assigned ID string for this tracklet
        # fly_group — all rows belonging to this ID, sorted chronologically
        fly_detections = fly_group.sort_values("frame")

        # --- Trajectory: full sequence of (frame, x, y) for this tracklet ---
        trajectory = [
            (int(row["frame"]), float(row["x"]), float(row["y"]))
            for _, row in fly_detections.iterrows()
        ]

        # --- Per-step velocity and heading profiles ---
        # add_kinematics fills the first row of each group with 0 (diff baseline),
        # so we skip it — it is not a real observation.
        
        # A step is the movement between two consecutive detections — frame N to frame N+1. 
        # So fly_steps is the list of those movements, one per consecutive frame pair.

        fly_steps = fly_detections.iloc[1:]
        velocities = fly_steps["velocity"].tolist()
        directions = np.degrees(fly_steps["heading"]).tolist()

        # --- Filter to finite velocities only ---
        # NaN and inf can appear for single-frame tracklets or numerical edge cases.
        # All velocity-based summaries below are computed on this filtered list.
        finite_velocities = [v for v in velocities if v is not None and np.isfinite(v)]

        # --- Burst detection ---
        # A step is a "large displacement" if the fly moved more than twice its own
        # median velocity in that step. This flags erratic or fast-climbing behaviour.
        if finite_velocities:
            median_velocity       = float(np.median(finite_velocities))
            n_large_displacements = sum(1 for v in finite_velocities if v > 2 * median_velocity)
        else:
            median_velocity       = 0.0
            n_large_displacements = 0

        # --- Spatial anchors: where the tracklet started and ended ---
        start_xy = (float(fly_detections.iloc[0]["x"]),  float(fly_detections.iloc[0]["y"]))
        end_xy   = (float(fly_detections.iloc[-1]["x"]), float(fly_detections.iloc[-1]["y"]))

        # --- Overall direction: angle of the straight line from start to end ---
        # This captures the dominant axis of movement, ignoring the path taken.
        overall_direction = np.degrees(np.arctan2(
            end_xy[1] - start_xy[1],
            end_xy[0] - start_xy[0],
        ))


        # --- Final direction: last recorded heading ---
        # Used later in link_score to extrapolate where the fly was headed
        # at the moment this tracklet ended.
        final_direction = directions[-1] if directions else None

        tracklets.append(Tracklet(
            orig_id               = str(fly_id),
            start_frame           = int(fly_detections.iloc[0]["frame"]),
            end_frame             = int(fly_detections.iloc[-1]["frame"]),
            start_xy              = start_xy,
            end_xy                = end_xy,
            num_frames            = int(len(fly_detections)),
            trajectory            = trajectory,
            velocities            = velocities,
            directions            = directions,
            n_large_displacements = n_large_displacements,
            # distance_traveled is a cumulative sum — the last row holds the total
            distance_traveled     = float(fly_detections["distance_traveled"].iloc[-1]),
            mean_velocity         = float(np.mean(finite_velocities)) if finite_velocities else 0.0,
            median_velocity       = median_velocity,
            # abs() because we care about magnitude of acceleration/turning, not sign
            mean_acceleration     = float(fly_steps["acceleration"].mean())           if len(fly_steps) > 0 else 0.0,
            mean_turning_angle    = float(np.degrees(fly_steps["turning_angle"].abs().mean()))    if len(fly_steps) > 0 else 0.0,
            mean_angular_velocity = float(np.degrees(fly_steps["angular_velocity"].abs().mean())) if len(fly_steps) > 0 else 0.0,
            # tortuosity and area_covered are tracklet-level constants; any row holds the same value
            tortuosity            = float(fly_detections["tortuosity"].iloc[0]),
            area_covered          = float(fly_detections["area_covered"].iloc[0]),
            # pause_fraction: proportion of the fly_detections where the fly was effectively stationary
            pause_fraction        = float((fly_steps["velocity"] < pause_threshold).mean()) if len(fly_steps) > 0 else 0.0,
            overall_direction     = overall_direction,
            final_direction       = final_direction,
        ))

    tracklets.sort(key=lambda t: t.orig_id)
    return tracklets


# ---------------------------------------------------------------------------
# Wall detection helpers
# ---------------------------------------------------------------------------

def _near_wall(
    xy:            Tuple[float, float],
    vial_roi:      Tuple[int, int, int, int],
    edge_fraction: float = 0.10,
) -> Tuple[bool, bool]:
    """
    Check whether a position is close to the horizontal or vertical walls of a vial.

    When a fly is near a wall, its direction of travel after the gap may be
    reflected relative to its direction before — we need to account for this
    in link_score rather than penalising a legitimate bounce as a direction mismatch.

    'Near' is defined as within (edge_fraction × vial dimension) of that wall.
    With edge_fraction=0.10, a fly within 10% of the vial width from either
    side is considered near a horizontal wall.

    Returns
    -------
    (near_h, near_v)
        near_h : True if the fly is near the left or right wall
                 (x-component of velocity may flip on the next bounce)
        near_v : True if the fly is near the top or bottom wall
                 (y-component of velocity may flip on the next bounce)
    """
    x0, y0, x1, y1 = vial_roi
    w = x1 - x0
    h = y1 - y0
    x, y = xy
    near_h = (x - x0) < edge_fraction * w or (x1 - x) < edge_fraction * w
    near_v = (y - y0) < edge_fraction * h or (y1 - y) < edge_fraction * h
    return near_h, near_v
 
 
def _mirror_angle(angle_deg: float, near_h: bool, near_v: bool) -> float:
    """
    Reflect a heading angle based on which vial wall the fly is approaching.

    A fly moving toward a wall will bounce: its velocity component perpendicular
    to the wall reverses, while the parallel component is unchanged. This maps
    to flipping dx (left/right wall) or dy (top/bottom wall) in the heading vector.

    Used in link_score to predict the expected post-gap direction when a fly
    was near a wall at the end of tracklet A.
    """
    rad = np.radians(angle_deg)
    dx  = np.cos(rad)
    dy  = np.sin(rad)
    if near_h:
        dx = -dx
    if near_v:
        dy = -dy
    return float(np.degrees(np.arctan2(dy, dx)))
 
 
def _angle_diff(a: float, b: float) -> float:
    """Smallest angle between two directions in degrees, result in [0, 180]."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)
 
 
def _find_vial(
    xy:        Tuple[float, float],
    vial_rois: Dict[str, Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Return the ROI tuple for whichever vial contains xy, or None."""
    x, y = xy
    for roi in vial_rois.values():
        x0, y0, x1, y1 = roi
        if x0 <= x <= x1 and y0 <= y <= y1:
            return roi
    return None
 
 
# ---------------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------------

def simulate_position(
    start_xy:     Tuple[float, float],
    direction:    float,
    velocity:     float,
    acceleration: float,
    gap:          int,
    fps:          float,
    vial_roi:     Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """
    Simulate a fly's position frame by frame over `gap` frames.

    At each frame the fly advances (velocity / fps) pixels along its current
    direction. Velocity is then updated by (acceleration / fps). When the
    projected position exits the vial ROI, the perpendicular velocity
    component is reflected and the position is corrected symmetrically
    (wall bounce).

    Parameters
    ----------
    start_xy     : (x, y) starting position in pixels
    direction    : initial heading in degrees
    velocity     : initial speed in px/s
    acceleration : signed speed change per second in px/s²
    gap          : number of frames to simulate
    fps          : frames per second
    vial_roi     : (x0, y0, x1, y1) pixel bounding box of the vial

    Returns
    -------
    (x, y) predicted position after `gap` frames
    """
    x0, y0, x1, y1 = vial_roi
    x,  y           = float(start_xy[0]), float(start_xy[1])

    rad = np.radians(direction)
    vx  = velocity * np.cos(rad)   # px/s
    vy  = velocity * np.sin(rad)   # px/s

    for _ in range(gap):
        nx = x + vx / fps
        ny = y + vy / fps

        # Reflect off left / right walls
        if nx < x0:
            vx = -vx
            nx = x0 + (x0 - nx)
        elif nx > x1:
            vx = -vx
            nx = x1 - (nx - x1)

        # Reflect off top / bottom walls
        if ny < y0:
            vy = -vy
            ny = y0 + (y0 - ny)
        elif ny > y1:
            vy = -vy
            ny = y1 - (ny - y1)

        x, y = nx, ny

        # Update speed magnitude along current direction; clamp to zero
        speed     = np.sqrt(vx * vx + vy * vy)
        new_speed = max(speed + acceleration / fps, 0.0)
        if speed > 1e-9:
            vx = vx * new_speed / speed
            vy = vy * new_speed / speed

    return x, y


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_feature_scales(
    tracklets:      List["Tracklet"],
    min_points:     int = 10,
) -> Dict[str, float]:
    """
    Compute the population standard deviation of each behavioural dissimilarity
    feature across tracklets.

    Only tracklets with n_points >= min_points are used to compute the scales.
    Short fragments have unreliable kinematics and would distort the std used
    to normalise dissimilarity scores for all pairs. If no tracklets pass the
    filter, all tracklets are used as a fallback.

    Non-finite values (e.g. tortuosity for near-stationary tracklets) are
    excluded before computing std. The result is floored at 1e-6 to avoid
    division by zero.
    """
    reliable = [t for t in tracklets if t.num_frames >= min_points]
    if not reliable:
        reliable = tracklets  # fallback: use all if none qualify

    def _std(vals: List[float]) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return max(float(np.std(finite)) if len(finite) > 1 else 1.0, 1e-6)

    return {
        "median_velocity":       _std([t.median_velocity             for t in reliable]),
        "pause_fraction":        _std([t.pause_fraction              for t in reliable]),
        "tortuosity":            _std([t.tortuosity                  for t in reliable]),
        "mean_turning_angle":    _std([t.mean_turning_angle          for t in reliable]),
        "mean_angular_velocity": _std([t.mean_angular_velocity       for t in reliable]),
        "mean_acceleration":     _std([t.mean_acceleration           for t in reliable]),
        "n_large_displacements": _std([float(t.n_large_displacements) for t in reliable]),
    }


def link_score(
    A:         "Tracklet",
    B:         "Tracklet",
    vial_rois: Dict[str, Tuple[int, int, int, int]],
    tracklets: List["Tracklet"],
    weights:   Dict[str, Dict],
) -> float:
    """
    Compute the cost of linking tracklet A into tracklet B (lower = more plausible).

    This is the core of the stitching pipeline. It answers the question:
    "Given that tracklet A ended and tracklet B started after a gap, how likely
    is it that they belong to the same fly?"

    Three additive terms, each capturing a different aspect of plausibility:

    Term 1 — Extrapolated position error (px)
        Simulate A's trajectory forward frame by frame (using A's final heading,
        speed, and mean acceleration, with wall bounces inside the vial).
        Measure the pixel distance between the predicted landing position and
        where B actually starts. Small error = plausible same-fly link.

    Term 2 — Direction agreement [0, 1]
        Weighted average of two sub-components:
          a. A's final heading (wall-reflected if near a vial edge) vs. the
             direction of the gap vector from A's end to B's start.
          b. A's overall direction vs. B's overall direction.
        Each is normalised to [0, 1] by dividing the angular difference by 180°.
        0 = perfectly aligned, 1 = opposite directions.

    Term 3 — Behavioral dissimilarity
        Weighted Euclidean distance in z-score space between A's and B's
        kinematic feature vectors (velocity, acceleration, tortuosity, etc.).
        Features are z-scored against the population std so that all are
        dimensionless and comparable. Short tracklets get down-weighted via a
        sigmoid: w = num_frames / (num_frames + k).

    Final score = link_score_weights['extrap']     * term1
               + link_score_weights['direction']   * term2
               + link_score_weights['behavioral']  * term3
    (weights from config.yaml)

    Parameters
    ----------
    A, B       : tracklets to link; A must precede B in time
    vial_rois  : {vial_id: (x0, y0, x1, y1)} bounding boxes for wall detection
    tracklets  : full population of tracklets for this vial; used to compute
                 per-feature population stds for z-scoring the behavioral term

    All tuning parameters (edge_fraction, min_points_for_scale,
    link_score_weights, direction_weights, behavioral_weights) are read from config.yaml.
    """
    gap            = B.start_frame - A.end_frame   # frames between the end of A and start of B
    edge_fraction  = cfg_stitching['edge_fraction']
    min_points     = cfg_stitching['min_points_for_scale']
    feature_scales = _compute_feature_scales(tracklets)

    # Use A's median velocity as the expected speed.
    # If A was stationary or has no velocity estimate, mv=0 and simulate_position
    # stays at A.end_xy, so extrap_term_error degrades to raw Euclidean distance.
    mv  = A.median_velocity
    fps = cfg_stitching['fps']

    dx = B.start_xy[0] - A.end_xy[0]   # horizontal gap between endpoints
    dy = B.start_xy[1] - A.end_xy[1]   # vertical gap between endpoints

    # ------------------------------------------------------------------
    # Term 1 — EXTRAPOLATED POSITION ERROR  (px)
    #
    # Simulate A's trajectory frame by frame — respecting vial wall bounces —
    # to get the specific position where B should start if it is the same fly.
    # extrap_term_error is the raw pixel distance between that prediction and where
    # B actually starts.
    # ------------------------------------------------------------------
    vial_roi = _find_vial(A.end_xy, vial_rois)
    if A.final_direction is not None and vial_roi is not None:
        ex, ey = simulate_position(
            start_xy     = A.end_xy,
            direction    = A.final_direction,
            velocity     = mv,
            acceleration = A.mean_acceleration,
            gap          = gap,
            fps          = fps,
            vial_roi     = vial_roi,
        )
    else:
        ex, ey = A.end_xy

    extrap_term_error = float(np.sqrt((B.start_xy[0] - ex) ** 2 + (B.start_xy[1] - ey) ** 2))

    # ------------------------------------------------------------------
    # Term 2 — DIRECTION AGREEMENT  [0, 1]
    #
    # Weighted average of two sub-components, each normalised to [0, 1]:
    #   a. A's final heading (wall-reflected if near a vial edge) vs. the
    #      actual gap vector from A's endpoint to B's start.
    #      Weight: direction_weights.heading_vs_gap  (config)
    #   b. A's overall trajectory direction vs. B's overall direction —
    #      a softer, longer-range consistency check.
    #      Weight: direction_weights.overall_vs_overall  (config)
    # When A has no final_direction, only sub-component b is used.
    # ------------------------------------------------------------------
    gap_direction = float(np.degrees(np.arctan2(dy, dx)))
    dir_overall   = _angle_diff(A.overall_direction, B.overall_direction) / 180.0

    if A.final_direction is not None:
        if vial_roi is not None:
            near_h, near_v = _near_wall(A.end_xy, vial_roi, edge_fraction)
            expected_dir   = (
                _mirror_angle(A.final_direction, near_h, near_v)
                if (near_h or near_v)
                else A.final_direction
            )
        else:
            expected_dir = A.final_direction
        dir_heading = _angle_diff(gap_direction, expected_dir) / 180.0
        w_hg        = weights['direction_weights']['heading_vs_gap']
        w_oo        = weights['direction_weights']['overall_vs_overall']
        direction_term = float((w_hg * dir_heading + w_oo * dir_overall) / (1+ w_hg + w_oo))
    else:
        direction_term = float(dir_overall)

    # ------------------------------------------------------------------
    # Term 3 — BEHAVIORAL DISSIMILARITY
    #
    # Flies have individual movement signatures: one might be a fast
    # climber with low tortuosity, another slow and meandering.  If A and B
    # are the same fly, their behavioral statistics should be similar.
    #
    # Each feature difference is z-scored against the population std
    # (from _compute_feature_scales), making all features dimensionless.
    # The score is a weighted Euclidean distance in z-score space:
    #
    #   behavioral_score = sqrt( sum_i( bw_i * z_i² ) )
    #
    # where bw_i are per-feature weights from config (behavioral_weights).
    # Using L2 rather than L1 means a single large deviation (e.g. tortuosity
    # differs by 3σ) is not diluted by features that agree perfectly.
    #
    # Short tracklets (few frames) have noisy feature estimates — a
    # 3-frame tracklet's "median velocity" is meaningless.  We down-weight
    # the behavioral score using a sigmoid in tracklet length:
    #
    #   w = num_frames / (num_frames + k)
    #
    # At num_frames = k  →  w = 0.5  (half weight)
    # At num_frames >> k →  w → 1.0  (full weight)
    # At num_frames << k →  w → 0.0  (ignored)
    #
    # We use min(wA, wB): the weaker tracklet limits behavioral confidence
    # for the whole pair.
    # ------------------------------------------------------------------
    behavioral_weights = weights['behavioral_weights']

    def _dissim(a_val: float, b_val: float, key: str) -> float:
        # |difference| normalised by population std → dimensionless z-score
        return abs(a_val - b_val) / feature_scales.get(key, 1.0)

    # Each entry: (z_score, per-feature weight)
    dissim_pairs: List[Tuple[float, float]] = [
        (_dissim(A.median_velocity,              B.median_velocity,              "median_velocity"),       behavioral_weights['median_velocity']),
        (_dissim(A.pause_fraction,               B.pause_fraction,               "pause_fraction"),        behavioral_weights['pause_fraction']),
        (_dissim(A.mean_turning_angle,           B.mean_turning_angle,           "mean_turning_angle"),    behavioral_weights['mean_turning_angle']),
        (_dissim(A.mean_angular_velocity,        B.mean_angular_velocity,        "mean_angular_velocity"), behavioral_weights['mean_angular_velocity']),
        (_dissim(A.mean_acceleration,            B.mean_acceleration,            "mean_acceleration"),     behavioral_weights['mean_acceleration']),
        (_dissim(float(A.n_large_displacements), float(B.n_large_displacements), "n_large_displacements"), behavioral_weights['n_large_displacements']),
    ]

    # Tortuosity is undefined (nan/inf) when a fly barely moved — net
    # displacement ≈ 0 makes the ratio blow up.  Skip rather than propagate nan.
    if np.isfinite(A.tortuosity) and np.isfinite(B.tortuosity):
        dissim_pairs.append((_dissim(A.tortuosity, B.tortuosity, "tortuosity"), behavioral_weights['tortuosity']))

    k  = max(min_points, 1)
    wA = A.num_frames / (A.num_frames + k)   # sigmoid weight for A
    wB = B.num_frames / (B.num_frames + k)   # sigmoid weight for B
    behavioral_score = min(wA, wB) * float(
        np.sqrt(
            sum(bw * z * z for z, bw in dissim_pairs) # basically multiply each dissimilarity value by its weight in config
            ) 
        ) # take the sum, and then sqrt to get magnitude.

    # ------------------------------------------------------------------
    # Final score — weighted sum of three terms (weights from config).
    #   extrap     : wall-aware predicted position error (px)
    #   direction  : angular agreement (final heading + overall direction)
    #   behavioral : kinematic profile similarity
    # ------------------------------------------------------------------
    link_score_weights = weights['link_score_weights']
    return (link_score_weights['extrap']     * extrap_term_error
          + link_score_weights['direction']  * direction_term
          + link_score_weights['behavioral'] * behavioral_score)
 
 
def build_cost_matrix(
    tracklets:        List[Tracklet],
    vial_rois:        Dict[str, Tuple[int, int, int, int]],
    weights:          Dict[str, Dict],
    chain_end_frames: Optional[Dict[str, int]] = None,
    debug_path:       Optional[str] = None,
) -> np.ndarray:
    """
    Build an N×N cost matrix over a list of tracklets.

    Every ordered pair (A, B) where A ends before B starts gets a
    link_score; all other cells are BIG (1e9). The three cost terms
    (extrapolated position error, direction agreement, behavioural
    dissimilarity) grow naturally with gap size and mismatch — no
    hard gap or score cap is applied here.

    chain_end_frames : if provided, maps each root orig_id to the maximum
        end_frame across all tracklets currently in its chain.  Used in
        round 2+ so the overlap check reflects the full chain, not just
        the root tracklet's own end_frame.

    If debug_path is set, saves the matrix as a CSV (BIG cells → NaN,
    rows/cols labelled by orig_id) for inspection.
    """
    BIG = 1e9
    n   = len(tracklets)
    C   = np.full((n, n), BIG, dtype=float)

    for i, A in enumerate(tracklets):
        for j, B in enumerate(tracklets):
            a_chain_end = chain_end_frames.get(A.orig_id, A.end_frame) if chain_end_frames else A.end_frame
            if i == j or a_chain_end >= B.start_frame:
                continue
            if _find_vial(A.start_xy, vial_rois) != _find_vial(B.start_xy, vial_rois):
                continue  # never link tracklets across vials
            C[i, j] = link_score(A, B, vial_rois, tracklets, weights)

    if debug_path is not None:
        labels   = [t.orig_id for t in tracklets]
        debug_df = pd.DataFrame(C, index=labels, columns=labels)
        debug_df = debug_df.replace(BIG, float("nan"))
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        debug_df.to_csv(debug_path)

    return C
 
 
# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
 
def _solve_hungarian(
    cost_matrix: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """
    Find the minimum-cost 1-to-1 assignment over the cost matrix using the
    Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Each returned match (i, j, cost) means: tracklet i should be linked to
    tracklet j at the given cost. Only matches with cost < BIG (i.e. valid
    candidate pairs) are returned.

    Falls back to a greedy sort-by-cost assignment if scipy raises an exception.
    """
    BIG = 1e9
    C   = cost_matrix
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        return [
            (int(i), int(j), float(C[i, j]))
            for i, j in zip(r, c)
            if C[i, j] < BIG / 10
        ]
    except Exception:
        matches             = []
        used_r, used_c      = set(), set()
        edges               = [(int(i), int(j), float(C[i, j]))
                               for i, j in np.argwhere(C < BIG / 10)]
        edges.sort(key=lambda t: t[2])
        for i, j, cost in edges:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            matches.append((i, j, cost))
        return matches
 
 
def _map_chains_to_roots(
    tracklets: List[Tracklet],
    matches:   List[Tuple[int, int, float]],
) -> Dict[str, str]:
    """
    Convert a list of pairwise matches into a mapping from each tracklet's
    orig_id to the orig_id of the root of its stitched chain.

    Matches form directed chains: if T1→T2 and T2→T4 are both matched,
    then T1, T2, and T4 all belong to the same fly — and T1 (the chain root,
    i.e. the earliest tracklet with no predecessor) is chosen as the
    representative ID for all three.

    Example: matches [(T1,T2), (T2,T4)] produces
             {T1: T1, T2: T1, T4: T1}
    """
    succ: Dict[int, int] = {}
    pred: Dict[int, int] = {}
 
    for i, j, _ in matches:
        if i in succ or j in pred:
            continue
        succ[i] = j
        pred[j] = i
 
    stitched_root: Dict[int, str] = {}
 
    for idx in range(len(tracklets)):
        if idx in pred:
            continue
        root, cur = tracklets[idx].orig_id, idx
        while True:
            stitched_root[cur] = root
            if cur not in succ:
                break
            cur = succ[cur]
 
    for idx in range(len(tracklets)):
        if idx not in stitched_root:
            stitched_root[idx] = tracklets[idx].orig_id
    return {tracklets[i].orig_id: stitched_root[i] for i in range(len(tracklets))}



# ---------------------------------------------------------------------------
# Per-vial iterative stitching
# ---------------------------------------------------------------------------
 
def _assign_to_vial(
    start_xy:  Tuple[float, float],
    vial_rois: Dict[str, Tuple[int, int, int, int]],
) -> Optional[str]:
    """Return the vial key whose ROI contains start_xy, or None."""
    x, y = start_xy
    for vial_id, (x0, y0, x1, y1) in vial_rois.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return vial_id
    return None
 
 
def _count_active_ids(tracklets: List[Tracklet], mapping: Dict[str, str]) -> int:
    """Count the number of distinct fly identities remaining after applying mapping."""
    return len({mapping.get(t.orig_id, t.orig_id) for t in tracklets})
 
 
def _select_prefix(
    matches:  List[Tuple[int, int, float]],
    n_roots:  int,
    expected: int,
    w_under:  float,
    w_over:   float,
) -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Pick how many of Hungarian's cost-sorted matches to accept this round.

    Each accepted match merges two chain roots into one, so accepting k matches
    leaves (n_roots - k) distinct IDs. We score every prefix k = 0..len(matches)
    by

        total(k) = sum(link_scores of k cheapest matches)
                 + w_under · max(n_roots - k - expected, 0)   # under-merged
                 + w_over  · max(expected - (n_roots - k), 0) # over-merged

    and return the prefix minimising total(k). w_under > w_over asymmetrically
    penalises leaving flies fragmented more than collapsing two flies together.

    Returns
    -------
    (accepted_matches, k_star)
        accepted_matches : the top-k_star matches by link_score
        k_star           : number of accepted matches (0 allowed)
    """
    ordered = sorted(matches, key=lambda m: m[2])

    def _penalty(num_flies: int) -> float:
        dev = num_flies - expected
        return w_under * max(dev, 0) + w_over * max(-dev, 0)

    running_sum = 0.0
    best_k      = 0
    best_total  = _penalty(n_roots)          # k = 0 baseline
    for k, (_, _, link_score_val) in enumerate(ordered, start=1):
        running_sum += link_score_val
        total = running_sum + _penalty(n_roots - k)
        if total < best_total:
            best_total = total
            best_k     = k

    return ordered[:best_k], best_k


def _run_assignment_round(
    tracklets:         List[Tracklet],
    frozen_mapping:    Dict[str, str],
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    weights:           Dict[str, Dict],
    expected:          int,
    w_under:           float,
    w_over:            float,
    debug_path:        Optional[str] = None,
) -> Tuple[Dict[str, str], int]:
    """
    Run one round of Hungarian assignment over currently-unmerged tracklets.

    The stitching loop is iterative: each round freezes the merges it finds,
    then the next round operates on the reduced set of chain roots. This
    allows chains longer than two tracklets to be resolved across rounds.
    Merges from previous rounds are never revisited or undone.

    Match acceptance is count-aware: among Hungarian's cost-sorted matches,
    we accept the prefix k that minimises Σ link_scores + asymmetric deviance
    penalty w.r.t. `expected`. See _select_prefix for the scoring rule.

    Parameters
    ----------
    tracklets      : all tracklets for this vial
    frozen_mapping : {orig_id: root_id} from all previous rounds; entries
                     here are preserved unchanged in the output
    vial_rois      : {vial_id: (x0, y0, x1, y1)} for wall detection in link_score
    expected       : target number of chain roots (flies) for the penalty
    w_under        : per-unit penalty when num_flies > expected (under-merging)
    w_over         : per-unit penalty when num_flies < expected (over-merging)
    debug_path     : if set, save the cost matrix as a CSV to this path

    Returns
    -------
    (updated_mapping, n_accepted)
        updated_mapping : {orig_id: root_id} with new merges folded in
        n_accepted      : number of matches accepted this round (0 → no progress)
    """
    roots = [t for t in tracklets
             if frozen_mapping.get(t.orig_id, t.orig_id) == t.orig_id]

    if len(roots) <= 1:
        return frozen_mapping, 0

    # Effective end_frame for each root = max end_frame across all chain members.
    # Prevents round 2+ from linking a root to a tracklet that overlaps with a
    # non-root tail of the same chain (the root's own end_frame would be stale).
    chain_end_frames: Dict[str, int] = {}
    for t in tracklets:
        root_id = frozen_mapping.get(t.orig_id, t.orig_id)
        chain_end_frames[root_id] = max(chain_end_frames.get(root_id, 0), t.end_frame)

    C = build_cost_matrix(roots, vial_rois, weights, chain_end_frames=chain_end_frames, debug_path=debug_path)
    matches = _solve_hungarian(C)

    if not matches:
        return frozen_mapping, 0

    matches, n_accepted = _select_prefix(
        matches  = matches,
        n_roots  = len(roots),
        expected = expected,
        w_under  = w_under,
        w_over   = w_over,
    )

    if not matches:
        return frozen_mapping, 0

    new_mapping = _map_chains_to_roots(roots, matches)

    # Propagate new merges into the frozen mapping.
    updated = dict(frozen_mapping)
    for orig_id, frozen_root in frozen_mapping.items():
        if frozen_root in new_mapping:
            updated[orig_id] = new_mapping[frozen_root]
    for orig_id, new_root in new_mapping.items():
        if orig_id not in updated:
            updated[orig_id] = new_root

    return updated, n_accepted


# ---------------------------------------------------------------------------
# Duplicate reporting
# ---------------------------------------------------------------------------

def _report_duplicates(
    out: pd.DataFrame,
    stitched_id_to_vial: Optional[Dict[str, str]] = None,
) -> None:
    """
    Print a succinct summary of (frame, stitched_id) duplicates in the output.
    Expected to be 0 after the chain_end_frames fix.
    """
    dupes = out[out.duplicated(subset=["frame", "stitched_id"], keep=False)]
    if dupes.empty:
        print("  Duplicate check: 0 duplicate (frame, stitched_id) pairs — clean.")
        return

    summary = dupes.groupby("stitched_id")["frame"].nunique().sort_values(ascending=False)
    total_frames = int(dupes.drop_duplicates(subset=["frame", "stitched_id"]).shape[0])
    print(f"  Duplicate check: {len(summary)} stitched_id(s) had duplicates across {total_frames} frame(s)")
    for sid, n_frames in summary.items():
        vial_str = f" ({stitched_id_to_vial[sid]})" if stitched_id_to_vial and sid in stitched_id_to_vial else ""
        print(f"    {sid}{vial_str}: {n_frames} affected frame(s)")


def _should_continue(
    stop_mode:      str,
    round_num:      int,
    n_active:       int,
    cap:            int,
    max_rounds:     int,
) -> bool:
    """Return True iff another assignment round should be run."""
    if stop_mode == "cap":
        return n_active > cap
    if stop_mode == "fixed":
        return round_num <= max_rounds
    if stop_mode == "converge":
        return True   # convergence is signalled by n_accepted==0 inside the loop
    raise ValueError(
        f"Unknown stop_mode {stop_mode!r}. Expected 'converge', 'cap', or 'fixed'."
    )


def stitch_per_vial(
    long_df:        pd.DataFrame,
    vial_rois:      Dict[str, Tuple[int, int, int, int]],
    tracklets:      List[Tracklet],
    weights:        Dict[str, Dict],
    output_dir:     Optional[str] = None,
    vial_count_cap: Optional[int] = None,
    w_under:        Optional[float] = None,
    w_over:         Optional[float] = None,
    stop_mode:      Optional[str] = None,
    max_rounds:     Optional[int] = None,
) -> pd.DataFrame:
    """
    Top-level stitching function. Merges tracklets into fly identities, per vial.

    Strategy
    --------
    Tracklets are processed independently per vial — a fly in vial 1 can never
    be linked to a tracklet in vial 2. Within each vial, Hungarian assignment
    runs iteratively: each round freezes the merges it finds, then the next
    round operates on the reduced set of chain roots. This allows chains longer
    than two tracklets to be resolved across rounds. Inside each round, matches
    are accepted as a cost-ordered prefix minimising
    Σ link_scores + w_under·max(n_flies-expected,0) + w_over·max(expected-n_flies,0).

    Stop condition depends on cfg_stitching['stop_mode']:
      - 'converge' (default): stop when a round accepts zero matches.
      - 'cap'               : stop when active IDs per vial ≤ vial_count_cap.
      - 'fixed'             : run exactly max_rounds rounds.

    Config (read from config.yaml unless overridden)
    ------------------------------------------------
    expected_per_vial, w_under, w_over, vial_count_cap, stop_mode, max_rounds
    """

    # 1. Assign tracklets to vials by start_xy
    vial_tracklets: Dict[str, List[Tracklet]] = {v: [] for v in vial_rois}
    for t in tracklets:
        vial = _assign_to_vial(t.start_xy, vial_rois)
        if vial is not None:
            vial_tracklets[vial].append(t)

    # 2. Per-vial iterative stitching
    global_mapping: Dict[str, str] = {}

    cap        = vial_count_cap if vial_count_cap is not None else cfg_stitching['vial_count_cap']
    expected   = cfg_stitching['expected_per_vial']
    w_under    = w_under    if w_under    is not None else cfg_stitching['w_under']
    w_over     = w_over     if w_over     is not None else cfg_stitching['w_over']
    stop_mode  = stop_mode  if stop_mode  is not None else cfg_stitching['stop_mode']
    max_rounds = max_rounds if max_rounds is not None else cfg_stitching['max_rounds']

    for vial_id in sorted(vial_rois.keys()):
        vt = vial_tracklets[vial_id]
        if not vt:
            continue

        mapping   = {t.orig_id: t.orig_id for t in vt}
        round_num = 1

        while _should_continue(stop_mode, round_num,
                               _count_active_ids(vt, mapping), cap, max_rounds):
            n_before = _count_active_ids(vt, mapping)

            debug_path = (
                os.path.join(output_dir, "debug", f"cost_matrix_{vial_id}_round{round_num}.csv")
                if output_dir is not None else None
            )
            mapping, n_accepted = _run_assignment_round(
                tracklets      = vt,
                frozen_mapping = mapping,
                vial_rois      = vial_rois,
                weights        = weights,
                expected       = expected,
                w_under        = w_under,
                w_over         = w_over,
                debug_path     = debug_path,
            )

            n_after = _count_active_ids(vt, mapping)

            print(
                f"  {vial_id} round {round_num}: "
                f"{n_before} -> {n_after} IDs "
                f"(accepted {n_accepted}, mode={stop_mode}, expected={expected})"
            )

            if stop_mode == "converge" and n_accepted == 0:
                break
            if n_after == n_before and stop_mode != "fixed":
                # No progress under cap mode either — abort to avoid infinite loop.
                print(f"  {vial_id}: stuck at {n_after} IDs, stopping")
                break

            round_num += 1

        global_mapping.update(mapping)
    # 3. Apply stitched_id
    out = long_df.copy()
    out["stitched_id"] = out["orig_id"].map(global_mapping).fillna(out["orig_id"])

    # Build vial lookup for duplicate report
    stitched_id_to_vial: Dict[str, str] = {}
    for vial_id, vt in vial_tracklets.items():
        for t in vt:
            root = global_mapping.get(t.orig_id, t.orig_id)
            stitched_id_to_vial.setdefault(root, vial_id)

    _report_duplicates(out, stitched_id_to_vial)
    return out


# ---------------------------------------------------------------------------
# General (whole-video) stitching
# ---------------------------------------------------------------------------

def stitch_general(
    long_df:    pd.DataFrame,
    vial_rois:  Dict[str, Tuple[int, int, int, int]],
    tracklets:  List[Tracklet],
    weights:    Dict[str, Dict],
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Stitch tracklets across the full video in one shared assignment loop.

    Unlike stitch_per_vial, all vials are processed together — the Hungarian
    solver sees a single cost matrix over all tracklets. Cross-vial links are
    still forbidden (enforced in build_cost_matrix). The loop stops when the
    total number of active identities across all vials reaches general_count_cap
    or no further merges are possible.

    Parameters
    ----------
    long_df    : long-format dataframe with columns (frame, orig_id, x, y)
    vial_rois  : {vial_id: (x0, y0, x1, y1)} bounding boxes from roi.py
    tracklets  : Tracklet objects from build_tracklets()
    output_dir : if set, cost matrices are saved to output_dir/debug/ as CSVs
    """
    cap        = cfg_stitching['general_count_cap']
    expected   = cfg_stitching['expected_per_vial'] * len(vial_rois)
    w_under    = cfg_stitching['w_under']
    w_over     = cfg_stitching['w_over']
    stop_mode  = cfg_stitching['stop_mode']
    max_rounds = cfg_stitching['max_rounds']

    mapping   = {t.orig_id: t.orig_id for t in tracklets}
    round_num = 1

    while _should_continue(stop_mode, round_num,
                           _count_active_ids(tracklets, mapping), cap, max_rounds):
        n_before = _count_active_ids(tracklets, mapping)

        debug_path = (
            os.path.join(output_dir, "debug", f"cost_matrix_global_round{round_num}.csv")
            if output_dir is not None else None
        )
        mapping, n_accepted = _run_assignment_round(
            tracklets      = tracklets,
            frozen_mapping = mapping,
            vial_rois      = vial_rois,
            weights        = weights,
            expected       = expected,
            w_under        = w_under,
            w_over         = w_over,
            debug_path     = debug_path,
        )

        n_after = _count_active_ids(tracklets, mapping)

        print(
            f"  global round {round_num}: "
            f"{n_before} -> {n_after} IDs "
            f"(accepted {n_accepted}, mode={stop_mode}, expected={expected})"
        )

        if stop_mode == "converge" and n_accepted == 0:
            break
        if n_after == n_before and stop_mode != "fixed":
            print(f"  global: stuck at {n_after} IDs, stopping")
            break

        round_num += 1

    out = long_df.copy()
    out["stitched_id"] = out["orig_id"].map(mapping).fillna(out["orig_id"])

    stitched_id_to_vial: Dict[str, str] = {}
    for t in tracklets:
        root = mapping.get(t.orig_id, t.orig_id)
        vial = _assign_to_vial(t.start_xy, vial_rois)
        if vial:
            stitched_id_to_vial.setdefault(root, vial)

    _report_duplicates(out, stitched_id_to_vial)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def stitch(
    long_df:        pd.DataFrame,
    vial_rois:      Dict[str, Tuple[int, int, int, int]],
    tracklets:      List[Tracklet],
    output_dir:     Optional[str] = None,
    weights:        Optional[Dict[str, Dict]] = None,
    vial_count_cap: Optional[int] = None,
    w_under:        Optional[float] = None,
    w_over:         Optional[float] = None,
    stop_mode:      Optional[str] = None,
    max_rounds:     Optional[int] = None,
) -> pd.DataFrame:
    """
    Dispatch to stitch_per_vial or stitch_general based on
    cfg_stitching['stitching_mode'].

    Parameters
    ----------
    long_df    : long-format dataframe with columns (frame, orig_id, x, y)
    vial_rois  : {vial_id: (x0, y0, x1, y1)} bounding boxes from roi.py
    tracklets  : Tracklet objects from build_tracklets()
    output_dir : if set, debug cost matrices are written here
    weights    : override for link_score_weights, direction_weights, and
                 behavioral_weights; reads from config.yaml when None
    w_under, w_over, stop_mode, max_rounds, vial_count_cap :
        optional overrides for the corresponding config.yaml keys. Passed
        through only to stitch_per_vial; stitch_general reads from config.
    """
    if weights is None:
        weights = {
            'link_score_weights': cfg_stitching['link_score_weights'],
            'direction_weights':  cfg_stitching['direction_weights'],
            'behavioral_weights': cfg_stitching['behavioral_weights'],
        }
    mode = cfg_stitching['stitching_mode']
    if mode == 'per_vial':
        return stitch_per_vial(
            long_df, vial_rois, tracklets, weights, output_dir,
            vial_count_cap = vial_count_cap,
            w_under        = w_under,
            w_over         = w_over,
            stop_mode      = stop_mode,
            max_rounds     = max_rounds,
        )
    elif mode == 'general':
        return stitch_general(long_df, vial_rois, tracklets, weights, output_dir)
    else:
        raise ValueError(
            f"Unknown stitching_mode {mode!r} in config.yaml. "
            f"Expected 'per_vial' or 'general'."
        )