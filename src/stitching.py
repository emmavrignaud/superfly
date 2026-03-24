"""
stitching.py

Full tracklet stitching pipeline for fly tracking.

Two stitching modes are available:

stitch_per_vial()  — preferred, vial-aware iterative stitcher
  wide CSV -> long format -> Tracklet summaries -> per-vial cost matrix
  (extrapolated position + 3-way direction + behavioural dissimilarity,
  all data-normalised) -> iterative 1-to-1 Hungarian assignment per vial
  until target fly count reached or no further merges are possible
  -> stitched long CSV with columns: frame, orig_id, x, y, stitched_id
"""

import ast
import math
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.spatial import ConvexHull

from src.features import add_kinematics, add_area_covered, add_path_tortuosity
 
 
# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
 
def parse_xy_cell(cell) -> Optional[Tuple[float, float]]:
    """
    Parse a cell containing (x, y) coordinates.
    Accepts tuple/list, string "(x, y)", or NaN/None (returns None).
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, (tuple, list)) and len(cell) == 2:
        return float(cell[0]), float(cell[1])
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        xy = ast.literal_eval(s)
        if isinstance(xy, (tuple, list)) and len(xy) == 2:
            return float(xy[0]), float(xy[1])
    except Exception:
        return None
    return None
 
 
def wide_to_long(
    df_wide:   pd.DataFrame,
    frame_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert wide tracker CSV (one column per ID) to long format.
    NaN rows are kept — they mark frames where the tracker lost the fly.
    Output columns: frame, orig_id, x, y
    """
    df = df_wide.copy()
    if frame_col is None:
        frame_col = next((c for c in ("frame", "Frame") if c in df.columns), None)
    if frame_col is None:
        df = df.reset_index().rename(columns={"index": "frame"})
        frame_col = "frame"
 
    id_cols = [c for c in df.columns if c != frame_col]
    records = []
    for _, row in df.iterrows():
        f = int(row[frame_col])
        for c in id_cols:
            xy = parse_xy_cell(row[c])
            if xy is not None:
                records.append((f, str(c), float(xy[0]), float(xy[1])))
 
    out = pd.DataFrame(records, columns=["frame", "orig_id", "x", "y"])
    out.sort_values(["orig_id", "frame"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
 
 
# ---------------------------------------------------------------------------
# Tracklet summaries
# ---------------------------------------------------------------------------
 
@dataclass
class Tracklet:
    orig_id:               str
    start_frame:           int
    end_frame:             int
    start_xy:              Tuple[float, float]
    end_xy:                Tuple[float, float]
    n_points:              int
    frames:                List[dict]

    # per-step profiles (used by link_score)
    velocities:            List[Optional[float]]   # true velocity (displacement / dt)
    directions:            List[Optional[float]]   # heading in degrees

    # motion summaries
    n_large_displacements: int
    distance_traveled:     float
    mean_velocity:         float
    median_velocity:       float
    mean_acceleration:     float
    mean_turning_angle:    float
    mean_angular_velocity: float

    # shape of the journey
    tortuosity:            float                    # distance_traveled / net_displacement; inf if net == 0
    area_covered:          float                    # convex hull area
    pause_fraction:        float                    # fraction of steps below velocity threshold

    # direction
    overall_direction:     float                    # atan2(end_y - start_y, end_x - start_x), degrees
    final_direction:       Optional[float]          # last valid heading, degrees
 
 
def build_tracklets(
    long_df: pd.DataFrame,
    fps: float,
    pause_threshold: float = 1.0,
) -> List[Tracklet]:
    """
    Collapse each orig_id into a Tracklet summary using kinematics
    from features.py. Requires fps to compute proper dt-based velocity.
    """
    # Run feature extraction on the long dataframe
    df = long_df.copy()
    df["fps"] = fps
    df = add_kinematics(df, group_col="orig_id")
    df = add_area_covered(df, group_col="orig_id")
    df = add_path_tortuosity(df, group_col="orig_id")

    tracklets = []
    for oid, g in df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")

        frames_data = [
            {"frame": int(row["frame"]), "x": float(row["x"]), "y": float(row["y"])}
            for _, row in g2.iterrows()
        ]

        # Per-step profiles (skip first row: it's the diff baseline with fillna(0))
        vel_series = g2["velocity"].iloc[1:]
        velocities = vel_series.tolist()
        directions = np.degrees(g2["heading"].iloc[1:]).tolist()

        # n_large_displacements
        valid_v = [v for v in velocities if v is not None and np.isfinite(v)]
        if valid_v:
            median_v = float(np.median(valid_v))
            n_large = sum(1 for v in valid_v if v > 2 * median_v)
        else:
            median_v = 0.0
            n_large = 0

        # Distance traveled: last cumsum value
        dist_traveled = float(g2["distance_traveled"].iloc[-1])

        # Net displacement for tortuosity
        start_xy = (float(g2.iloc[0]["x"]), float(g2.iloc[0]["y"]))
        end_xy = (float(g2.iloc[-1]["x"]), float(g2.iloc[-1]["y"]))
        net_disp = math.sqrt(
            (end_xy[0] - start_xy[0]) ** 2 + (end_xy[1] - start_xy[1]) ** 2
        )
        tortuosity = dist_traveled / net_disp if net_disp > 0 else np.nan

        # Area covered
        area = float(g2["area_covered"].iloc[0])

        # Pause fraction
        pause_frac = float((vel_series < pause_threshold).mean()) if len(vel_series) > 0 else 0.0

        # Overall direction: start -> end angle
        overall_dir = math.degrees(math.atan2(
            end_xy[1] - start_xy[1], end_xy[0] - start_xy[0]
        ))

        # Final direction: last valid heading in degrees
        final_dir = directions[-1] if directions else None

        tracklets.append(Tracklet(
            orig_id               = str(oid),
            start_frame           = int(g2.iloc[0]["frame"]),
            end_frame             = int(g2.iloc[-1]["frame"]),
            start_xy              = start_xy,
            end_xy                = end_xy,
            n_points              = int(len(g2)),
            frames                = frames_data,
            velocities            = velocities,
            directions            = directions,
            n_large_displacements = n_large,
            distance_traveled     = dist_traveled,
            mean_velocity         = float(np.mean(valid_v)) if valid_v else 0.0,
            median_velocity       = median_v,
            mean_acceleration     = float(g2["acceleration"].iloc[1:].mean()) if len(g2) > 1 else 0.0,
            mean_turning_angle    = float(g2["turning_angle"].iloc[1:].abs().mean()) if len(g2) > 1 else 0.0,
            mean_angular_velocity = float(g2["angular_velocity"].iloc[1:].abs().mean()) if len(g2) > 1 else 0.0,
            tortuosity            = tortuosity,
            area_covered          = area,
            pause_fraction        = pause_frac,
            overall_direction     = overall_dir,
            final_direction       = final_dir,
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
    Returns (near_horizontal_wall, near_vertical_wall).
    near_h = near left or right edge -> expect x-direction to flip
    near_v = near top or bottom edge -> expect y-direction to flip
    """
    x0, y0, x1, y1 = vial_roi
    w = x1 - x0
    h = y1 - y0
    x, y = xy
    near_h = (x - x0) < edge_fraction * w or (x1 - x) < edge_fraction * w
    near_v = (y - y0) < edge_fraction * h or (y1 - y) < edge_fraction * h
    return near_h, near_v
 
 
def _mirror_angle(angle_deg: float, near_h: bool, near_v: bool) -> float:
    """Mirror an angle based on which wall was hit."""
    rad = math.radians(angle_deg)
    dx  = math.cos(rad)
    dy  = math.sin(rad)
    if near_h:
        dx = -dx
    if near_v:
        dy = -dy
    return math.degrees(math.atan2(dy, dx))
 
 
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
    reliable = [t for t in tracklets if t.n_points >= min_points]
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
    A:                 "Tracklet",
    B:                 "Tracklet",
    gap:               int,
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    feature_scales:    Dict[str, float],
    edge_fraction:     float = 0.10,
    fallback_velocity: float = 10.0,
) -> float:
    """
    Cost of linking tracklet A -> tracklet B.  Lower = more plausible.

    Two groups of terms are averaged and then summed:

    Continuity terms — does A's end state predict B's start?
      1. Extrapolated position error: project A forward using its final heading
         and median velocity for `gap` frames; normalise by expected travel
         distance so the result is dimensionless.
      2. Direction (average of two components):
           a. A's final heading (wall-reflected when near a vial edge) vs.
              the actual gap vector.
           b. A's overall direction vs. B's overall direction — do both
              fragments share the same dominant movement tendency?

    Dissimilarity terms — same fly should show a consistent behavioural profile:
      |A.feature - B.feature| / population_std  for each of:
        median_velocity, pause_fraction, tortuosity (*),
        mean_turning_angle, mean_angular_velocity, mean_acceleration,
        n_large_displacements.
      (*) Skipped when either tracklet has non-finite tortuosity (e.g. fly
          barely moved, net displacement ≈ 0).

    Dissimilarity terms are normalised by the population std computed over
    all tracklets passed to build_cost_matrix, making all terms dimensionless
    and comparable without any manual weight selection.

    Parameters
    ----------
    A, B             : tracklets to link (A precedes B in time)
    gap              : B.start_frame - A.end_frame  (frames)
    vial_rois        : {vial_id: (x0, y0, x1, y1)} for wall detection
    feature_scales   : per-feature population std from _compute_feature_scales()
    edge_fraction    : fraction of vial dimension defining the wall zone
    fallback_velocity: used when A has no reliable velocity estimate
    """
    mv = A.median_velocity if A.median_velocity > 1e-6 else fallback_velocity

    # ------------------------------------------------------------------
    # Continuity term 1: extrapolated position error
    # ------------------------------------------------------------------
    if A.final_direction is not None:
        rad = math.radians(A.final_direction)
        ex  = A.end_xy[0] + math.cos(rad) * mv * gap
        ey  = A.end_xy[1] + math.sin(rad) * mv * gap
    else:
        # No heading available — fall back to A's last known position.
        ex, ey = A.end_xy

    extrap_error = math.sqrt((B.start_xy[0] - ex) ** 2 + (B.start_xy[1] - ey) ** 2)
    extrap_term  = extrap_error / (mv * max(gap, 1))

    # ------------------------------------------------------------------
    # Continuity term 2: direction (3-way)
    # ------------------------------------------------------------------
    dx            = B.start_xy[0] - A.end_xy[0]
    dy            = B.start_xy[1] - A.end_xy[1]
    gap_direction = math.degrees(math.atan2(dy, dx))

    dir_components: List[float] = []

    # a. A's final heading (wall-reflected if near vial edge) vs. gap vector.
    if A.final_direction is not None:
        vial_roi = _find_vial(A.end_xy, vial_rois)
        if vial_roi is not None:
            near_h, near_v = _near_wall(A.end_xy, vial_roi, edge_fraction)
            expected_dir   = (
                _mirror_angle(A.final_direction, near_h, near_v)
                if (near_h or near_v)
                else A.final_direction
            )
        else:
            expected_dir = A.final_direction
        dir_components.append(_angle_diff(gap_direction, expected_dir) / 180.0)

    # b. A's overall trajectory direction vs. B's overall trajectory direction.
    dir_components.append(_angle_diff(A.overall_direction, B.overall_direction) / 180.0)

    direction_term = float(np.mean(dir_components))

    continuity_score = (extrap_term + direction_term) / 2.0

    # ------------------------------------------------------------------
    # Dissimilarity terms: behavioural profile matching
    # ------------------------------------------------------------------
    def _dissim(a_val: float, b_val: float, key: str) -> float:
        return abs(a_val - b_val) / feature_scales.get(key, 1.0)

    dissim_components: List[float] = [
        _dissim(A.median_velocity,                B.median_velocity,                "median_velocity"),
        _dissim(A.pause_fraction,                 B.pause_fraction,                 "pause_fraction"),
        _dissim(A.mean_turning_angle,             B.mean_turning_angle,             "mean_turning_angle"),
        _dissim(A.mean_angular_velocity,          B.mean_angular_velocity,          "mean_angular_velocity"),
        _dissim(A.mean_acceleration,              B.mean_acceleration,              "mean_acceleration"),
        _dissim(float(A.n_large_displacements),   float(B.n_large_displacements),   "n_large_displacements"),
    ]

    # Tortuosity is undefined (nan/inf) for near-stationary tracklets; skip
    # those pairs rather than propagating nan into the score.
    if np.isfinite(A.tortuosity) and np.isfinite(B.tortuosity):
        dissim_components.append(_dissim(A.tortuosity, B.tortuosity, "tortuosity"))

    dissim_score = float(np.mean(dissim_components))

    return continuity_score + dissim_score
 
 
def build_cost_matrix(
    tracklets:          List[Tracklet],
    vial_rois:          Dict[str, Tuple[int, int, int, int]],
    max_gap:            int,
    max_score:          float = 2.0,
    edge_fraction:      float = 0.10,
    fallback_velocity:  float = 10.0,
    min_points_for_scale: int = 10,
) -> np.ndarray:
    """
    Build an N×N cost matrix over a list of tracklets.

    Impossible cells (wrong temporal order, gap too large, score above cap)
    are filled with BIG (1e9).  Pass max_score=1e9 to disable the cap.

    Feature scales are computed from tracklets with n_points >= min_points_for_scale
    to avoid short noisy fragments contaminating the normalisation.
    """
    BIG            = 1e9
    n              = len(tracklets)
    C              = np.full((n, n), BIG, dtype=float)
    feature_scales = _compute_feature_scales(tracklets, min_points=min_points_for_scale)

    for i, A in enumerate(tracklets):
        for j, B in enumerate(tracklets):
            if i == j or A.end_frame >= B.start_frame:
                continue
            gap = B.start_frame - A.end_frame
            if gap < 1 or gap > max_gap:
                continue
            score = link_score(
                A, B, gap, vial_rois, feature_scales,
                edge_fraction, fallback_velocity,
            )
            if score <= max_score:
                C[i, j] = score

    return C
 
 
# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------
 
def solve_assignment(
    cost_matrix: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """Hungarian assignment with greedy fallback."""
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
 
 
def build_orig_to_stitched(
    tracklets: List[Tracklet],
    matches:   List[Tuple[int, int, float]],
) -> Dict[str, str]:
    """
    Walk each matched chain from its root and label every member
    with the root's orig_id.
    Example: T1->T2->T4 becomes {T1:"T1", T2:"T1", T4:"T1"}
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
 
 
def _count_ids(tracklets: List[Tracklet], mapping: Dict[str, str]) -> int:
    return len({mapping.get(t.orig_id, t.orig_id) for t in tracklets})
 
 
def _stitch_pass(
    tracklets:            List[Tracklet],
    frozen_mapping:       Dict[str, str],
    vial_rois:            Dict[str, Tuple[int, int, int, int]],
    max_gap:              int,
    edge_fraction:        float,
    fallback_velocity:    float,
    max_score:            Optional[float],
    min_points_for_scale: int = 10,
) -> Dict[str, str]:
    """
    One Hungarian assignment pass over currently-unmerged tracklets.

    Only chain roots (tracklets not yet merged into another) are considered.
    Frozen merges from previous rounds are preserved unchanged.
    """
    roots = [t for t in tracklets
             if frozen_mapping.get(t.orig_id, t.orig_id) == t.orig_id]

    if len(roots) <= 1:
        return frozen_mapping

    _cap    = max_score if max_score is not None else 1e9
    C       = build_cost_matrix(roots, vial_rois, max_gap, _cap,
                                edge_fraction, fallback_velocity,
                                min_points_for_scale)
    matches = solve_assignment(C)

    if not matches:
        return frozen_mapping

    new_mapping = build_orig_to_stitched(roots, matches)

    # Propagate new merges into the frozen mapping.
    updated = dict(frozen_mapping)
    for orig_id, frozen_root in frozen_mapping.items():
        if frozen_root in new_mapping:
            updated[orig_id] = new_mapping[frozen_root]
    for orig_id, new_root in new_mapping.items():
        if orig_id not in updated:
            updated[orig_id] = new_root

    return updated
 
 
def stitch_per_vial(
    long_df:              pd.DataFrame,
    vial_rois:            Dict[str, Tuple[int, int, int, int]],
    n_flies_per_vial:     int,
    max_gap:              int,
    tracklets:            List[Tracklet],
    edge_fraction:        float = 0.10,
    fallback_velocity:    float = 10.0,
    max_score:            float = 2.0,
    min_points_for_scale: int   = 10,
) -> pd.DataFrame:
    """
    Stitch tracklets per vial using iterative Hungarian assignment.

    Each round operates only on unmerged tracklet roots; frozen merges are
    kept.  max_gap doubles each round to progressively allow longer gaps.
    Stops when the target fly count is reached or no further merges occur.

    Final compact_id numbering:
      vial1 -> 1..n, vial2 -> n+1..2n, ... (left to right within each vial)

    Parameters
    ----------
    long_df           : long-format dataframe (frame, orig_id, x, y)
    vial_rois         : {vial_id: (x0, y0, x1, y1)}
    n_flies_per_vial  : target number of distinct fly identities per vial
    max_gap           : starting maximum allowed frame gap between tracklets
    tracklets         : Tracklet objects from build_tracklets()
    edge_fraction        : fraction of vial dimension defining the wall zone
    fallback_velocity    : velocity used when a tracklet has no reliable estimate
    max_score            : score cap; pairs above this cost are never linked
    min_points_for_scale : tracklets shorter than this are excluded from feature
                           scale computation (they have unreliable kinematics)

    Returns
    -------
    long_df with added column: stitched_id.
    compact_id and vial_id are assigned by assign_vials_and_compact_ids() in roi.py.
    """

    # 1. Assign tracklets to vials by start_xy
    vial_tracklets: Dict[str, List[Tracklet]] = {v: [] for v in vial_rois}
    for t in tracklets:
        vial = _assign_to_vial(t.start_xy, vial_rois)
        if vial is not None:
            vial_tracklets[vial].append(t)

    # 2. Per-vial iterative stitching
    global_mapping: Dict[str, str] = {}

    for vial_id in sorted(vial_rois.keys()):
        vt = vial_tracklets[vial_id]
        if not vt:
            continue

        mapping     = {t.orig_id: t.orig_id for t in vt}
        current_gap = max_gap
        round_num   = 1

        while _count_ids(vt, mapping) > n_flies_per_vial:
            n_before = _count_ids(vt, mapping)

            mapping = _stitch_pass(
                tracklets            = vt,
                frozen_mapping       = mapping,
                vial_rois            = vial_rois,
                max_gap              = current_gap,
                edge_fraction        = edge_fraction,
                fallback_velocity    = fallback_velocity,
                max_score            = max_score,
                min_points_for_scale = min_points_for_scale,
            )

            n_after = _count_ids(vt, mapping)

            print(
                f"  {vial_id} round {round_num}: "
                f"{n_before} -> {n_after} IDs "
                f"(target {n_flies_per_vial}, max_gap={current_gap})"
            )

            if n_after == n_before:
                print(f"  {vial_id}: stuck at {n_after} IDs, stopping")
                break

            current_gap *= 2
            round_num   += 1

        global_mapping.update(mapping)
    # 3. Apply stitched_id
    out = long_df.copy()
    out["stitched_id"] = out["orig_id"].map(global_mapping).fillna(out["orig_id"])

    return out