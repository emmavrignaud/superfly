
"""
stitching.py
 
Full tracklet stitching pipeline for fly tracking.
 
Pipeline:
  wide CSV
    -> long format (NaN rows kept for gap analysis)
    -> Tracklet summaries (with velocity, direction, n_large_displacements)
    -> per-vial cost matrix (distance + direction + wall reflection)
    -> iterative 1-to-1 assignment per vial until target fly count reached
    -> stitched long CSV with compact_id (vial1->1..n, vial2->n+1..2n, ...)
"""
 
import ast
import math
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
 
 
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
    velocities:            List[Optional[float]]   # displacement per frame step
    directions:            List[Optional[float]]   # angle in degrees per frame step
    n_large_displacements: int                     # steps > 2x median velocity
 
 
def compute_displacement(p1, p2) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    if any(v is None for v in [p1[0], p1[1], p2[0], p2[1]]):
        return None
    return float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
 
 
def compute_direction(p1, p2) -> Optional[float]:
    """Angle in degrees. 0 = right, 90 = up, counter-clockwise."""
    if p1 is None or p2 is None:
        return None
    if any(v is None for v in [p1[0], p1[1], p2[0], p2[1]]):
        return None
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
 
 
def build_tracklets(long_df: pd.DataFrame) -> List[Tracklet]:
    """
    Collapse each orig_id into a Tracklet summary with full velocity
    and direction profiles. All positions are valid (NaN rows are dropped
    in wide_to_long so every row here has a real x, y).
    """
    tracklets = []
    for oid, g in long_df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")
 
        frames_data = [
            {"frame": int(row["frame"]), "x": float(row["x"]), "y": float(row["y"])}
            for _, row in g2.iterrows()
        ]
 
        points     = [(f["x"], f["y"]) for f in frames_data]
        pairs      = list(zip(points[:-1], points[1:]))
        velocities = [compute_displacement(p1, p2) for p1, p2 in pairs]
        directions = [compute_direction(p1, p2)    for p1, p2 in pairs]
 
        valid_v = [v for v in velocities if v is not None]
        if valid_v:
            median_v = float(np.median(valid_v))
            n_large  = sum(1 for v in valid_v if v > 2 * median_v)
        else:
            median_v = 0.0
            n_large  = 0
 
        tracklets.append(Tracklet(
            orig_id               = str(oid),
            start_frame           = int(g2.iloc[0]["frame"]),
            end_frame             = int(g2.iloc[-1]["frame"]),
            start_xy              = (float(g2.iloc[0]["x"]),  float(g2.iloc[0]["y"])),
            end_xy                = (float(g2.iloc[-1]["x"]), float(g2.iloc[-1]["y"])),
            n_points              = int(len(g2)),
            frames                = frames_data,
            velocities            = velocities,
            directions            = directions,
            n_large_displacements = n_large,
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
 
def link_score(
    A:                 Tracklet,
    B:                 Tracklet,
    gap:               int,
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    w_dist:            float = 0.5,
    w_dir:             float = 0.5,
    edge_fraction:     float = 0.10,
    fallback_velocity: float = 10.0,
) -> float:
    """
    Score the plausibility of linking tracklet A -> tracklet B.
 
    Two components:
      distance term = dist(A.end, B.start) / (A.median_velocity * gap)
                      how far did the fly travel relative to its own typical speed?
      direction term = angular difference between A's last direction and the gap
                       vector, normalised to [0, 1]. Near a wall the expected
                       direction is mirrored (horizontal reflection).
 
    score = w_dist * distance_term + w_dir * direction_term
    Lower score = more plausible link.
    """
    # Distance term
    dx   = B.start_xy[0] - A.end_xy[0]
    dy   = B.start_xy[1] - A.end_xy[1]
    dist = math.sqrt(dx**2 + dy**2)
 
    valid_v  = [v for v in A.velocities if v is not None]
    median_v = float(np.median(valid_v)) if valid_v else fallback_velocity
    median_v = max(median_v, 1e-6)
 
    distance_term = dist / (median_v * max(gap, 1))
 
    # Direction term
    gap_direction = math.degrees(math.atan2(dy, dx))
    last_dir      = next((d for d in reversed(A.directions) if d is not None), None)
 
    if last_dir is None:
        direction_term = 0.0
    else:
        vial_roi = _find_vial(A.end_xy, vial_rois)
        if vial_roi is not None:
            near_h, near_v = _near_wall(A.end_xy, vial_roi, edge_fraction)
            expected_dir   = _mirror_angle(last_dir, near_h, near_v) if (near_h or near_v) else last_dir
        else:
            expected_dir = last_dir
        direction_term = _angle_diff(gap_direction, expected_dir) / 180.0
 
    return w_dist * distance_term + w_dir * direction_term
 
 
def build_cost_matrix(
    tracklets:         List[Tracklet],
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    max_gap:           int,
    max_score:         float = 2.0,
    w_dist:            float = 0.5,
    w_dir:             float = 0.5,
    edge_fraction:     float = 0.10,
    fallback_velocity: float = 10.0,
) -> np.ndarray:
    """
    Build cost matrix over a list of tracklets.
    Impossible cells (wrong time order, gap too large, score too high) = BIG.
    Pass max_score=1e9 to remove the score cap entirely.
    """
    BIG = 1e9
    n   = len(tracklets)
    C   = np.full((n, n), BIG, dtype=float)
 
    for i, A in enumerate(tracklets):
        for j, B in enumerate(tracklets):
            if i == j or A.end_frame >= B.start_frame:
                continue
            gap = B.start_frame - A.end_frame
            if gap < 1 or gap > max_gap:
                continue
            score = link_score(A, B, gap, vial_rois, w_dist, w_dir,
                               edge_fraction, fallback_velocity)
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
    tracklets:         List[Tracklet],
    frozen_mapping:    Dict[str, str],
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    max_gap:           int,
    w_dist:            float,
    w_dir:             float,
    edge_fraction:     float,
    fallback_velocity: float,
    max_score:         Optional[float],
) -> Dict[str, str]:
    """
    One assignment pass over currently-unmerged tracklets (chain roots only).
    Frozen (already-merged) tracklets are left untouched.
    """
    # Only operate on current chain roots
    roots = [t for t in tracklets
             if frozen_mapping.get(t.orig_id, t.orig_id) == t.orig_id]
 
    if len(roots) <= 1:
        return frozen_mapping
 
    _cap = max_score if max_score is not None else 1e9
 
    C       = build_cost_matrix(roots, vial_rois, max_gap, _cap,
                                w_dist, w_dir, edge_fraction, fallback_velocity)
    matches = solve_assignment(C)
 
    if not matches:
        return frozen_mapping
 
    new_mapping = build_orig_to_stitched(roots, matches)
 
    # Merge new_mapping into frozen_mapping
    updated = dict(frozen_mapping)
    for orig_id, frozen_root in frozen_mapping.items():
        if frozen_root in new_mapping:
            updated[orig_id] = new_mapping[frozen_root]
    for orig_id, new_root in new_mapping.items():
        if orig_id not in updated:
            updated[orig_id] = new_root
 
    return updated
 
 
def stitch_per_vial(
    long_df:           pd.DataFrame,
    vial_rois:         Dict[str, Tuple[int, int, int, int]],
    n_flies_per_vial:  int,
    max_gap:           int,
    tracklets:         List[Tracklet],
    w_dist:            float = 0.5,
    w_dir:             float = 0.5,
    edge_fraction:     float = 0.10,
    fallback_velocity: float = 10.0,
    max_score:         float = 2.0,
) -> pd.DataFrame:
    """
    Stitch tracklets per vial, iterating until n_flies_per_vial IDs remain
    or no further merges are possible.
 
    Round 1: max_gap,     max_score cap
    Round 2: max_gap * 2, max_score cap
    Round 3+: max_gap * 4, no cap (BIG removed)
 
    Each round only touches unmerged tracklets — frozen merges are kept.
    Stops early if nothing merges and we are still over target.
 
    Final compact_id numbering:
      vial1 -> 1..n, vial2 -> n+1..2n, etc. (left to right within each vial)
 
    Parameters
    ----------
    long_df            : long-format dataframe (frame, orig_id, x, y)
    vial_rois          : {vial_id: (x0, y0, x1, y1)}
    n_flies_per_vial   : target number of flies per vial
    max_gap            : starting max frame gap
    tracklets          : list of Tracklet objects from build_tracklets()
    w_dist, w_dir      : score weights (should sum to 1)
    edge_fraction      : fraction of vial width/height defining the wall zone
    fallback_velocity  : step size used when a tracklet has no valid velocity
    max_score          : base score cap (removed after round 2)
 
    Returns
    -------
    long_df with added columns: stitched_id, compact_id
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
                tracklets         = vt,
                frozen_mapping    = mapping,
                vial_rois         = vial_rois,
                max_gap           = current_gap,
                w_dist            = w_dist,
                w_dir             = w_dir,
                edge_fraction     = edge_fraction,
                fallback_velocity = fallback_velocity,
                max_score         = max_score,
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
 
    # 4. Compact ID: left-to-right within each vial
    # vial1 -> 1..n, vial2 -> n+1..2n, etc.
    out["compact_id"] = -1
    offset = 0
 
    for vial_id in sorted(vial_rois.keys()):
        vt = vial_tracklets[vial_id]
        if not vt:
            continue
 
        vial_rows = out[out["orig_id"].isin([t.orig_id for t in vt])]
        x_rep     = vial_rows.groupby("stitched_id")["x"].median().sort_values()
 
        for rank, sid in enumerate(x_rep.index, start=1):
            out.loc[out["stitched_id"] == sid, "compact_id"] = offset + rank
 
        offset += len(x_rep)
 
    return out