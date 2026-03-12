"""
src/stitching.py

Hungarian-based tracklet stitching: link fragmented OC-SORT tracks across gaps.

Pipeline:
  wide CSV  ->  long format  ->  Tracklet summaries  ->  cost matrix
  ->  1-to-1 assignment  ->  stitched long CSV
"""

import ast
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def parse_xy_cell(cell) -> Optional[Tuple[float, float]]:
    """
    Parse a cell containing (x, y) coordinates.

    Accepts:
    - tuple / list of length 2
    - string "(x, y)"
    - NaN / None -> returns None
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
    df_wide: pd.DataFrame,
    frame_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert wide tracking CSV (one column per ID) to long format.

    Output columns: frame, orig_id, x, y
    """
    df = df_wide.copy()

    if frame_col is None:
        if "frame" in df.columns:
            frame_col = "frame"
        elif "Frame" in df.columns:
            frame_col = "Frame"

    if frame_col is None:
        df = df.reset_index().rename(columns={"index": "frame"})
        frame_col = "frame"

    id_cols = [c for c in df.columns if c != frame_col]

    records = []
    for _, row in df.iterrows():
        f = int(row[frame_col])
        for c in id_cols:
            xy = parse_xy_cell(row[c])
            if xy is None:
                continue
            x, y = xy
            records.append((f, str(c), float(x), float(y)))

    out = pd.DataFrame(records, columns=["frame", "orig_id", "x", "y"])
    out.sort_values(["orig_id", "frame"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Tracklet summaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tracklet:
    orig_id: str
    start_frame: int
    end_frame: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    n_points: int


def build_tracklets(long_df: pd.DataFrame) -> List[Tracklet]:
    """Collapse each orig_id into a single Tracklet summary."""
    tracklets = []
    for oid, g in long_df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")
        tracklets.append(
            Tracklet(
                orig_id=str(oid),
                start_frame=int(g2.iloc[0]["frame"]),
                end_frame=int(g2.iloc[-1]["frame"]),
                start_xy=(float(g2.iloc[0]["x"]), float(g2.iloc[0]["y"])),
                end_xy=(float(g2.iloc[-1]["x"]), float(g2.iloc[-1]["y"])),
                n_points=int(len(g2)),
            )
        )
    tracklets.sort(key=lambda t: t.orig_id)
    return tracklets


def estimate_step_scale(long_df: pd.DataFrame) -> Dict[str, float]:
    """Robust estimate of per-frame step-size statistics across all tracklets."""
    steps = []
    for _, g in long_df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")
        f = g2["frame"].to_numpy()
        x = g2["x"].to_numpy()
        y = g2["y"].to_numpy()

        consec = (f[1:] - f[:-1]) == 1
        if not np.any(consec):
            continue

        dx = x[1:][consec] - x[:-1][consec]
        dy = y[1:][consec] - y[:-1][consec]
        steps.extend(np.sqrt(dx * dx + dy * dy).tolist())

    if len(steps) == 0:
        return {"median_step": 10.0, "mad_step": 5.0, "sigma_step": 10.0}

    steps = np.asarray(steps)
    med = float(np.median(steps))
    mad = float(np.median(np.abs(steps - med)))
    sigma = float(max(1.4826 * mad, 0.25 * med, 2.0))

    return {"median_step": med, "mad_step": mad, "sigma_step": sigma}


# ---------------------------------------------------------------------------
# Link cost + assignment
# ---------------------------------------------------------------------------

def link_cost(
    end_xy: Tuple[float, float],
    start_xy: Tuple[float, float],
    gap: int,
    sigma_step: float,
    gap_penalty: float,
) -> float:
    dx = start_xy[0] - end_xy[0]
    dy = start_xy[1] - end_xy[1]
    dist = math.sqrt(dx * dx + dy * dy)
    denom = sigma_step * math.sqrt(max(gap, 1))
    z = dist / max(denom, 1e-6)
    return (z * z) + gap_penalty * gap


def build_cost_matrix(
    tracklets: List[Tracklet],
    max_gap: int,
    sigma_step: float,
    gap_penalty: float,
    max_cost: float,
) -> np.ndarray:
    n = len(tracklets)
    BIG = 1e9
    C = np.full((n, n), BIG, dtype=float)

    for i, ti in enumerate(tracklets):
        for j, tj in enumerate(tracklets):
            if i == j or ti.end_frame >= tj.start_frame:
                continue
            gap = tj.start_frame - ti.end_frame
            if gap < 1 or gap > max_gap:
                continue
            c = link_cost(ti.end_xy, tj.start_xy, gap, sigma_step, gap_penalty)
            if c <= max_cost:
                C[i, j] = c

    return C


def solve_assignment(
    cost_matrix: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """Hungarian assignment with greedy fallback."""
    BIG = 1e9
    C = cost_matrix

    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        return [
            (int(i), int(j), float(C[i, j]))
            for i, j in zip(r, c)
            if C[i, j] < BIG / 10
        ]
    except Exception:
        matches = []
        used_r, used_c = set(), set()
        edges = [(int(i), int(j), float(C[i, j])) for i, j in np.argwhere(C < BIG / 10)]
        edges.sort(key=lambda t: t[2])
        for i, j, cost in edges:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            matches.append((i, j, cost))
        return matches


def build_orig_to_stitched(
    tracklets: List[Tracklet],
    matches: List[Tuple[int, int, float]],
) -> Dict[str, str]:
    """Build orig_id -> stitched_id mapping from match pairs."""
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
        root = tracklets[idx].orig_id
        cur = idx
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
# Main entry point
# ---------------------------------------------------------------------------

def stitch_wide_csv_to_long(
    input_csv: str,
    output_stitched_long: str,
    max_gap: int,
    gap_penalty: float = 0.05,
    max_cost_quantile: float = 0.995,
    frame_col: Optional[str] = None,
) -> dict:
    """
    Stitch fragmented tracklets in a wide CSV using motion-consistent assignment.

    Parameters
    ----------
    input_csv : str
        Wide-format tracking CSV (rows = frames, cols = track IDs).
    output_stitched_long : str
        Output path for the stitched long CSV.
    max_gap : int
        Maximum frame gap to consider for linking (usually = lost_track_buffer).
    gap_penalty : float
        Cost per frame in the gap (penalises longer gaps).
    max_cost_quantile : float
        Prune the top (1 - q) most expensive candidate links before solving.
    frame_col : str, optional
        Name of the frame column if not "frame".

    Returns
    -------
    dict with summary statistics.
    """
    df_wide = pd.read_csv(input_csv)
    long_df = wide_to_long(df_wide, frame_col=frame_col)

    tracklets = build_tracklets(long_df)
    stats = estimate_step_scale(long_df)
    sigma_step = stats["sigma_step"]

    tmp_costs = [
        link_cost(ti.end_xy, tj.start_xy, tj.start_frame - ti.end_frame, sigma_step, gap_penalty)
        for ti in tracklets
        for tj in tracklets
        if ti.end_frame < tj.start_frame and 1 <= tj.start_frame - ti.end_frame <= max_gap
    ]

    max_cost = float(np.quantile(tmp_costs, max_cost_quantile)) if tmp_costs else 0.0

    C = build_cost_matrix(tracklets, max_gap, sigma_step, gap_penalty, max_cost)
    matches = solve_assignment(C)
    orig_to_stitched = build_orig_to_stitched(tracklets, matches)

    stitched_long = long_df.copy()
    stitched_long["stitched_id"] = (
        stitched_long["orig_id"].map(orig_to_stitched).fillna(stitched_long["orig_id"])
    )
    stitched_long.sort_values(["stitched_id", "frame"], inplace=True)
    stitched_long.to_csv(output_stitched_long, index=False)

    return {
        "out_stitched_long": output_stitched_long,
        "n_points": int(len(long_df)),
        "n_orig_tracklets": int(len(tracklets)),
        "n_links": int(len(matches)),
        "sigma_step": float(sigma_step),
        "median_step": float(stats["median_step"]),
        "max_cost_threshold": float(max_cost),
    }
