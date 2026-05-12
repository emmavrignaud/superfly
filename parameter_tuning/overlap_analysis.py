"""Overlap analysis: can behavioral_weight_overlap help during fly overlaps?

Loads GT CSVs, finds frames where two GT fly bboxes overlap (IoU > 0),
and simulates the tracker's behavioral bonus for the correct vs swapped
assignment at each overlap event.

Key output: "bonus advantage" = bonus(correct) - bonus(wrong).
  > 0  → behavioral signal pushes toward the right decision
  = 0  → no discriminative power (flies look the same behaviorally)
  < 0  → behavioral signal would make the wrong choice

Usage:
    cd superfly
    python parameter_tuning/overlap_analysis.py [--pre-frames 30] [--fps 30] [--weight 0.30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SEQUENCES = {
    "13d_002": DATA_DIR / "ground_truth_13d_002.csv",
    "31d_005": DATA_DIR / "ground_truth_31d_005.csv",
}


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Behavioral profile from a GT trajectory segment
# Returns (median_speed_px_per_frame, median_scale_px2)
# ---------------------------------------------------------------------------

def _profile(traj: pd.DataFrame, fps: float) -> dict:
    """
    traj: rows sorted by frame with columns x, y, x1, y1, x2, y2.
    Returns the same two quantities the tracker uses in behavioral_consistency_batch:
        median_speed  : px / frame  (tracker uses px/frame internally, not px/s)
        median_scale  : median bbox area in px²
    """
    if len(traj) < 2:
        return None
    xs = traj["x"].values
    ys = traj["y"].values
    frames = traj["frame"].values

    dt = np.diff(frames.astype(float))
    dt = np.where(dt == 0, 1.0, dt)
    dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    speeds_per_frame = dists / dt          # px / frame

    areas = (traj["x2"] - traj["x1"]) * (traj["y2"] - traj["y1"])

    # mean_acceleration in px/frame² (mirroring behavioral_profile in ocsort.py)
    if len(speeds_per_frame) >= 2:
        dt2 = (dt[:-1] + dt[1:]) / 2.0
        dt2 = np.where(dt2 == 0, 1e-6, dt2)
        accel = np.abs(np.diff(speeds_per_frame)) / dt2
        mean_accel = float(np.mean(accel))
    else:
        mean_accel = 0.0

    return {
        "median_speed":   float(np.median(speeds_per_frame)),
        "mean_accel":     mean_accel,
        "median_scale":   float(np.median(areas)),
    }


# ---------------------------------------------------------------------------
# Behavioral bonus (mirrors behavioral_consistency_batch for one track pair)
# ---------------------------------------------------------------------------

def _bonus(det_cx: float, det_cy: float, det_area: float,
           trk_last_cx: float, trk_last_cy: float,
           prof: dict, weight: float) -> float:
    """
    Compute the behavioral bonus for assigning detection (det_cx, det_cy, det_area)
    to a tracker whose last observed centre is (trk_last_cx, trk_last_cy) and
    whose profile is prof.  Mirrors behavioral_consistency_batch exactly.
    """
    if prof is None:
        return 0.0
    dist = np.sqrt((det_cx - trk_last_cx) ** 2 + (det_cy - trk_last_cy) ** 2)
    expected = max(prof["median_speed"] + prof["mean_accel"], 0.0)
    excess   = max(0.0, dist - expected)
    b_speed  = max(0.0, 1.0 - excess / (expected + 1.0))

    med_scale = prof["median_scale"]
    b_scale   = max(0.0, 1.0 - abs(det_area - med_scale) / (det_area + med_scale + 1e-6))

    return weight * 0.5 * (b_speed + b_scale)


# ---------------------------------------------------------------------------
# Find overlap events in one video's GT
# ---------------------------------------------------------------------------

def find_overlap_events(gt: pd.DataFrame, min_iou: float = 0.0) -> list[dict]:
    """
    Returns a list of overlap events. Each event is a dict:
        id_a, id_b   : the two track IDs involved
        frames       : list of frame indices where the overlap is active
        iou_values   : IoU per frame
    """
    # Build per-frame lookup
    by_frame: dict[int, pd.DataFrame] = {
        f: grp for f, grp in gt.groupby("frame")
    }

    # Scan every frame for overlapping pairs
    # Store overlap state as (id_a, id_b) -> list of (frame, iou)
    pair_frames: dict[tuple, list] = {}

    for frame, grp in sorted(by_frame.items()):
        rows = grp.drop_duplicates("ID").set_index("ID")
        ids  = rows.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                box_a = rows.loc[a, ["x1","y1","x2","y2"]].to_numpy().astype(float)
                box_b = rows.loc[b, ["x1","y1","x2","y2"]].to_numpy().astype(float)
                iou   = _iou(box_a, box_b)
                if iou > min_iou:
                    key = (min(a, b), max(a, b))
                    pair_frames.setdefault(key, []).append((frame, iou))

    # Group consecutive frames per pair into events
    events = []
    for (a, b), entries in pair_frames.items():
        entries.sort()
        # split into runs of consecutive frames
        run_frames = [entries[0][0]]
        run_ious   = [entries[0][1]]
        for frame, iou in entries[1:]:
            if frame == run_frames[-1] + 1:
                run_frames.append(frame)
                run_ious.append(iou)
            else:
                events.append({"id_a": a, "id_b": b,
                                "frames": run_frames, "iou_values": run_ious})
                run_frames = [frame]
                run_ious   = [iou]
        events.append({"id_a": a, "id_b": b,
                        "frames": run_frames, "iou_values": run_ious})
    return events


# ---------------------------------------------------------------------------
# Simulate bonus advantage at each overlap event
# ---------------------------------------------------------------------------

def analyse_events(
    gt: pd.DataFrame,
    events: list[dict],
    pre_frames: int,
    fps: float,
    weight: float,
) -> pd.DataFrame:
    """
    For each overlap event, compute:
        - profiles of track A and track B in the pre_frames before the overlap
        - at the first overlap frame, simulate:
              correct assignment: A→det_A, B→det_B
              swapped  assignment: A→det_B, B→det_A
          where det_X is the GT centroid of fly X at the overlap start
        - bonus_advantage = bonus(correct) - bonus(swapped)  per track, then averaged

    Returns a DataFrame with one row per event.
    """
    trk_df = {tid: grp.drop_duplicates("frame").sort_values("frame")
               for tid, grp in gt.groupby("ID")}

    rows = []
    for ev in events:
        a, b   = ev["id_a"], ev["id_b"]
        f_start = ev["frames"][0]

        # Pre-overlap trajectory windows
        def _pre(tid):
            t = trk_df.get(tid, pd.DataFrame())
            return t[t["frame"] < f_start].tail(pre_frames)

        pre_a = _pre(a)
        pre_b = _pre(b)
        prof_a = _profile(pre_a, fps)
        prof_b = _profile(pre_b, fps)

        if prof_a is None or prof_b is None:
            continue   # not enough history

        # Last known centres before overlap
        last_a = pre_a.iloc[-1]
        last_b = pre_b.iloc[-1]
        lcx_a, lcy_a = float(last_a["x"]), float(last_a["y"])
        lcx_b, lcy_b = float(last_b["x"]), float(last_b["y"])

        # GT positions at start of overlap = "true" detection centroids
        frame_gt = gt[gt["frame"] == f_start].drop_duplicates("ID").set_index("ID")
        if a not in frame_gt.index or b not in frame_gt.index:
            continue
        ra, rb = frame_gt.loc[a], frame_gt.loc[b]
        dcx_a, dcy_a = float(ra["x"]), float(ra["y"])
        dcx_b, dcy_b = float(rb["x"]), float(rb["y"])
        darea_a = float((ra["x2"]-ra["x1"]) * (ra["y2"]-ra["y1"]))
        darea_b = float((rb["x2"]-rb["x1"]) * (rb["y2"]-rb["y1"]))

        # Correct assignment bonuses
        bon_aa = _bonus(dcx_a, dcy_a, darea_a, lcx_a, lcy_a, prof_a, weight)
        bon_bb = _bonus(dcx_b, dcy_b, darea_b, lcx_b, lcy_b, prof_b, weight)
        correct_total = bon_aa + bon_bb

        # Swapped assignment bonuses
        bon_ab = _bonus(dcx_b, dcy_b, darea_b, lcx_a, lcy_a, prof_a, weight)
        bon_ba = _bonus(dcx_a, dcy_a, darea_a, lcx_b, lcy_b, prof_b, weight)
        swapped_total = bon_ab + bon_ba

        advantage = correct_total - swapped_total

        rows.append({
            "id_a":           a,
            "id_b":           b,
            "start_frame":    f_start,
            "duration":       len(ev["frames"]),
            "mean_iou":       float(np.mean(ev["iou_values"])),
            "speed_a":        prof_a["median_speed"],
            "speed_b":        prof_b["median_speed"],
            "speed_diff":     abs(prof_a["median_speed"] - prof_b["median_speed"]),
            "scale_a":        prof_a["median_scale"],
            "scale_b":        prof_b["median_scale"],
            "scale_diff":     abs(prof_a["median_scale"] - prof_b["median_scale"]),
            "correct_bonus":  correct_total,
            "swapped_bonus":  swapped_total,
            "advantage":      advantage,      # >0 = behavioral signal is correct
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------

def _pct(series, q):
    return float(np.percentile(series, q))


def report(seq: str, df: pd.DataFrame, weight: float) -> None:
    if df.empty:
        print(f"\n{seq}: no overlap events found.")
        return

    n_total  = len(df)
    n_helps  = int((df["advantage"] > 0).sum())
    n_hurts  = int((df["advantage"] < 0).sum())
    n_neutral = n_total - n_helps - n_hurts

    print(f"\n{'='*58}")
    print(f"  {seq}  (behavioral_weight_overlap = {weight})")
    print(f"{'='*58}")
    print(f"  Overlap events found     : {n_total}")
    print(f"  Median duration (frames) : {df['duration'].median():.0f}")
    print(f"  Median overlap IoU       : {df['mean_iou'].median():.3f}")
    print()
    print(f"  Speed (px/frame) before overlap")
    print(f"    Median fly A  : {df['speed_a'].median():.2f}")
    print(f"    Median fly B  : {df['speed_b'].median():.2f}")
    print(f"    Median |A-B|  : {df['speed_diff'].median():.2f}   "
          f"(p25={_pct(df['speed_diff'],25):.2f}, p75={_pct(df['speed_diff'],75):.2f})")
    print()
    all_speeds = pd.concat([df["speed_a"], df["speed_b"]])
    all_scales = pd.concat([df["scale_a"], df["scale_b"]])
    speed_cv = all_speeds.std() / (all_speeds.mean() + 1e-6)
    scale_cv = all_scales.std() / (all_scales.mean() + 1e-6)
    # Normalize diffs by their own medians to compare on same scale
    norm_spd = df["speed_diff"] / (all_speeds.median() + 1e-6)
    norm_scl = df["scale_diff"] / (all_scales.median() + 1e-6)
    scale_more_disc = int((norm_scl > norm_spd).sum())

    print(f"  Bbox area (px^2) before overlap")
    print(f"    Median scale    : {all_scales.median():.1f} px^2  (CV={scale_cv:.2f})")
    print(f"    Median |A-B|    : {df['scale_diff'].median():.1f}  (norm={norm_scl.median():.3f})")
    print(f"  Speed (normalised) before overlap")
    print(f"    Speed CV        : {speed_cv:.2f}")
    print(f"    Norm speed diff : {norm_spd.median():.3f}")
    print(f"  Scale > speed (norm) in {scale_more_disc}/{n_total} events "
          f"({100*scale_more_disc//n_total}%) -- scale is the stronger signal there")
    print()
    print(f"  Behavioral bonus advantage (correct - swapped)")
    print(f"    Helps  (>0)  : {n_helps:>4}  ({100*n_helps/n_total:.0f}%)")
    print(f"    Neutral (=0) : {n_neutral:>4}  ({100*n_neutral/n_total:.0f}%)")
    print(f"    Hurts  (<0)  : {n_hurts:>4}  ({100*n_hurts/n_total:.0f}%)")
    print(f"    Median advantage  : {df['advantage'].median():.4f}")
    print(f"    Mean   advantage  : {df['advantage'].mean():.4f}")
    print(f"    p25 / p75         : {_pct(df['advantage'],25):.4f} / "
          f"{_pct(df['advantage'],75):.4f}")
    print()

    # Verdict
    pct_helps = 100 * n_helps / n_total
    med_adv   = df["advantage"].median()
    if pct_helps >= 70 and med_adv > 0.01:
        verdict = "GOOD — signal is mostly correct and meaningful."
    elif pct_helps >= 50 and med_adv > 0:
        verdict = "WEAK — signal helps more than it hurts, but margin is thin."
    elif pct_helps < 50:
        verdict = "RISKY — signal hurts more often than it helps. Consider lowering weight."
    else:
        verdict = "NEUTRAL — signal is correct but advantage is near zero."
    print(f"  Verdict: {verdict}")
    print(f"{'='*58}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-frames", type=int, default=30,
                        help="Frames of history to use for behavioral profiles (default 30)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video fps (default 30)")
    parser.add_argument("--weight", type=float, default=0.30,
                        help="behavioral_weight_overlap to simulate (default 0.30)")
    parser.add_argument("--min-iou", type=float, default=0.0,
                        help="Min GT IoU to count as an overlap (default 0.0 = any touch)")
    args = parser.parse_args()

    for seq, gt_path in SEQUENCES.items():
        if not gt_path.exists():
            print(f"{seq}: GT file not found at {gt_path}, skipping.")
            continue
        gt = pd.read_csv(gt_path)
        print(f"\nLoading {seq}: {len(gt)} GT rows, "
              f"{gt['ID'].nunique()} tracks, {gt['frame'].nunique()} frames")

        events = find_overlap_events(gt, min_iou=args.min_iou)
        print(f"  Found {len(events)} overlap events (min_iou={args.min_iou})")

        df = analyse_events(gt, events, pre_frames=args.pre_frames,
                            fps=args.fps, weight=args.weight)
        report(seq, df, weight=args.weight)


if __name__ == "__main__":
    main()
