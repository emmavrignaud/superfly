"""Augment GT files with bbox columns (x1, y1, x2, y2).

GT files store only annotation centers (frame, ID, x, y). The grid search
metric (HOTA / CLEAR) needs bboxes. We construct them as follows:

  1. For each GT row, search the same frame in detections_raw.csv for a
     detection whose center is within MATCH_RADIUS_PX of (x, y). If found,
     copy that detection's bbox directly. These are "matched" rows.
  2. For each fly ID, compute the median bbox width and height from its
     matched rows. This is the fly's true size, taken from real detections
     of the same fly elsewhere in the video.
  3. For synthesized rows (human annotated a fly the detector missed —
     no matching detection in detections_raw.csv), build the bbox by
     centering the fly's median (w, h) on the click (x, y).
  4. Flies that have zero matched rows (annotated entirely through
     synthesis) fall back to the global median (w, h) from all
     detections in the video.

This preserves all annotated points — including the hard occluded /
clustered cases — without inventing geometry from thin air.

Usage:
    python parameter_tuning/add_gt_bboxes.py \\
        --gt parameter_tuning/data/ground_truth_31d_005_cleaned.csv \\
        --dets outputs/run_132_31DPE_n005/detections_raw.csv \\
        --out parameter_tuning/data/ground_truth_31d_005_cleaned.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_RADIUS_PX = 10.0   # GT center → detection center distance threshold


def augment(gt_path: Path, dets_path: Path, out_path: Path) -> None:
    gt = pd.read_csv(gt_path)
    dets = pd.read_csv(dets_path)

    required_gt = {"frame", "ID", "x", "y"}
    if not required_gt.issubset(gt.columns):
        sys.exit(f"GT file missing required columns {required_gt}: {gt_path}")
    required_dets = {"frame", "x1", "y1", "x2", "y2"}
    if not required_dets.issubset(dets.columns):
        sys.exit(f"Detection file missing required columns {required_dets}: {dets_path}")

    dets = dets.copy()
    dets["cx"] = (dets["x1"] + dets["x2"]) / 2.0
    dets["cy"] = (dets["y1"] + dets["y2"]) / 2.0
    dets["w"]  = dets["x2"] - dets["x1"]
    dets["h"]  = dets["y2"] - dets["y1"]

    global_w = float(dets["w"].median())
    global_h = float(dets["h"].median())

    # Index detections by frame for fast lookup
    dets_by_frame: dict[int, pd.DataFrame] = {
        int(f): grp.reset_index(drop=True) for f, grp in dets.groupby("frame")
    }

    matched_mask = np.zeros(len(gt), dtype=bool)
    out_x1 = np.full(len(gt), np.nan)
    out_y1 = np.full(len(gt), np.nan)
    out_x2 = np.full(len(gt), np.nan)
    out_y2 = np.full(len(gt), np.nan)

    for i, row in enumerate(gt.itertuples(index=False)):
        frame = int(row.frame)
        gx, gy = float(row.x), float(row.y)
        frame_dets = dets_by_frame.get(frame)
        if frame_dets is None or len(frame_dets) == 0:
            continue
        dx = frame_dets["cx"].values - gx
        dy = frame_dets["cy"].values - gy
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if d2[j] <= MATCH_RADIUS_PX * MATCH_RADIUS_PX:
            matched_mask[i] = True
            out_x1[i] = float(frame_dets["x1"].iloc[j])
            out_y1[i] = float(frame_dets["y1"].iloc[j])
            out_x2[i] = float(frame_dets["x2"].iloc[j])
            out_y2[i] = float(frame_dets["y2"].iloc[j])

    # Per-fly median size from matched rows
    matched_widths  = out_x2 - out_x1
    matched_heights = out_y2 - out_y1
    fly_size: dict[int, tuple[float, float]] = {}
    for fly_id, idx in gt.groupby("ID").groups.items():
        idx_arr = np.array(idx)
        ok = matched_mask[idx_arr]
        if ok.any():
            w = float(np.nanmedian(matched_widths[idx_arr[ok]]))
            h = float(np.nanmedian(matched_heights[idx_arr[ok]]))
            fly_size[int(fly_id)] = (w, h)

    # Fill synthesized rows
    synth_count = 0
    fly_fallback_count = 0
    for i in range(len(gt)):
        if matched_mask[i]:
            continue
        synth_count += 1
        fly_id = int(gt["ID"].iloc[i])
        if fly_id in fly_size:
            w, h = fly_size[fly_id]
        else:
            w, h = global_w, global_h
            fly_fallback_count += 1
        gx, gy = float(gt["x"].iloc[i]), float(gt["y"].iloc[i])
        out_x1[i] = gx - w / 2.0
        out_y1[i] = gy - h / 2.0
        out_x2[i] = gx + w / 2.0
        out_y2[i] = gy + h / 2.0

    gt_out = gt[["frame", "ID", "x", "y"]].copy()
    gt_out["x1"] = out_x1
    gt_out["y1"] = out_y1
    gt_out["x2"] = out_x2
    gt_out["y2"] = out_y2
    gt_out.to_csv(out_path, index=False)

    n = len(gt)
    n_matched = int(matched_mask.sum())
    print(f"GT rows           : {n}")
    print(f"  matched          : {n_matched}  ({100*n_matched/n:.1f}%)")
    print(f"  synthesized      : {synth_count}  ({100*synth_count/n:.1f}%)")
    print(f"    via fly median : {synth_count - fly_fallback_count}")
    print(f"    via global med : {fly_fallback_count}  (flies with zero matched rows)")
    print(f"Global median bbox: {global_w:.1f} x {global_h:.1f} px")
    print(f"Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="GT csv (frame,ID,x,y[,x1,y1,x2,y2])")
    ap.add_argument("--dets", required=True, type=Path, help="detections_raw.csv")
    ap.add_argument("--out", required=True, type=Path, help="output GT csv with bboxes")
    args = ap.parse_args()
    augment(args.gt, args.dets, args.out)


if __name__ == "__main__":
    main()
