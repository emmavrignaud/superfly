"""Build a labeler session JSON from a ground-truth CSV.

Use this when the parameter_tuning GT CSV is the source of truth and the
existing .labeler.json in the session folder is stale (e.g. the CSV was
edited outside the labeler).

The labeler reads its state from a session JSON (annotations keyed by
(frame, det_idx) + synthetic detections + a couple of view fields). This
script reverses the labeler's export:

    GT CSV row (frame, ID, x, y, x1, y1, x2, y2)
        ↓
    match (frame, x, y) against detections_raw.csv (same indexing rule as
    labeler.data_model.load_raw_detections: per-frame, sorted by (x, y),
    det_idx = position in sorted list)
        ↓
    if matched: annotation (frame, det_idx) -> track_id, source=human
    if not matched: synthetic detection with a fresh negative det_idx,
        annotation -> track_id, source=human_synth

The resulting JSON is a valid `labeler.session` payload (version 3).

Usage:
    python parameter_tuning/csv_to_session.py \\
        --gt-csv  parameter_tuning/data/ground_truth_<seq>.csv \\
        --raw     data/manual_labelling/<folder>/detections_raw.csv \\
        --video   data/manual_labelling/<folder>/<stem>.mp4 \\
        --out     data/manual_labelling/<folder>/<stem>.labeler.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import numpy as np

SESSION_VERSION = 3
SOURCE_HUMAN = "human"
SOURCE_HUMAN_SYNTH = "human_synth"
# Centroid match tolerance. Float roundoff between GT (x,y) columns and a
# recomputed (x1+x2)/2 is on the order of 1e-13. Real synthetic detections
# (annotator drew a bbox where YOLO had nothing nearby) are typically many
# pixels away. 0.01 px cleanly separates the two.
MATCH_EPS_PX = 0.01


def _build_det_arrays(raw_csv: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Reproduce labeler.data_model.load_raw_detections indexing rule:
    per-frame, sort by (x, y), det_idx = sorted position.

    Returns {frame: (centroids[N,2], det_idx[N])}. We use arrays (not a
    centroid->idx dict) so the caller can do nearest-neighbor matching
    inside the per-frame group with a small tolerance.
    """
    df = pd.read_csv(raw_csv)
    required = {"frame", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{raw_csv.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["x"] = (df["x1"] + df["x2"]) / 2.0
    df["y"] = (df["y1"] + df["y2"]) / 2.0

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame, grp in df.groupby("frame"):
        grp = grp.sort_values(["x", "y"]).reset_index(drop=True)
        centroids = grp[["x", "y"]].to_numpy()
        det_idx = np.arange(len(grp), dtype=int)  # sorted position is the det_idx
        out[int(frame)] = (centroids, det_idx)
    return out


def build_session(
    *,
    gt_csv: Path,
    raw_csv: Path,
    video_path: Path,
    tracks_csv: Path | None,
) -> dict:
    """Return a session payload dict ready to json.dump."""
    det_arrays = _build_det_arrays(raw_csv)

    gt = pd.read_csv(gt_csv)
    required = {"frame", "ID", "x", "y", "x1", "y1", "x2", "y2"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"{gt_csv.name} missing columns: {sorted(missing)}")

    annotations: dict[str, dict] = {}
    synthetics: list[dict] = []
    synth_counter: dict[int, int] = {}  # frame -> next negative idx to assign
    n_matched = 0
    n_synth = 0
    # Track which (frame, det_idx) have already been claimed so duplicate GT
    # rows pointing at the same real detection don't both win; the second is
    # routed to a synthetic slot. Should only fire on the GT duplicates the
    # user is about to fix, but better to surface than silently overwrite.
    claimed: set[tuple[int, int]] = set()

    for r in gt.itertuples(index=False):
        frame = int(r.frame)
        track_id = int(r.ID)
        cx, cy = float(r.x), float(r.y)

        # Nearest-neighbor match in the frame's detection list, within EPS.
        det_idx = None
        frame_data = det_arrays.get(frame)
        if frame_data is not None:
            centroids, idxs = frame_data
            dists = np.hypot(centroids[:, 0] - cx, centroids[:, 1] - cy)
            # Walk candidates in ascending distance; first unclaimed within
            # EPS wins. Skipping claimed ones gracefully handles the rare
            # case of duplicate GT rows.
            for k in np.argsort(dists):
                if dists[k] > MATCH_EPS_PX:
                    break
                cand = int(idxs[k])
                if (frame, cand) not in claimed:
                    det_idx = cand
                    break

        if det_idx is not None:
            claimed.add((frame, det_idx))
            annotations[f"{frame}:{det_idx}"] = {
                "track_id": track_id,
                "source": SOURCE_HUMAN,
            }
            n_matched += 1
        else:
            # Synthetic: assign a fresh negative det_idx for this frame.
            next_neg = synth_counter.get(frame, -1)
            synth_counter[frame] = next_neg - 1
            synthetics.append({
                "frame": frame, "det_idx": next_neg,
                "x": cx, "y": cy,
                "x1": float(r.x1), "y1": float(r.y1),
                "x2": float(r.x2), "y2": float(r.y2),
            })
            annotations[f"{frame}:{next_neg}"] = {
                "track_id": track_id,
                "source": SOURCE_HUMAN_SYNTH,
            }
            n_synth += 1

    payload = {
        "version": SESSION_VERSION,
        "video_path": str(Path(video_path).as_posix()),
        "raw_csv": str(Path(raw_csv).as_posix()),
        "ocsort_csv": str(Path(tracks_csv).as_posix()) if tracks_csv else None,
        "current_frame": 0,
        "current_mode": "frame",
        "annotations": annotations,
        "synthetic_detections": synthetics,
        "confirmed_tracks": [],
    }
    print(f"  matched to real detections : {n_matched}")
    print(f"  synthetic detections       : {n_synth}")
    print(f"  total annotations          : {len(annotations)}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gt-csv", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--tracks", default=None, type=Path,
                   help="optional tracks CSV path to store in the session "
                        "(matches labeler's --tracks)")
    p.add_argument("--out", required=True, type=Path,
                   help="output session JSON path "
                        "(e.g. <session_dir>/<stem>.labeler.json)")
    args = p.parse_args()

    print(f"Building session from {args.gt_csv}")
    payload = build_session(
        gt_csv=args.gt_csv,
        raw_csv=args.raw,
        video_path=args.video,
        tracks_csv=args.tracks,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote session: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
