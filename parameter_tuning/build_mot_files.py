"""Build MOTChallenge-format text files for TrackEval HOTA scoring.

Inputs (one per video-day, in parameter_tuning/data/):
    ground_truth_<seq>.csv   : frame, ID, x, y, x1, y1, x2, y2
    tracks_baseline_<seq>.csv: frame, orig_id, x, y, vial_id, ordered_id, fps
    detections_raw_<seq>.csv : frame, x1, y1, x2, y2, conf

Outputs (under parameter_tuning/results/):
    mot_inputs/gt/<seq>/gt/gt.txt
    mot_inputs/trackers/baseline/data/<seq>.txt

MOT line (9 cols, same schema for GT and predictions; see
external/TrackEval/trackeval/datasets/mot_challenge_2d_box.py lines 238-262):
    frame, id, bb_left, bb_top, bb_width, bb_height, c7, c8, c9
        c7: GT zero_marked (1 = consider) / predictions confidence (1.0)
        c8: class id (1 = pedestrian, the only class accepted by the loader)
        c9: visibility (1.0; no occlusion taxonomy for flies)
    frame is 1-indexed (TrackEval iterates t in [1, seq_length]).

Tracker bboxes:
    Tracker CSVs only carry centroids. Each tracker row is one detection
    selected from detections_raw at that frame, so we recover the bbox by
    exact join on (frame, x, y). Any row that fails to match is a hard
    error — silent drops would falsify HOTA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUT_ROOT = Path(__file__).parent / "results" / "mot_inputs"
SEQUENCES = ["13d_002", "31d_005"]
TRACKER_NAME = "baseline"


def _mot_lines(df: pd.DataFrame, *, c7: float) -> list[str]:
    """df columns required: frame (0-indexed int), id (int), x1, y1, x2, y2.
    c7 = 1 for GT (zero_marked=consider) or 1.0 for predictions (confidence).
    """
    w = df["x2"] - df["x1"]
    h = df["y2"] - df["y1"]
    if (w <= 0).any() or (h <= 0).any():
        bad = df[(w <= 0) | (h <= 0)]
        raise ValueError(f"non-positive bbox dimensions in {len(bad)} rows:\n{bad.head()}")
    out = pd.DataFrame({
        "frame": df["frame"].astype(int) + 1,   # 1-indexed for MOT
        "id":    df["id"].astype(int),
        "bbl":   df["x1"].astype(float),
        "bbt":   df["y1"].astype(float),
        "bbw":   w.astype(float),
        "bbh":   h.astype(float),
        "c7":    c7,
        "c8":    1,        # class id = pedestrian
        "c9":    1.0,      # visibility
    })
    # Format: ints for frame/id/class, fixed-precision for the rest.
    return [
        f"{r.frame},{r.id},{r.bbl:.3f},{r.bbt:.3f},{r.bbw:.3f},{r.bbh:.3f},"
        f"{r.c7},{r.c8},{r.c9}"
        for r in out.itertuples(index=False)
    ]


def _build_gt(seq: str) -> list[str]:
    src = DATA_DIR / f"ground_truth_{seq}.csv"
    df = pd.read_csv(src)
    required = {"frame", "ID", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{src.name} missing columns: {sorted(missing)}")
    df = df.rename(columns={"ID": "id"})
    return _mot_lines(df[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1)


def _build_tracker(seq: str) -> list[str]:
    tracks_path = DATA_DIR / f"tracks_baseline_{seq}.csv"
    det_path = DATA_DIR / f"detections_raw_{seq}.csv"
    tracks = pd.read_csv(tracks_path)
    dets = pd.read_csv(det_path)

    if "ordered_id" not in tracks.columns:
        raise ValueError(f"{tracks_path.name} missing ordered_id column")
    tracks = tracks.dropna(subset=["ordered_id"]).copy()
    tracks["ordered_id"] = tracks["ordered_id"].astype(int)

    # Recover bbox for each tracker row by exact-matching its centroid to a
    # raw detection at the same frame. Each tracker row should correspond to
    # exactly one detection — the one the tracker accepted that frame.
    dets = dets.copy()
    dets["x"] = (dets["x1"] + dets["x2"]) / 2.0
    dets["y"] = (dets["y1"] + dets["y2"]) / 2.0
    # Rare YOLO duplicates: same centroid, same frame, different confidence
    # (NMS near-miss). Keep the highest-conf one so the join stays many-to-one.
    n_before = len(dets)
    dets = dets.sort_values("conf", ascending=False).drop_duplicates(
        subset=["frame", "x", "y"], keep="first"
    )
    if len(dets) < n_before:
        print(f"  dedup: dropped {n_before - len(dets)} duplicate detection rows")

    merged = tracks.merge(
        dets[["frame", "x", "y", "x1", "y1", "x2", "y2"]],
        on=["frame", "x", "y"],
        how="left",
        validate="many_to_one",  # multiple track rows may share one det only
                                  # if duplicate detections exist; we expect 1:1
    )
    unmatched = merged[merged["x1"].isna()]
    if len(unmatched):
        raise RuntimeError(
            f"{len(unmatched)} tracker rows in {tracks_path.name} failed to "
            f"join to a detection in {det_path.name} on (frame, x, y). "
            f"First few:\n{unmatched[['frame', 'ordered_id', 'x', 'y']].head()}"
        )

    merged = merged.rename(columns={"ordered_id": "id"})
    return _mot_lines(merged[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1.0)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for seq in SEQUENCES:
        gt_lines = _build_gt(seq)
        tr_lines = _build_tracker(seq)
        gt_path = OUT_ROOT / "gt" / seq / "gt" / "gt.txt"
        tr_path = OUT_ROOT / "trackers" / TRACKER_NAME / "data" / f"{seq}.txt"
        _write(gt_path, gt_lines)
        _write(tr_path, tr_lines)
        print(f"{seq}: wrote {len(gt_lines):>6} GT rows -> {gt_path}")
        print(f"{seq}: wrote {len(tr_lines):>6} tracker rows -> {tr_path}")


if __name__ == "__main__":
    main()
