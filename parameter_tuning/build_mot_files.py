"""Build MOTChallenge-format text files for TrackEval HOTA scoring.

Layout produced (suitable for TrackEval's MotChallenge2DBox with
SKIP_SPLIT_FOL=True and SEQ_INFO supplied directly):

    <out_dir>/gt/<video>/gt/gt.txt
    <out_dir>/trackers/<tracker_name>/data/<video>.txt

MOT line (9 cols, same schema for GT and predictions; see
external/TrackEval/trackeval/datasets/mot_challenge_2d_box.py lines 238-262):
    frame, id, bb_left, bb_top, bb_width, bb_height, c7, c8, c9
        c7: GT zero_marked (1 = consider) / predictions confidence (1.0)
        c8: class id (1 = pedestrian, the only class accepted by the loader)
        c9: visibility (1.0; no occlusion taxonomy for flies)
    frame is 1-indexed (TrackEval iterates t in [1, seq_length]).

Tracker bboxes:
    Tracker CSVs carry only centroids. Each tracker row is one detection
    selected from the run's detections_raw.csv at that frame, so we recover
    the bbox by exact join on (frame, x, y). Any row that fails to match is
    a hard error — silent drops would falsify HOTA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Public API
__all__ = ["build", "DEFAULT_TRACKER_NAME"]

DEFAULT_TRACKER_NAME = "baseline"


# ── public API ──────────────────────────────────────────────────────────────

def build(
    *,
    gt_csv: str | Path,
    tracks_csv: str | Path,
    dets_csv: str | Path,
    video_name: str,
    out_dir: str | Path,
    tracker_name: str = DEFAULT_TRACKER_NAME,
    id_column: str = "ordered_id",
) -> tuple[int, int]:
    """Write one (GT, predictions) pair of MOT files for `video_name` under `out_dir`.

    Args:
        gt_csv:       GT csv (cols: frame, ID, x, y, x1, y1, x2, y2).
        tracks_csv:   Tracker output csv (cols: frame, x, y, <id_column>, ...).
        dets_csv:     Raw detections csv (cols: frame, x1, y1, x2, y2, conf).
                      Used to recover tracker bboxes via (frame, centroid) join.
        video_name:   Sequence name used in folder paths and TrackEval SEQ_INFO.
        out_dir:      Root for the MOT folder layout. Created if missing.
        tracker_name: Subfolder under trackers/. Use the run id (e.g. "run_104").
        id_column:    Column to use as track id in tracks_csv (default ordered_id).

    Returns:
        (n_gt_rows, n_tracker_rows) actually written.
    """
    out_dir = Path(out_dir)
    gt_lines = _build_gt(Path(gt_csv))
    tr_lines = _build_tracker(Path(tracks_csv), Path(dets_csv), id_column=id_column)

    gt_path = out_dir / "gt" / video_name / "gt" / "gt.txt"
    tr_path = out_dir / "trackers" / tracker_name / "data" / f"{video_name}.txt"
    _write(gt_path, gt_lines)
    _write(tr_path, tr_lines)
    return len(gt_lines), len(tr_lines)


# ── internals ───────────────────────────────────────────────────────────────

def _mot_lines(df: pd.DataFrame, *, c7: float) -> list[str]:
    """df columns required: frame (0-indexed int), id (int), x1, y1, x2, y2."""
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
        "c8":    1,        # class id = pedestrian (the only class TrackEval accepts)
        "c9":    1.0,      # visibility
    })
    return [
        f"{r.frame},{r.id},{r.bbl:.3f},{r.bbt:.3f},{r.bbw:.3f},{r.bbh:.3f},"
        f"{r.c7},{r.c8},{r.c9}"
        for r in out.itertuples(index=False)
    ]


def _build_gt(gt_csv: Path) -> list[str]:
    df = pd.read_csv(gt_csv)
    required = {"frame", "ID", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{gt_csv.name} missing columns: {sorted(missing)}")
    df = df.rename(columns={"ID": "id"})
    return _mot_lines(df[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1)


def _build_tracker(tracks_csv: Path, dets_csv: Path, *, id_column: str) -> list[str]:
    tracks = pd.read_csv(tracks_csv)
    dets = pd.read_csv(dets_csv)

    if id_column not in tracks.columns:
        raise ValueError(f"{tracks_csv.name} missing id column {id_column!r}")
    tracks = tracks.dropna(subset=[id_column]).copy()
    tracks[id_column] = tracks[id_column].astype(int)

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
        on=["frame", "x", "y"], how="left", validate="many_to_one",
    )
    unmatched = merged[merged["x1"].isna()]
    if len(unmatched):
        raise RuntimeError(
            f"{len(unmatched)} tracker rows in {tracks_csv.name} failed to "
            f"join to a detection in {dets_csv.name} on (frame, x, y). "
            f"First few:\n{unmatched[['frame', id_column, 'x', 'y']].head()}"
        )

    merged = merged.rename(columns={id_column: "id"})
    return _mot_lines(merged[["frame", "id", "x1", "y1", "x2", "y2"]], c7=1.0)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(description="Build MOT files for one (video, tracker) pair.")
    p.add_argument("--gt-csv", required=True, type=Path)
    p.add_argument("--tracks-csv", required=True, type=Path)
    p.add_argument("--dets-csv", required=True, type=Path)
    p.add_argument("--video-name", required=True,
                   help="sequence name (used as folder name in the MOT layout)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="root for the MOT layout (will create gt/ and trackers/ inside)")
    p.add_argument("--tracker-name", default=DEFAULT_TRACKER_NAME)
    p.add_argument("--id-column", default="ordered_id")
    args = p.parse_args()

    n_gt, n_tr = build(
        gt_csv=args.gt_csv, tracks_csv=args.tracks_csv, dets_csv=args.dets_csv,
        video_name=args.video_name, out_dir=args.out_dir,
        tracker_name=args.tracker_name, id_column=args.id_column,
    )
    print(f"{args.video_name}: wrote {n_gt} GT rows + {n_tr} tracker rows under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
