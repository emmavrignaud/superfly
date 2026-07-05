#!/usr/bin/env python
"""
scripts/grid_search_tracker_params.py

Grid search over OC-SORT tracker hyperparameters.
Roboflow inference is never re-run — raw detections are reused from a
detections_raw.csv generated at very low confidence (conf~0.001).

For each (confidence, min_matching_threshold, brownian_pos_noise, lost_track_buffer,
asso_func) config the tracker is re-run from detections, and the four stitching
objectives are computed directly on the raw tracker output (no stitching step).

Search space (edit the five lists at the top of this file)
------------------------------------------------------------
  CONFIDENCE_VALUES          det_thresh fed to OC-SORT  (re-filters detections_raw.csv)
  MIN_MATCHING_VALUES        iou_threshold in the Hungarian matcher
  BROWNIAN_POS_NOISE_VALUES  scale on Kalman Q position noise
  LOST_TRACK_BUFFER_VALUES   max_age — frames before a lost track is deleted
  ASSO_FUNC_VALUES           association metric: "diou" or "giou"

Fixed params (loaded from config.yaml, not searched)
-----------------------------------------------------
  minimum_consecutive_frames min_hits — frames before a track is emitted

Objectives (all lower-is-better)
---------------------------------
  vial_count_error       sum_vials |n_tracks - expected_per_vial|
  per_id_coverage_loss   num_frames - mean(track_length)
  short_track_count      tracks shorter than short_frac * num_frames
  per_frame_id_variance  sum_vials std(id_count_at_frame_t)

Outputs (in outputs\\grid_search\\tracker_params\\<short_name>\\)
-------
  grid_search_results.csv    one row per config; all params + all objectives
  grid_search_plot.html      interactive Plotly line plot sorted by vial_count_error
  overlay_detections.mp4     raw RF-DETR detections (no tracking)   [--video only]
  overlay_best.mp4           best config by vial_count_error         [--video only]
  overlay_worst.mp4          worst config by vial_count_error        [--video only]

Usage
-----
  python scripts\\grid_search_tracker_params.py ^
      --detections-csv  outputs\\run_32_6DPE_n001\\detections_raw.csv ^
      --roi-json        outputs\\run_32_6DPE_n001\\vial_rois.json ^
      --run-params      outputs\\run_32_6DPE_n001\\run_params.json

  # also generate three overlay videos:
  python scripts\\grid_search_tracker_params.py ^
      --detections-csv  outputs\\run_32_6DPE_n001\\detections_raw.csv ^
      --roi-json        outputs\\run_32_6DPE_n001\\vial_rois.json ^
      --run-params      outputs\\run_32_6DPE_n001\\run_params.json ^
      --video           outputs\\run_32_6DPE_n001\\video.mp4

  # re-plot from an existing results CSV:
  python scripts\\grid_search_tracker_params.py ^
      --detections-csv  outputs\\run_32_6DPE_n001\\detections_raw.csv ^
      --roi-json        outputs\\run_32_6DPE_n001\\vial_rois.json ^
      --run-params      outputs\\run_32_6DPE_n001\\run_params.json ^
      --plot-only
"""

import argparse
import contextlib
import io
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# Add repo root to path so src/ imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocsort import OCSort
from src.stitching import wide_to_long
from src.metrics import compute_stitching_objectives
from src.visualization import render_detections_video, render_raw_overlay_video


# ---------------------------------------------------------------------------
# Search space — edit these lists to change what is explored
# ---------------------------------------------------------------------------
#
# Each list is one axis of the grid. Every combination is evaluated, so the
# total number of configs = product of all list lengths.
# Current total: 6 × 6 × 5 × 6 × 2 = 2160 configs.

# det_thresh: minimum confidence score for a Roboflow detection to be fed
# into OC-SORT. The detections_raw.csv is loaded at near-zero confidence,
# so this re-filters it in-memory without re-running inference.
CONFIDENCE_VALUES         = [0.1, 0.25, 0.4, 0.55, 0.7, 0.9]

# iou_threshold: minimum association score (DIoU or GIoU) for a detection–
# track pair to be accepted by the Hungarian matcher. Below this, the pair
# is treated as unmatched even if it was the globally cheapest assignment.
# For tiny fly bboxes, keep this low — IoU-family metrics collapse quickly
# even for nearby boxes.
MIN_MATCHING_VALUES       = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

# brownian_pos_noise: scale factor on the Kalman filter's process noise
# matrix Q at the position entries (cx, cy). Higher values widen the
# predicted uncertainty ellipse, giving the matcher a larger search radius.
# Useful for flies that stop, turn sharply, or fall — behaviours the
# constant-velocity Kalman model cannot predict.
BROWNIAN_POS_NOISE_VALUES = [1.0,  3.0,  5.0, 10.0, 20.0]

# max_age: how many consecutive frames a track is allowed to persist without
# receiving a matched detection before OC-SORT deletes it. A track in this
# "lost" state is kept alive and its position is extrapolated by the Kalman
# filter — it can be re-matched if a detection appears nearby later.
# Too low → tracks die during occlusions (fragmentation).
# Too high → zombie tracks accumulate.
LOST_TRACK_BUFFER_VALUES  = [30,   60,   90,  120,  180,  240]

# asso_func: which geometric similarity metric is used to build the cost
# matrix fed into the Hungarian algorithm each frame.
# "diou" — Distance-IoU: IoU + centre-distance penalty normalised by the
#           enclosing diagonal. Provides a gradient even when boxes don't
#           overlap. Current baseline.
# "giou" — Generalised IoU: IoU + enclosing-area penalty. Similar motivation
#           to DIoU but area-based rather than distance-based.
# "iou", "hmiou", "ciou", "ct_dist" are excluded — see analysis in comments
# of grid_search_tracker_params.py or association.py for reasons.
ASSO_FUNC_VALUES          = ["diou", "giou"]

# Canonical ordering used to build config keys for resume support and CSV columns.
_PARAM_KEYS = ["confidence", "min_matching", "brownian_pos_noise", "lost_track_buffer", "asso_func"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_short_name(run_params_path: str) -> str:
    """
    Extract a human-readable experiment label from run_params.json.

    The video path stored in run_params.json follows the lab directory
    convention: .../<N> DPE/<NNN>/video.mp4, where N is days post-eclosion
    and NNN is the experiment number (zero-padded to 3 digits).

    Example: "...6 DPE/001/..." → "6DPE_n001"

    Falls back to the run directory name if the pattern is not found.
    """
    with open(run_params_path) as f:
        params = json.load(f)
    video = params.get("config", {}).get("video", "")
    m = re.search(r"(\d+)\s+DPE[/\\](\d+)", video)
    if m:
        return f"{m.group(1)}DPE_n{m.group(2).zfill(3)}"
    return Path(run_params_path).parent.name   # fallback: run dir name


def _config_key(d: dict) -> str:
    """
    Build a stable string that uniquely identifies one hyperparameter config.

    Used to check whether a config was already evaluated in a previous run
    (resume support). We hash only the parameter columns, not the objective
    columns, so this works both on a fresh row_meta dict and on a row read
    back from an existing results CSV.

    Example: "confidence=0.05|min_matching=0.1|brownian_pos_noise=5.0|..."
    """
    return "|".join(f"{k}={d[k]}" for k in _PARAM_KEYS if k in d)


def _run_ocsort(
    frame_dets: dict,
    num_frames: int,
    img_h: int,
    img_w: int,
    confidence: float,
    min_matching: float,
    brownian_pos_noise: float,
    lost_track_buffer: int,
    asso_func: str,
    min_hits: int,
) -> pd.DataFrame:
    """
    Run OC-SORT for one hyperparameter config on pre-loaded detections.

    Instead of reading from a video file (which would require re-running
    Roboflow inference), we receive `frame_dets`: a dict mapping each frame
    index to a (N, 5) numpy array of [x1, y1, x2, y2, conf] rows. This
    lets us iterate over 2160 configs without a single API call.

    Returns a wide-format DataFrame: rows = frames, columns = track IDs
    (id{N}), cells = "(cx, cy)" centre of the bounding box or NaN.
    This is the same format produced by src/tracking.py so the rest of the
    pipeline (wide_to_long, compute_stitching_objectives) works unchanged.
    """
    tracker = OCSort(
        det_thresh=confidence,          # detections below this are ignored
        max_age=lost_track_buffer,      # frames a lost track survives
        min_hits=min_hits,              # consecutive frames before track is emitted
        iou_threshold=min_matching,     # minimum association score to accept a match
        delta_t=3,                      # frames lookback for velocity estimation
        asso_func=asso_func,            # similarity metric for the cost matrix
        inertia=0.2,                    # weight of the OCM velocity direction term
        use_byte=False,                 # BYTE second-stage matching disabled
        brownian_pos_noise=brownian_pos_noise,  # scale on Kalman Q position entries
    )

    rows = []
    all_ids: set = set()

    for fidx in range(num_frames):
        # Retrieve this frame's detections; default to empty if none were logged.
        dets = frame_dets.get(fidx, np.zeros((0, 5)))

        # Re-apply the confidence threshold in memory. Because detections_raw.csv
        # was generated at near-zero confidence, it contains all Roboflow
        # detections. Filtering here simulates what would happen if we had run
        # inference with this specific confidence value.
        dets = dets[dets[:, 4] >= confidence]

        frame_row: dict = {"frame": fidx}

        if len(dets):
            # tracker.update() expects (N, 5) array [x1, y1, x2, y2, conf] and
            # the image dimensions (used internally to clip predictions to frame).
            # It returns (M, 6) array [x1, y1, x2, y2, track_id, conf] for the
            # currently active tracks — only those that have survived min_hits.
            tracks = tracker.update(dets[:, :5], [img_h, img_w], [img_h, img_w])
            if tracks is not None and len(tracks):
                if tracks.ndim == 1:
                    tracks = tracks[None]   # single track → add batch dimension
                xyxy = tracks[:, :4]
                tids = tracks[:, 4].astype(int)
                # Store the bounding box centre, not the corners, to match
                # the format expected by wide_to_long() downstream.
                cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
                cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
                for tid, x, y in zip(tids, cx, cy):
                    all_ids.add(int(tid))
                    frame_row[f"id{int(tid)}"] = f"({x:.2f}, {y:.2f})"

        rows.append(frame_row)

    # Reindex to guarantee every track ID has a column in every frame row,
    # with NaN where the track was not active. This is the standard wide format.
    id_cols = [f"id{t}" for t in sorted(all_ids)]
    return pd.DataFrame(rows).reindex(columns=["frame"] + id_cols)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results_df: pd.DataFrame, short_name: str, out_html: str) -> None:
    """
    Interactive Plotly line chart: one line per objective, x = config rank.

    Configs are sorted by vial_count_error (primary objective) so the leftmost
    point is always the best overall config. Hovering over any point shows the
    full hyperparameter set and all four objective values.
    """
    # Sort so rank 0 = best config by primary objective.
    configs = results_df.sort_values("vial_count_error").reset_index(drop=True)

    obj_cols = ["vial_count_error", "per_id_coverage_loss", "short_track_count", "per_frame_id_variance"]
    labels = {
        "vial_count_error":      "Vial count error",
        "per_id_coverage_loss":  "Per-ID coverage loss (frames)",
        "short_track_count":     "Short track count",
        "per_frame_id_variance": "Per-frame ID variance",
    }
    colors = {
        "vial_count_error":      "#e74c3c",
        "per_id_coverage_loss":  "#3498db",
        "short_track_count":     "#2ecc71",
        "per_frame_id_variance": "#9b59b6",
    }

    def _hover(row: pd.Series) -> str:
        # First line: the hyperparameter config. Second line: objective values.
        return (
            f"Tracker Confidence={row['confidence']}  Match Threshold={row['min_matching']}  "
            f"Brownian Noise={row['brownian_pos_noise']}  Track Buffer={int(row['lost_track_buffer'])}  "
            f"Association Function={row['asso_func']}<br>"
            f"count_err={row['vial_count_error']:.1f}  "
            f"cov_loss={row['per_id_coverage_loss']:.1f}  "
            f"short={row['short_track_count']}  "
            f"id_var={row['per_frame_id_variance']:.3f}"
        )

    hover = [_hover(row) for _, row in configs.iterrows()]

    fig = go.Figure()
    for col in obj_cols:
        fig.add_trace(go.Scatter(
            x=list(range(len(configs))),
            y=configs[col],
            mode="lines",
            name=labels[col],
            line=dict(color=colors[col], width=1.5),
            hovertext=hover,
            hoverinfo="text+name",
        ))

    fig.update_layout(
        title=(
            f"Tracker param grid search — {short_name}<br>"
            f"<sup>configs sorted by vial_count_error (ascending).  "
            f"y-axis: raw objective values (lower is better)</sup>"
        ),
        xaxis_title="Config rank (sorted by vial_count_error)",
        yaxis_title="Objective value (lower is better)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        template="plotly_white",
    )

    fig.write_html(out_html)
    print(f"Plot saved: {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over OC-SORT tracker params (reuses detections_raw.csv)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--detections-csv",    required=True,            help="Path to detections_raw.csv (generated at low confidence)")
    parser.add_argument("--roi-json",          required=True,            help="Path to vial_rois.json")
    parser.add_argument("--run-params",        required=True,            help="Path to run_params.json (for image dims, video label)")
    parser.add_argument("--expected-per-vial", type=int,   default=7,    help="Expected flies per vial")
    parser.add_argument("--short-frac",        type=float, default=0.10, help="Short track threshold as fraction of num_frames")
    parser.add_argument("--plot-only",         action="store_true",      help="Skip search, re-plot from existing results CSV")
    parser.add_argument("--output-dir",        default=None,             help="Use this folder directly (skips auto-increment)")
    parser.add_argument("--video",             default=None,             help="Source video path; if provided, generates three overlay MP4s")
    args = parser.parse_args()

    # min_hits is kept fixed across the grid — it controls how many consecutive
    # frames a new detection must appear before OC-SORT emits a track. For flies
    # that are always present in frame, this matters less than lost_track_buffer.
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    t = cfg.get("tracker", {})
    min_hits = int(t.get("minimum_consecutive_frames", 3))

    # run_params.json was written by run_tracking.py and stores video metadata
    # (dimensions, frame count) alongside the original video path. We need the
    # dimensions to call tracker.update() correctly, and the frame count to
    # iterate over all frames even if some have zero detections.
    short_name = _parse_short_name(args.run_params)
    with open(args.run_params) as f:
        rp = json.load(f)
    cfg_meta   = rp.get("config", {})
    img_h      = int(cfg_meta.get("video_height",  1080))
    img_w      = int(cfg_meta.get("video_width",   1920))
    num_frames = int(cfg_meta.get("video_frames",    0))

    # Auto-increment the output directory so re-runs don't overwrite each other.
    # If --output-dir is given explicitly, use it as-is (useful for --plot-only).
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        _base    = Path("outputs") / "grid_search" / "tracker_params" / short_name
        out_dir  = _base
        _counter = 2
        while out_dir.exists():
            out_dir  = _base.parent / f"{_base.name}_{_counter}"
            _counter += 1
        out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "grid_search_results.csv"
    plot_html   = out_dir / "grid_search_plot.html"

    # --plot-only skips the search entirely and just re-renders the HTML from
    # the existing results CSV. Useful after tweaking plot aesthetics or if the
    # search completed but the script crashed before writing the plot.
    if args.plot_only:
        if not results_csv.exists():
            print(f"No results file at {results_csv}. Run without --plot-only first.")
            sys.exit(1)
        plot_results(pd.read_csv(results_csv), short_name, str(plot_html))
        return

    # vial_rois maps vial label → (x1, y1, x2, y2) bounding rectangle.
    # Used by compute_stitching_objectives to assign each track to a vial and
    # compute per-vial metrics.
    with open(args.roi_json) as f:
        vial_rois = {k: tuple(v) for k, v in json.load(f).items()}

    print(f"Video label    : {short_name}")
    print(f"Detections CSV : {args.detections_csv}")
    print(f"ROI JSON       : {args.roi_json}")
    print(f"Image size     : {img_w}x{img_h}  |  {num_frames} frames")
    print(f"Fixed params   : min_hits={min_hits}")
    print(f"Output dir     : {out_dir}\n")

    dets_df = pd.read_csv(args.detections_csv)

    # Fall back to deriving num_frames from the detections CSV if run_params.json
    # didn't store it (e.g. if it was written by an older version of run_tracking.py).
    if num_frames == 0:
        num_frames = int(dets_df["frame"].max()) + 1
        print(f"  num_frames derived from detections CSV: {num_frames}")

    # Pre-group detections by frame index for O(1) lookup inside the inner loop.
    # Each value is a (N, 5) numpy array: [x1, y1, x2, y2, conf].
    # Frames with no detections will simply be missing from this dict; the
    # inner loop handles that with a .get() defaulting to an empty array.
    frame_dets: dict = {
        f: grp[["x1", "y1", "x2", "y2", "conf"]].values
        for f, grp in dets_df.groupby("frame")
    }
    print(f"  {len(dets_df)} detections across {len(frame_dets)} frames "
          f"(conf range: {dets_df['conf'].min():.3f}–{dets_df['conf'].max():.3f})\n")

    # itertools.product generates the full Cartesian product of all search axes.
    # Converting to a list upfront lets us report total progress.
    all_combos = list(itertools.product(
        CONFIDENCE_VALUES,
        MIN_MATCHING_VALUES,
        BROWNIAN_POS_NOISE_VALUES,
        LOST_TRACK_BUFFER_VALUES,
        ASSO_FUNC_VALUES,
    ))
    total = len(all_combos)
    print(
        f"Search space: {len(CONFIDENCE_VALUES)} conf × {len(MIN_MATCHING_VALUES)} mmt × "
        f"{len(BROWNIAN_POS_NOISE_VALUES)} Q × {len(LOST_TRACK_BUFFER_VALUES)} buf × "
        f"{len(ASSO_FUNC_VALUES)} asso = {total} configs\n"
    )

    # Resume support: if this run was interrupted (laptop lid, crash, etc.),
    # re-running the script picks up from where it left off. We read the set of
    # already-evaluated config keys and skip them in the loop. The file is opened
    # in append mode so new rows are added after the existing ones.
    completed: set = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        completed = {_config_key(row) for _, row in existing.iterrows()}
        print(f"Resuming — {len(completed)} configs done, {total - len(completed)} remaining\n")
        out_f        = open(results_csv, "a", newline="", encoding="utf-8")
        write_header = False   # header already present from the previous run
    else:
        out_f        = open(results_csv, "w", newline="", encoding="utf-8")
        write_header = True

    n_resumed = len(completed)   # configs already on disk from a previous run
    n_done    = 0                # configs evaluated in this run
    t0        = time.time()

    for conf, mmt, bn, buf, asso in all_combos:
        row_meta = {
            "confidence":         conf,
            "min_matching":       mmt,
            "brownian_pos_noise": bn,
            "lost_track_buffer":  buf,
            "asso_func":          asso,
        }

        # Skip configs already evaluated (resume support).
        if _config_key(row_meta) in completed:
            continue

        # OC-SORT prints progress lines internally; suppress them so our own
        # progress lines aren't drowned out.
        with contextlib.redirect_stdout(io.StringIO()):
            wide_df = _run_ocsort(
                frame_dets=frame_dets,
                num_frames=num_frames,
                img_h=img_h,
                img_w=img_w,
                confidence=conf,
                min_matching=mmt,
                brownian_pos_noise=bn,
                lost_track_buffer=buf,
                asso_func=asso,
                min_hits=min_hits,
            )

        # Convert wide format (rows=frames, cols=track IDs) to long format
        # (one row per active detection per frame) — this is what
        # compute_stitching_objectives expects.
        long_df = wide_to_long(wide_df)

        # We evaluate the raw tracker output without stitching. Stitching would
        # merge tracklet fragments into longer identities, masking fragmentation
        # — exactly what we want to measure here. So we set stitched_id = orig_id
        # (identity mapping) to use the objectives as a direct tracker quality signal.
        eval_df  = long_df.assign(stitched_id=long_df["orig_id"])
        n_tracks = eval_df["orig_id"].nunique()

        objs = compute_stitching_objectives(
            df_stitched       = eval_df,
            vial_rois         = vial_rois,
            num_frames        = num_frames,
            expected_per_vial = args.expected_per_vial,
            short_frac        = args.short_frac,
        )

        row = {**row_meta, **objs}
        completed.add(_config_key(row))
        n_done += 1

        print(
            f"  [{n_done + n_resumed}/{total}]  "
            f"Tracker Confidence={conf}  Match Threshold={mmt}  Brownian Noise={bn}  Track Buffer={buf}  Association Function={asso}  "
            f"→ {n_tracks} tracks  "
            f"count_err={objs['vial_count_error']:.1f}"
        )

        # Write and flush immediately so that if the script crashes the row is
        # already on disk and won't need to be re-evaluated on resume.
        pd.DataFrame([row]).to_csv(out_f, header=write_header, index=False)
        out_f.flush()
        write_header = False   # only write the CSV header on the first row

        if n_done % 50 == 0:
            elapsed    = time.time() - t0
            total_done = n_done + n_resumed
            rate       = n_done / elapsed if elapsed > 0 else 1e-9
            eta_h      = (total - total_done) / rate / 3600
            print(
                f"  {total_done}/{total}  |  "
                f"elapsed {elapsed/60:.1f} min  |  "
                f"ETA {eta_h:.1f} h"
            )

    out_f.close()
    print(f"\nGrid search complete — {n_done} new configs ({n_resumed} resumed from disk).")

    all_results = pd.read_csv(results_csv)
    print(f"\nTop 5 configs by vial_count_error:")
    print(all_results.sort_values("vial_count_error").head(5)[
        _PARAM_KEYS + ["vial_count_error", "per_id_coverage_loss",
                       "short_track_count", "per_frame_id_variance"]
    ].to_string(index=False))

    plot_results(all_results, short_name, str(plot_html))
    print(f"\nResults : {results_csv}")
    print(f"Plot    : {plot_html}")

    # --- Three overlay videos ---
    # Only generated when --video is provided. Intended for qualitative inspection:
    # watch the detections, then the best and worst tracking configs side by side.
    if args.video:
        configs_only = all_results.copy()
        if configs_only.empty:
            print("\nNo configs in results — skipping overlays.")
            return

        best_row  = configs_only.sort_values("vial_count_error").iloc[0]
        worst_row = configs_only.sort_values("vial_count_error").iloc[-1]

        print("\nGenerating overlay videos...")

        # Video 1: raw RF-DETR detections drawn as bounding boxes, no tracking.
        # Shows what the detector sees before OC-SORT assigns any identities.
        render_detections_video(
            video_path  = args.video,
            det_log_csv = args.detections_csv,
            out_mp4     = str(out_dir / "overlay_detections.mp4"),
        )

        def _render_config(label: str, row: pd.Series, out_mp4: str) -> None:
            """Re-run OC-SORT for one config row and render its tracks as an overlay."""
            wide_df = _run_ocsort(
                frame_dets=frame_dets, num_frames=num_frames,
                img_h=img_h, img_w=img_w,
                confidence=float(row["confidence"]),
                min_matching=float(row["min_matching"]),
                brownian_pos_noise=float(row["brownian_pos_noise"]),
                lost_track_buffer=int(row["lost_track_buffer"]),
                asso_func=str(row["asso_func"]), min_hits=min_hits,
            )
            long_df = wide_to_long(wide_df)
            # render_raw_overlay_video reads a temporary CSV of long-format tracks.
            # We write it, render the video, then delete it.
            tmp_csv = str(out_dir / f"_tmp_{label}.csv")
            long_df[["frame", "orig_id", "x", "y"]].to_csv(tmp_csv, index=False)
            render_raw_overlay_video(video_path=args.video, csv_path=tmp_csv, out_mp4=out_mp4)
            Path(tmp_csv).unlink()

        # Videos 2 & 3: best and worst configs by vial_count_error.
        _render_config("best",  best_row,  str(out_dir / "overlay_best.mp4"))
        _render_config("worst", worst_row, str(out_dir / "overlay_worst.mp4"))

        print(f"  overlay_detections.mp4  (raw RF-DETR detections)")
        print(
            f"  overlay_best.mp4        "
            f"(Tracker Confidence={best_row['confidence']}  Match Threshold={best_row['min_matching']}  "
            f"Brownian Noise={best_row['brownian_pos_noise']}  Track Buffer={int(best_row['lost_track_buffer'])}  "
            f"Association Function={best_row['asso_func']})"
        )
        print(
            f"  overlay_worst.mp4       "
            f"(Tracker Confidence={worst_row['confidence']}  Match Threshold={worst_row['min_matching']}  "
            f"Brownian Noise={worst_row['brownian_pos_noise']}  Track Buffer={int(worst_row['lost_track_buffer'])}  "
            f"Association Function={worst_row['asso_func']})"
        )


if __name__ == "__main__":
    main()
