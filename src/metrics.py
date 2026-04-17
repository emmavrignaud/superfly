"""
src/metrics.py

Quantitative metrics and diagnostics for the tracking pipeline.

Purpose
-------
After running the tracker and stitching, it is hard to know from the overlay
video alone whether a problem is caused by the detector, the tracker, or the
stitching. This module answers the question: at which stage is information
being lost, and why?

It works in three stages, each pointing to different fixes:

  Stage 1 — Detector
    Are the right number of flies being detected each frame?
    Low here → labelling or confidence threshold changes perhaps

  Stage 2 — OCSort tracker
    Of those detections, how many became tracks? And how long do they last?
    With any short suppressed tracks → min_hits too high (takes 10 frames in a row to create a track but should be lower), 
    or flies jumping a lot!
    Many IDs but high coverage → maybe matching problem (wrong asso_func or IoU threshold).

  Stage 3 — Stitching
    Did stitching reduce IDs without losing frame coverage?
    IDs reduced but coverage drops → score may be too high (strict)
    IDs not reduced enough → link_score too lenient or gaps too large.

Main entry point
----------------
    run_diagnostics(tracker, df_wide, df_stitched, n_expected, fps)

Prints a summary and shows four plots stacked vertically:
  1. Detection log — raw detections vs emitted tracks per frame
  2. Tracklet timeline — one bar per track, coloured by emitted vs suppressed
  3. Suppressed track length histogram — where in the min_hits range do they cluster?
  4. Coverage before vs after stitching per vial (if df_stitched provided)
"""

import io
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Stage 1 + 2 summary numbers
# ---------------------------------------------------------------------------

def compute_tracker_stats(tracker, df_wide):
    """
    Compute summary statistics from the OCSort tracker and the wide CSV.

    Parameters
    ----------
    tracker  : OCSort object returned by export_tracks_xy_tuple_csv_one_config
    df_wide  : wide-format DataFrame (rows=frames, columns=track IDs)

    Returns
    -------
    dict with keys:
      n_frames            total frames processed
      mean_detections     mean raw detections per frame (above det_thresh)
      mean_emitted        mean tracks emitted per frame
      n_emitted_ids       unique track IDs in the CSV
      n_suppressed        number of suppressed tracks (never reached min_hits)
      mean_suppressed_hits  mean hit count of suppressed tracks
      mean_id_coverage    mean % of total frames each emitted ID is present
    """
    log = np.array(tracker.detection_log)   # (n_frames, 3): frame_idx, n_dets, n_emitted
    n_frames     = len(log)
    mean_dets    = float(np.mean(log[:, 1]))
    mean_emitted = float(np.mean(log[:, 2]))

    id_cols = [c for c in df_wide.columns if c != "frame"]
    n_ids   = len(id_cols)

    # coverage per ID: fraction of frames where this ID has a detection
    coverages = [(df_wide[c].notna() & (df_wide[c] != "")).mean() for c in id_cols]
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0

    sup = tracker.suppressed_tracks
    n_sup = len(sup)
    if n_sup > 0:
        hits = [s["hits"] for s in sup]
        mean_sup_hits = float(np.mean(hits))
        near_thresh = sum(1 for h in hits if h >= tracker.min_hits * 0.7)
        pct_near = near_thresh / n_sup * 100
    else:
        mean_sup_hits = 0.0
        pct_near      = 0.0

    return {
        "n_frames":                       n_frames,
        "mean_detections":                round(mean_dets, 2),
        "mean_emitted":                   round(mean_emitted, 2),
        "n_emitted_ids":                  n_ids,
        "n_suppressed":                   n_sup,
        "mean_suppressed_hits":           round(mean_sup_hits, 2),
        "mean_id_coverage":               round(mean_coverage * 100, 1),
    }


def print_tracker_summary(stats, tracker, n_expected=None):
    """
    Print a human-readable summary with basic interpretation hints.
    Also returns the full text as a string (for saving to a report file).

    Parameters
    ----------
    stats      : dict from compute_tracker_stats()
    tracker    : OCSort object (used for min_hits, max_age)
    n_expected : expected number of flies in total (optional, used for fragmentation ratio)

    Returns
    -------
    str — the same text that was printed
    """
    lines = []
    lines.append("=" * 55)
    lines.append("  TRACKER DIAGNOSTICS")
    lines.append("=" * 55)
    lines.append(f"  Frames processed        : {stats['n_frames']}")
    lines.append(f"  Mean detections / frame : {stats['mean_detections']}")
    lines.append(f"  Mean emitted / frame    : {stats['mean_emitted']}")
    lines.append(f"  Unique IDs in CSV       : {stats['n_emitted_ids']}")
    if n_expected:
        ratio = stats['n_emitted_ids'] / n_expected
        lines.append(f"  Fragmentation ratio     : {ratio:.1f}x  ({stats['n_emitted_ids']} ids / {n_expected} expected)")
    lines.append(f"  Mean ID coverage        : {stats['mean_id_coverage']}% of frames")
    lines.append(f"  Suppressed tracks       : {stats['n_suppressed']}  (died before min_hits={tracker.min_hits})")
    lines.append(f"  Mean hits (suppressed)  : {stats['mean_suppressed_hits']}")
    lines.append(f"  Near threshold (>=70%)  : {stats.get('pct_suppressed_near_threshold', 0.0)}% of suppressed")
    lines.append("")

    # interpretation hints — each triggered by a specific threshold
    hints = []
    if stats['mean_detections'] < (n_expected or 1) * 0.8:
        hints.append("⚠ Detector is missing flies frequently → check preprocessing or lower confidence threshold")
    if stats.get('pct_suppressed_near_threshold', 0.0) > 40:
        hints.append(f"⚠ {stats['pct_suppressed_near_threshold']}% of suppressed tracks nearly reached min_hits → consider lowering min_hits (currently {tracker.min_hits})")
    if stats['mean_id_coverage'] < 30:
        hints.append("⚠ Low mean coverage per ID → tracks are very short and broken → check Q matrix (Brownian noise) or IoU threshold")
    if n_expected and stats['n_emitted_ids'] > n_expected * 5:
        hints.append(f"⚠ Very high fragmentation ({stats['n_emitted_ids']} ids vs {n_expected} expected) → matching is the main problem")

    if hints:
        lines.append("  Interpretation:")
        for h in hints:
            lines.append(f"    {h}")
    else:
        lines.append("  No obvious issues flagged.")
    lines.append("=" * 55)

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_xy_trajectories(df_wide, vial_rois=None, ax=None):
    """
    XY trajectory plot: one coloured line per emitted track ID.

    Each track is drawn as a path through (x, y) space, so you can see:
      - tracks that stay within one vial column  → good
      - tracks that jump across vials            → ID-switch bug
      - tracks that wander erratically           → matching/IoU problem

    Parameters
    ----------
    df_wide    : wide-format DataFrame (rows=frames, columns=track IDs)
    vial_rois  : optional dict  {vial_id: [x1, y1, x2, y2]}
                 draws grey ROI rectangles behind the trajectories so you
                 can see which vial each track belongs to.
                 Load from run_params.json["roi"].
    ax         : optional matplotlib axis

    Returns
    -------
    ax
    """
    id_cols = [c for c in df_wide.columns if c != "frame"]
    cmap = plt.cm.get_cmap("tab20", max(len(id_cols), 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    # draw vial ROI rectangles first (behind trajectories)
    if vial_rois:
        for label, (x1, y1, x2, y2) in vial_rois.items():
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor="lightgrey", facecolor="whitesmoke", zorder=0
            )
            ax.add_patch(rect)
            ax.text((x1 + x2) / 2, y1 - 8, label, ha="center", va="bottom",
                    fontsize=7, color="grey")

    for i, col in enumerate(id_cols):
        sub = df_wide[df_wide[col].notna() & (df_wide[col] != "")][["frame", col]].copy()
        if sub.empty:
            continue
        # parse "(x, y)" strings
        xy = sub[col].str.strip("()").str.split(", ", expand=True).astype(float)
        xs, ys = xy.iloc[:, 0].values, xy.iloc[:, 1].values
        frames = sub["frame"].values

        color = cmap(i)
        # draw line with alpha so overlapping tracks are visible
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.7)
        # mark start with a circle, end with a cross
        ax.scatter(xs[0],  ys[0],  marker="o", s=30, color=color, zorder=5)
        ax.scatter(xs[-1], ys[-1], marker="x", s=30, color=color, zorder=5)
        # label at the midpoint
        mid = len(xs) // 2
        ax.text(xs[mid], ys[mid], col, fontsize=6, color=color,
                ha="center", va="center", zorder=6)

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()   # image coordinates: y increases downward
    ax.set_title("XY trajectories per track ID  (○=start  ×=end)\n"
                 "Tracks that jump across vials indicate ID switches")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_xy_trajectories_compact(df_compact, vial_rois=None, ax=None):
    """
    XY trajectory plot using compact IDs (after stitching + vial assignment).

    Each line is one compact_id, so you can verify that stitching correctly
    merged fragments and that each compact ID stays within its vial.

    Parameters
    ----------
    df_compact : compact_tracks DataFrame (frame, x, y, compact_id, vial_id, ...)
    vial_rois  : optional dict {vial_id: [x1, y1, x2, y2]}
    ax         : optional matplotlib axis
    """
    compact_ids = sorted(df_compact["compact_id"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(compact_ids), 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    if vial_rois:
        for label, (x1, y1, x2, y2) in vial_rois.items():
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor="lightgrey", facecolor="whitesmoke", zorder=0
            )
            ax.add_patch(rect)
            ax.text((x1 + x2) / 2, y1 - 8, label, ha="center", va="bottom",
                    fontsize=7, color="grey")

    for i, cid in enumerate(compact_ids):
        sub = df_compact[df_compact["compact_id"] == cid].sort_values("frame")
        if sub.empty:
            continue
        xs, ys = sub["x"].values, sub["y"].values
        color = cmap(i % 20)
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.7)
        ax.scatter(xs[0],  ys[0],  marker="o", s=30, color=color, zorder=5)
        ax.scatter(xs[-1], ys[-1], marker="x", s=30, color=color, zorder=5)
        mid = len(xs) // 2
        ax.text(xs[mid], ys[mid], str(cid), fontsize=6, color=color,
                ha="center", va="center", zorder=6)

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.set_title("XY trajectories — compact IDs after stitching  (○=start  ×=end)\n"
                 "Each line = one stitched fly. Cross-vial lines = stitching bug.")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_detection_log(tracker, fps=30, ax=None):
    """
    Line plot: raw detections vs emitted tracks per frame.

    The gap between the two lines is the signal suppressed by min_hits
    or lost in association. A persistently large gap means a lot of real
    fly detections are never making it into the CSV.
    """
    log = np.array(tracker.detection_log)
    frames    = log[:, 0]
    n_dets    = log[:, 1]
    n_emitted = log[:, 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.fill_between(frames, n_dets, n_emitted, alpha=0.15, color="red", label="suppressed gap")
    ax.plot(frames, n_dets,    color="steelblue", lw=1.2, label="detections (RF-DETR)")
    ax.plot(frames, n_emitted, color="darkorange", lw=1.2, label="emitted tracks")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")
    ax.set_title("Stage 1→2: Detections vs emitted tracks per frame")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_tracklet_timeline(df_wide, suppressed_tracks=None, ax=None):
    """
    Tracklet timeline: one horizontal bar per track ID showing when it is active.
    Emitted tracks (in CSV) are blue. Suppressed tracks are shown in red if provided.

    Gaps in emitted tracks are visible as breaks in the bars — these are the
    fragments stitching has to repair.
    """
    id_cols = [c for c in df_wide.columns if c != "frame"]
    n_frames = len(df_wide)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, max(4, len(id_cols) * 0.25)))

    # emitted tracks
    for y, col in enumerate(id_cols):
        present = df_wide[col].notna() & (df_wide[col] != "")
        frames_present = df_wide.loc[present, "frame"].values
        if len(frames_present) == 0:
            continue
        # draw contiguous segments as bars
        starts, ends = _contiguous_segments(frames_present)
        for s, e in zip(starts, ends):
            ax.barh(y, e - s + 1, left=s, height=0.7, color="steelblue", alpha=0.8)

    # suppressed tracks (below the emitted ones)
    if suppressed_tracks:
        offset = len(id_cols) + 1
        for y, trk in enumerate(suppressed_tracks):
            frames = [xy[0] for xy in trk["xy"]]
            if not frames:
                continue
            starts, ends = _contiguous_segments(np.array(frames))
            for s, e in zip(starts, ends):
                ax.barh(offset + y, e - s + 1, left=s, height=0.7, color="tomato", alpha=0.7)

    blue_patch = mpatches.Patch(color="steelblue", label="emitted tracks")
    red_patch  = mpatches.Patch(color="tomato",    label="suppressed tracks")
    ax.legend(handles=[blue_patch, red_patch], loc="upper right", fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Track")
    ax.set_title("Stage 2: Tracklet timeline (blue=emitted, red=suppressed)")
    ax.set_xlim(0, n_frames)
    return ax


def plot_suppressed_histogram(tracker, ax=None):
    """
    Histogram of suppressed track lengths (hit counts).

    Where the histogram peaks tells you why tracks are being suppressed:
      Peak at 1-3  → genuine noise, very fast movement, or det_thresh too low
      Peak at 7-9  → min_hits is too strict (currently {min_hits}), lower it
      Flat spread  → mixed causes
    """
    sup = tracker.suppressed_tracks
    if not sup:
        print("No suppressed tracks.")
        return ax

    hits = [s["hits"] for s in sup]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    ax.hist(hits, bins=range(1, tracker.min_hits + 2), color="tomato",
            edgecolor="white", alpha=0.85)
    ax.axvline(tracker.min_hits, color="black", ls="--", lw=1.2,
               label=f"min_hits = {tracker.min_hits}")
    ax.set_xlabel("Hits before track was suppressed")
    ax.set_ylabel("Number of tracks")
    ax.set_title("Stage 2: Suppressed track length distribution")
    ax.legend(fontsize=8)
    return ax


def plot_coverage_comparison(df_wide, df_stitched, ax=None):
    """
    Bar chart: mean frame coverage per ID before and after stitching.

    If coverage drops after stitching, the link_score is merging tracks
    in a way that loses detections. If coverage increases, stitching is
    successfully bridging gaps.
    """
    id_cols_wide = [c for c in df_wide.columns if c != "frame"]
    cov_before = np.mean(
        [(df_wide[c].notna() & (df_wide[c] != "")).mean() for c in id_cols_wide]
    ) * 100 if id_cols_wide else 0

    if df_stitched is not None and "stitched_id" in df_stitched.columns:
        n_frames = df_wide["frame"].max() + 1
        cov_after_vals = []
        for sid, grp in df_stitched.groupby("stitched_id"):
            cov_after_vals.append(grp["frame"].nunique() / n_frames)
        cov_after = np.mean(cov_after_vals) * 100
        n_before = len(id_cols_wide)
        n_after  = df_stitched["stitched_id"].nunique()
    else:
        cov_after = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    labels = ["Before stitching"]
    values = [cov_before]
    colors = ["steelblue"]
    if cov_after is not None:
        labels.append("After stitching")
        values.append(cov_after)
        colors.append("mediumseagreen")

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Mean frame coverage per ID (%)")
    ax.set_title("Stage 2→3: Coverage before vs after stitching")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9)
    if cov_after is not None:
        ax.text(0.5, -0.25,
                f"IDs: {n_before} → {n_after}  |  Coverage: {cov_before:.1f}% → {cov_after:.1f}%",
                ha="center", transform=ax.transAxes, fontsize=8, color="gray")
    return ax


# ---------------------------------------------------------------------------
# Stage 3 — Stitching duplicate check
# ---------------------------------------------------------------------------

def compute_stitch_duplicate_stats(df_stitched, vial_rois=None):
    """
    Check the stitched output for (frame, stitched_id) duplicates.

    A duplicate means two rows share the same frame and stitched_id —
    i.e. one stitched fly has two recorded positions in one frame.
    Expected to be 0 after the chain_end_frames fix in stitching.py.

    Parameters
    ----------
    df_stitched : stitched long-format DataFrame with columns (frame, stitched_id, x, y)
    vial_rois   : optional dict {vial_id: (x0, y0, x1, y1)} — used to label
                  which vial each duplicated stitched_id belongs to

    Returns
    -------
    dict with keys:
      n_duplicate_ids    : number of stitched_ids that appear twice in the same frame
      n_duplicate_frames : total (frame, stitched_id) pairs that have >1 row
      details            : list of {stitched_id, vial_id, n_frames} sorted by n_frames desc
    """
    dupes = df_stitched[df_stitched.duplicated(subset=["frame", "stitched_id"], keep=False)]
    if dupes.empty:
        return {"n_duplicate_ids": 0, "n_duplicate_frames": 0, "details": []}

    def _vial_for(x, y):
        if not vial_rois:
            return "?"
        for vial_id, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return vial_id
        return "?"

    id_vial = {
        sid: _vial_for(grp["x"].iloc[0], grp["y"].iloc[0])
        for sid, grp in df_stitched.groupby("stitched_id")
    }

    summary      = dupes.groupby("stitched_id")["frame"].nunique()
    total_frames = int(dupes.drop_duplicates(subset=["frame", "stitched_id"]).shape[0])
    details      = [
        {"stitched_id": str(sid), "vial_id": id_vial.get(sid, "?"), "n_frames": int(n)}
        for sid, n in summary.sort_values(ascending=False).items()
    ]

    return {
        "n_duplicate_ids":    len(details),
        "n_duplicate_frames": total_frames,
        "details":            details,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Stitching quality objectives
# ---------------------------------------------------------------------------

def compute_stitching_objectives(
    df_stitched:       pd.DataFrame,
    vial_rois:         dict,
    num_frames:        int,
    expected_per_vial: int   = 7,
    short_frac:        float = 0.10,
) -> dict:
    """
    Compute the four stitching quality objectives for one stitched output.
    All are lower-is-better.

    Parameters
    ----------
    df_stitched       : long-format stitched DataFrame with columns (frame, stitched_id, x, y)
    vial_rois         : {vial_id: (x0, y0, x1, y1)}
    num_frames        : total number of frames in the video
    expected_per_vial : ground-truth fly count per vial (default 7)
    short_frac        : threshold for short_track_count as fraction of num_frames (default 0.10)

    Returns
    -------
    dict with keys: vial_count_error, per_id_coverage_loss, short_track_count, per_frame_id_variance
    """
    df = df_stitched.copy()

    # assign each detection to a vial
    df["_vial"] = None
    for vial_id, (x0, y0, x1, y1) in vial_rois.items():
        mask = (df["x"] >= x0) & (df["x"] <= x1) & (df["y"] >= y0) & (df["y"] <= y1)
        df.loc[mask, "_vial"] = vial_id

    track_lengths = df.groupby("stitched_id")["frame"].nunique()

    vial_count_error = float(sum(
        abs(df[df["_vial"] == v]["stitched_id"].nunique() - expected_per_vial)
        for v in vial_rois
    ))
    per_id_coverage_loss  = float(num_frames - track_lengths.mean())
    short_track_count     = int((track_lengths < short_frac * num_frames).sum())
    per_frame_id_variance = 0.0
    for v in vial_rois:
        sub = df[df["_vial"] == v]
        if sub.empty:
            continue
        per_frame = sub.groupby("frame")["stitched_id"].nunique()
        full = per_frame.reindex(range(num_frames), fill_value=0)
        per_frame_id_variance += float(full.std())

    return {
        "vial_count_error":      vial_count_error,
        "per_id_coverage_loss":  per_id_coverage_loss,
        "short_track_count":     short_track_count,
        "per_frame_id_variance": per_frame_id_variance,
    }


def print_stitching_objectives(objectives: dict) -> str:
    """
    Print a formatted stitching objectives block and return the text.

    Parameters
    ----------
    objectives : dict from compute_stitching_objectives()

    Returns
    -------
    str — the same text that was printed
    """
    lines = [
        "=" * 55,
        "  STITCHING QUALITY OBJECTIVES",
        "=" * 55,
        f"  vial_count_error      : {objectives['vial_count_error']:.1f}  (0 = perfect)",
        f"  per_id_coverage_loss  : {objectives['per_id_coverage_loss']:.1f}  frames/fly",
        f"  short_track_count     : {objectives['short_track_count']}",
        f"  per_frame_id_variance : {objectives['per_frame_id_variance']:.3f}",
        "=" * 55,
    ]
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(tracker, df_wide, df_stitched=None, df_compact=None,
                    n_expected=None, fps=30, vial_rois=None,
                    config=None, output_dir=None,
                    stitching_objectives=None):
    """
    Run all diagnostics, display plots, and optionally save a report.

    Parameters
    ----------
    tracker     : OCSort object returned by export_tracks_xy_tuple_csv_one_config
    df_wide     : wide-format DataFrame from tracking
    df_stitched : stitched long-format DataFrame (optional, for stage 3)
    df_compact  : compact_tracks DataFrame (optional, shows post-stitching trajectories)
    n_expected  : total number of flies expected (e.g. n_vials × flies_per_vial)
    fps         : video frame rate (used for axis labels)
    vial_rois   : optional dict {vial_id: [x1, y1, x2, y2]} for XY plot background
                  (load from run_params.json["roi"])
    config      : optional dict — the full config used for this run (from config.yaml or
                  run_params.json). Printed in the summary and included in the report.
    output_dir  : optional path — if given, saves plots as PNGs and writes
                  metrics_report.md inside that directory.
    """
    # --- print config if provided ---
    if config is not None:
        print("=" * 55)
        print("  RUN CONFIGURATION")
        print("=" * 55)
        print(json.dumps(config, indent=2, default=str))
        print("=" * 55)
        print()

    stats = compute_tracker_stats(tracker, df_wide)
    summary_text = print_tracker_summary(stats, tracker, n_expected)

    dup_stats = compute_stitch_duplicate_stats(df_stitched, vial_rois) if df_stitched is not None else None

    # --- XY trajectory plots: raw tracker IDs vs compact IDs side by side ---
    if df_compact is not None:
        fig_xy, (ax_raw, ax_compact) = plt.subplots(1, 2, figsize=(20, 8))
        fig_xy.suptitle("XY trajectories: raw tracker IDs (left) vs compact IDs after stitching (right)",
                         fontsize=12, fontweight="bold")
        plot_xy_trajectories(df_wide, vial_rois=vial_rois, ax=ax_raw)
        plot_xy_trajectories_compact(df_compact, vial_rois=vial_rois, ax=ax_compact)
    else:
        fig_xy, ax_raw = plt.subplots(figsize=(14, 8))
        fig_xy.suptitle("Track XY trajectories", fontsize=13, fontweight="bold")
        plot_xy_trajectories(df_wide, vial_rois=vial_rois, ax=ax_raw)
    plt.tight_layout()

    # --- Pipeline stage plots stacked ---
    fig_pipeline, axes = plt.subplots(4, 1, figsize=(14, 14),
                                      gridspec_kw={"height_ratios": [2, 4, 2, 2]})
    fig_pipeline.suptitle("Tracking pipeline diagnostics", fontsize=13, fontweight="bold")

    plot_detection_log(tracker, fps=fps, ax=axes[0])
    plot_tracklet_timeline(df_wide, suppressed_tracks=tracker.suppressed_tracks, ax=axes[1])
    plot_suppressed_histogram(tracker, ax=axes[2])
    plot_coverage_comparison(df_wide, df_stitched, ax=axes[3])

    plt.tight_layout()

    # --- save report if output_dir given ---
    if output_dir is not None:
        _save_report(output_dir, summary_text, config, fig_xy, fig_pipeline, dup_stats, stitching_objectives)

    plt.show()


def _save_report(output_dir, summary_text, config, fig_xy, fig_pipeline, dup_stats=None, objectives=None):
    """
    Save the two diagnostic figures as PNGs and write a metrics_report.md
    that embeds them alongside the text summary and config.

    Files written to output_dir:
      metrics_xy_trajectories.png  — XY trajectory side-by-side plot
      metrics_pipeline.png         — 4-panel pipeline diagnostics plot
      metrics_report.md            — markdown report with all of the above
    """
    os.makedirs(output_dir, exist_ok=True)

    xy_png       = os.path.join(output_dir, "metrics_xy_trajectories.png")
    pipeline_png = os.path.join(output_dir, "metrics_pipeline.png")
    report_md    = os.path.join(output_dir, "metrics_report.md")

    fig_xy.savefig(xy_png,       dpi=150, bbox_inches="tight")
    fig_pipeline.savefig(pipeline_png, dpi=150, bbox_inches="tight")

    # build markdown
    md = ["# Metrics Report\n"]

    if config is not None:
        md.append("## Configuration\n")
        md.append("```json")
        md.append(json.dumps(config, indent=2, default=str))
        md.append("```\n")

    md.append("## Summary\n")
    md.append("```")
    md.append(summary_text)
    md.append("```\n")

    md.append("## Stitching Duplicate Check\n")
    if dup_stats is not None:
        n_ids    = dup_stats["n_duplicate_ids"]
        n_frames = dup_stats["n_duplicate_frames"]
        if n_ids == 0:
            md.append("**0 duplicate (frame, stitched_id) pairs — clean.**\n")
        else:
            md.append(f"**{n_ids} stitched_id(s) had duplicates across {n_frames} frame(s).**\n")
            md.append("| stitched_id | vial | affected frames |")
            md.append("|---|---|---|")
            for d in dup_stats["details"]:
                md.append(f"| {d['stitched_id']} | {d['vial_id']} | {d['n_frames']} |")
            md.append("")
    else:
        md.append("_Stitched output not provided — duplicate check skipped._\n")

    md.append("## Stitching Quality Objectives\n")
    if objectives is not None:
        md.append("| Objective | Value |")
        md.append("|---|---|")
        md.append(f"| vial_count_error | {objectives['vial_count_error']:.1f} |")
        md.append(f"| per_id_coverage_loss | {objectives['per_id_coverage_loss']:.1f} frames/fly |")
        md.append(f"| short_track_count | {objectives['short_track_count']} |")
        md.append(f"| per_frame_id_variance | {objectives['per_frame_id_variance']:.3f} |")
        md.append("")
    else:
        md.append("_Not computed for this run._\n")

    md.append("## XY Trajectories\n")
    md.append("![XY Trajectories](metrics_xy_trajectories.png)\n")

    md.append("## Pipeline Diagnostics\n")
    md.append("![Pipeline Diagnostics](metrics_pipeline.png)\n")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Report saved to: {report_md}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _contiguous_segments(frames):
    """Given a sorted array of frame indices, return (starts, ends) of contiguous runs."""
    if len(frames) == 0:
        return [], []
    starts, ends = [frames[0]], []
    for i in range(1, len(frames)):
        if frames[i] > frames[i - 1] + 1:
            ends.append(frames[i - 1])
            starts.append(frames[i])
    ends.append(frames[-1])
    return starts, ends
