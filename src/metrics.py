"""
src/metrics.py

Quantitative metrics and diagnostics for the tracking pipeline.

Purpose
-------
After running the tracker, it is hard to know from the overlay video alone
whether a problem is caused by the detector or the tracker. This module
answers the question: at which stage is information being lost, and why?

It works in two stages, each pointing to different fixes:

  Stage 1 — Detector
    Are the right number of flies being detected each frame?
    Low here → labelling or confidence threshold changes perhaps

  Stage 2 — OCSort tracker
    Of those detections, how many became tracks? And how long do they last?
    With any short suppressed tracks → min_hits too high (takes 10 frames in a row to create a track but should be lower),
    or flies jumping a lot!
    Many IDs but high coverage → maybe matching problem (wrong asso_func or IoU threshold).

Main entry point
----------------
    run_diagnostics(tracker, df_wide, n_expected, fps)

Builds two composite Plotly figures (interactive hover) and optionally writes:
  metrics_report.html          — interactive, all figures embedded
  metrics_report.md            — markdown with static PNG references
  metrics_xy_trajectories.png  — side-by-side XY (raw vs ordered)
  metrics_pipeline.png         — 4-panel pipeline diagnostics
"""

import base64
import copy
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots


# Stable palette (20+ distinct colours, cycled for many-ID runs)
_PALETTE = pc.qualitative.Dark24


def _track_color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


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
# Stage 2.5 — Relinking comparison
# ---------------------------------------------------------------------------

def compute_relink_stats(df_wide, df_relinked):
    """
    Compare tracking quality before and after the relinking pass.

    Parameters
    ----------
    df_wide      : wide-format DataFrame (rows=frames, cols=track IDs)
    df_relinked  : long-format DataFrame with columns frame, x, y, original_id, relinked_id

    Returns
    -------
    dict with before/after statistics and swap counts.
    """
    id_cols = [c for c in df_wide.columns if c != "frame"]
    n_frames = len(df_wide)

    before_lengths = np.array([
        (df_wide[c].notna() & (df_wide[c] != "")).sum()
        for c in id_cols
    ], dtype=float)

    after_lengths = df_relinked.groupby("relinked_id")["frame"].count().values.astype(float)

    swapped = df_relinked[df_relinked["original_id"] != df_relinked["relinked_id"]]
    unique_swap_pairs = set(
        tuple(sorted([a, b]))
        for a, b in zip(swapped["original_id"], swapped["relinked_id"])
    )
    affected_ids = (
        set(swapped["original_id"].unique()) | set(swapped["relinked_id"].unique())
    )

    return {
        "n_ids_before":        len(id_cols),
        "n_ids_after":         int(df_relinked["relinked_id"].nunique()),
        "mean_length_before":  round(float(np.mean(before_lengths)), 1) if len(before_lengths) else 0.0,
        "median_length_before":round(float(np.median(before_lengths)), 1) if len(before_lengths) else 0.0,
        "min_length_before":   int(np.min(before_lengths)) if len(before_lengths) else 0,
        "max_length_before":   int(np.max(before_lengths)) if len(before_lengths) else 0,
        "mean_length_after":   round(float(np.mean(after_lengths)), 1) if len(after_lengths) else 0.0,
        "median_length_after": round(float(np.median(after_lengths)), 1) if len(after_lengths) else 0.0,
        "min_length_after":    int(np.min(after_lengths)) if len(after_lengths) else 0,
        "max_length_after":    int(np.max(after_lengths)) if len(after_lengths) else 0,
        "coverage_before":     round(float(np.mean(before_lengths / n_frames)) * 100, 1) if len(before_lengths) else 0.0,
        "coverage_after":      round(float(np.mean(after_lengths / n_frames)) * 100, 1) if len(after_lengths) else 0.0,
        "n_swaps":             len(unique_swap_pairs),
        "n_affected_tracks":   len(affected_ids),
    }


def print_relink_comparison(stats):
    """Print a before/after table for the relinking pass. Returns the text."""
    lines = []
    lines.append("=" * 55)
    lines.append("  RELINKING COMPARISON  (raw OC-SORT  ->  relinked)")
    lines.append("=" * 55)
    lines.append(f"  Swaps accepted          : {stats['n_swaps']}")
    lines.append(f"  Tracks affected         : {stats['n_affected_tracks']}")
    lines.append("")
    lines.append(f"  {'Metric':<26}{'Before':>8}  {'After':>8}  {'Delta':>8}")
    lines.append(f"  {'-'*26}{'-'*8}  {'-'*8}  {'-'*8}")

    def row(label, before, after, fmt=".1f"):
        delta = after - before
        sign  = "+" if delta > 0 else ""
        lines.append(
            f"  {label:<26}{before:>8{fmt}}  {after:>8{fmt}}  {sign}{delta:>{fmt}}"
        )

    row("Unique track IDs",      stats["n_ids_before"],        stats["n_ids_after"],        fmt="d")
    row("Mean track length (f)", stats["mean_length_before"],  stats["mean_length_after"])
    row("Median track length",   stats["median_length_before"],stats["median_length_after"])
    row("Min track length",      stats["min_length_before"],   stats["min_length_after"],   fmt="d")
    row("Max track length",      stats["max_length_before"],   stats["max_length_after"],   fmt="d")
    row("Mean ID coverage (%)",  stats["coverage_before"],     stats["coverage_after"])

    lines.append("")
    if stats["n_swaps"] == 0:
        lines.append("  No swaps were accepted — tracks unchanged.")
    else:
        lines.append(f"  {stats['n_swaps']} swap(s) corrected across "
                     f"{stats['n_affected_tracks']} track(s).")
        if stats["mean_length_after"] > stats["mean_length_before"]:
            lines.append("  Mean track length increased — relinking is helping.")
        else:
            lines.append("  Mean track length unchanged — swaps may be cosmetic.")
    lines.append("=" * 55)

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Plots (Plotly)
# ---------------------------------------------------------------------------

def _add_vial_shapes(fig, vial_rois, row=None, col=None):
    """Add grey ROI rectangles + labels behind trajectories."""
    if not vial_rois:
        return
    for label, (x1, y1, x2, y2) in vial_rois.items():
        fig.add_shape(
            type="rect", x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color="lightgrey", width=1),
            fillcolor="whitesmoke", layer="below",
            row=row, col=col,
        )
        fig.add_annotation(
            x=(x1 + x2) / 2, y=y1 - 8, text=label,
            showarrow=False, font=dict(size=9, color="grey"),
            xanchor="center", yanchor="bottom",
            row=row, col=col,
        )


def plot_xy_trajectories(df_wide, vial_rois=None, fig=None, row=None, col=None):
    """
    XY trajectory plot: one coloured line per emitted track ID.

    Each track is drawn as a path through (x, y) space, so you can see:
      - tracks that stay within one vial column  → good
      - tracks that jump across vials            → ID-switch bug
      - tracks that wander erratically           → matching/IoU problem

    Hovering a trajectory reveals the track ID and frame number for each point.

    Parameters
    ----------
    df_wide    : wide-format DataFrame (rows=frames, columns=track IDs)
    vial_rois  : optional dict  {vial_id: [x1, y1, x2, y2]}
    fig, row, col : optional parent figure + subplot cell
    """
    if fig is None:
        fig = go.Figure()

    _add_vial_shapes(fig, vial_rois, row=row, col=col)

    id_cols = [c for c in df_wide.columns if c != "frame"]
    for i, tid in enumerate(id_cols):
        sub = df_wide[df_wide[tid].notna() & (df_wide[tid] != "")][["frame", tid]].copy()
        if sub.empty:
            continue
        xy = sub[tid].str.strip("()").str.split(", ", expand=True).astype(float)
        xs = xy.iloc[:, 0].values
        ys = xy.iloc[:, 1].values
        frames = sub["frame"].values
        color = _track_color(i)
        customdata = np.stack([frames, np.full(len(frames), str(tid))], axis=-1)

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1),
                opacity=0.75,
                name=f"ID {tid}",
                legendgroup=f"raw-{tid}",
                customdata=customdata,
                hovertemplate=(
                    "ID %{customdata[1]}<br>"
                    "frame %{customdata[0]}<br>"
                    "(%{x:.1f}, %{y:.1f})<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )
        # start (circle) and end (x) markers
        fig.add_trace(
            go.Scatter(
                x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
                mode="markers+text",
                marker=dict(color=color, size=[8, 9], symbol=["circle", "x"]),
                text=[f"{tid}", ""],
                textposition="middle right",
                textfont=dict(size=8, color=color),
                legendgroup=f"raw-{tid}",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_xaxes(title_text="x (pixels)", row=row, col=col)
    fig.update_yaxes(title_text="y (pixels)", row=row, col=col)
    return fig


def plot_xy_trajectories_ordered(df_ordered, vial_rois=None, fig=None, row=None, col=None):
    """
    XY trajectory plot using ordered IDs (after vial assignment).

    Each line is one ordered_id, so you can verify that each ordered ID stays within its vial.
    """
    if fig is None:
        fig = go.Figure()

    _add_vial_shapes(fig, vial_rois, row=row, col=col)

    ordered_ids = sorted(df_ordered["ordered_id"].unique())
    has_vial = "vial_id" in df_ordered.columns
    for i, cid in enumerate(ordered_ids):
        sub = df_ordered[df_ordered["ordered_id"] == cid].sort_values("frame")
        if sub.empty:
            continue
        xs = sub["x"].values
        ys = sub["y"].values
        frames = sub["frame"].values
        vials = sub["vial_id"].values if has_vial else np.full(len(frames), "?")
        color = _track_color(i)
        customdata = np.stack(
            [frames, np.full(len(frames), str(cid)), np.asarray(vials, dtype=object)],
            axis=-1,
        )

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1),
                opacity=0.75,
                name=f"oID {cid}",
                legendgroup=f"ordered-{cid}",
                customdata=customdata,
                hovertemplate=(
                    "ordered %{customdata[1]}<br>"
                    "vial %{customdata[2]}<br>"
                    "frame %{customdata[0]}<br>"
                    "(%{x:.1f}, %{y:.1f})<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
                mode="markers+text",
                marker=dict(color=color, size=[8, 9], symbol=["circle", "x"]),
                text=[f"{cid}", ""],
                textposition="middle right",
                textfont=dict(size=8, color=color),
                legendgroup=f"ordered-{cid}",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_xaxes(title_text="x (pixels)", row=row, col=col)
    fig.update_yaxes(title_text="y (pixels)", row=row, col=col)
    return fig


def plot_xy_trajectories_relinked(df_relinked, vial_rois=None, fig=None, row=None, col=None):
    """
    XY trajectory plot using relinked IDs from the second-round re-linking pass.

    df_relinked must have columns: frame, x, y, original_id, relinked_id.
    Each line is one relinked_id.
    """
    if fig is None:
        fig = go.Figure()

    _add_vial_shapes(fig, vial_rois, row=row, col=col)

    relinked_ids = sorted(df_relinked["relinked_id"].unique())
    for i, rid in enumerate(relinked_ids):
        sub = df_relinked[df_relinked["relinked_id"] == rid].sort_values("frame")
        if sub.empty:
            continue
        xs = sub["x"].values
        ys = sub["y"].values
        frames = sub["frame"].values
        orig_ids = sub["original_id"].values
        color = _track_color(i)
        customdata = np.stack(
            [frames, np.full(len(frames), str(rid)), np.asarray(orig_ids, dtype=object)],
            axis=-1,
        )

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=1),
                opacity=0.75,
                name=f"rID {rid}",
                legendgroup=f"relinked-{rid}",
                customdata=customdata,
                hovertemplate=(
                    "relinked %{customdata[1]}<br>"
                    "orig %{customdata[2]}<br>"
                    "frame %{customdata[0]}<br>"
                    "(%{x:.1f}, %{y:.1f})<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[xs[0], xs[-1]], y=[ys[0], ys[-1]],
                mode="markers+text",
                marker=dict(color=color, size=[8, 9], symbol=["circle", "x"]),
                text=[f"{rid}", ""],
                textposition="middle right",
                textfont=dict(size=8, color=color),
                legendgroup=f"relinked-{rid}",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_xaxes(title_text="x (pixels)", row=row, col=col)
    fig.update_yaxes(title_text="y (pixels)", row=row, col=col)
    return fig


def plot_detection_log(tracker, fps=30, fig=None, row=None, col=None):
    """
    Line plot: raw detections vs emitted tracks per frame.

    The shaded area between the two lines is the signal suppressed by min_hits
    or lost in association.
    """
    if fig is None:
        fig = go.Figure()

    log = np.array(tracker.detection_log)
    frames    = log[:, 0]
    n_dets    = log[:, 1]
    n_emitted = log[:, 2]

    # emitted first, then detections with fill='tonexty' → fills between them
    fig.add_trace(
        go.Scatter(
            x=frames, y=n_emitted, mode="lines",
            line=dict(color="darkorange", width=1.4),
            name="emitted tracks",
            hovertemplate="frame %{x}<br>emitted %{y}<extra></extra>",
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=frames, y=n_dets, mode="lines",
            line=dict(color="steelblue", width=1.4),
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.15)",
            name="detections (RF-DETR)",
            hovertemplate="frame %{x}<br>detections %{y}<extra></extra>",
        ),
        row=row, col=col,
    )
    fig.update_xaxes(title_text="Frame", row=row, col=col)
    fig.update_yaxes(title_text="Count", row=row, col=col)
    return fig


def plot_tracklet_timeline(df_wide, suppressed_tracks=None, fig=None, row=None, col=None):
    """
    Tracklet timeline: one horizontal bar per track segment.
    Emitted tracks (in CSV) are blue. Suppressed tracks are red if provided.

    Gaps in emitted tracks are visible as breaks in the bars — these are the
    fragments stitching has to repair. Hover a bar to see the track ID and
    frame range.
    """
    if fig is None:
        fig = go.Figure()

    id_cols = [c for c in df_wide.columns if c != "frame"]
    n_frames = len(df_wide)

    emitted_x, emitted_y, emitted_base, emitted_cd = [], [], [], []
    for y, col_id in enumerate(id_cols):
        present = df_wide[col_id].notna() & (df_wide[col_id] != "")
        frames_present = df_wide.loc[present, "frame"].values
        if len(frames_present) == 0:
            continue
        starts, ends = _contiguous_segments(frames_present)
        for s, e in zip(starts, ends):
            emitted_x.append(e - s + 1)
            emitted_y.append(y)
            emitted_base.append(s)
            emitted_cd.append([str(col_id), int(s), int(e)])

    fig.add_trace(
        go.Bar(
            x=emitted_x, y=emitted_y, base=emitted_base,
            orientation="h",
            marker=dict(color="steelblue", line=dict(width=0)),
            opacity=0.85,
            name="emitted tracks",
            customdata=emitted_cd,
            hovertemplate=(
                "ID %{customdata[0]}<br>"
                "frames %{customdata[1]}–%{customdata[2]}<extra></extra>"
            ),
        ),
        row=row, col=col,
    )

    sup_tick_vals, sup_tick_text = [], []
    if suppressed_tracks:
        offset = len(id_cols) + 1
        sup_x, sup_y, sup_base, sup_cd = [], [], [], []
        for y, trk in enumerate(suppressed_tracks):
            frames = [xy[0] for xy in trk["xy"]]
            if not frames:
                continue
            label = f"sup-{y}"
            sup_tick_vals.append(offset + y)
            sup_tick_text.append(label)
            starts, ends = _contiguous_segments(np.array(frames))
            for s, e in zip(starts, ends):
                sup_x.append(e - s + 1)
                sup_y.append(offset + y)
                sup_base.append(int(s))
                sup_cd.append([label, int(s), int(e)])

        fig.add_trace(
            go.Bar(
                x=sup_x, y=sup_y, base=sup_base,
                orientation="h",
                marker=dict(color="tomato", line=dict(width=0)),
                opacity=0.80,
                name="suppressed tracks",
                customdata=sup_cd,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "frames %{customdata[1]}–%{customdata[2]}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    tick_vals = list(range(len(id_cols))) + sup_tick_vals
    tick_text = [str(c) for c in id_cols] + sup_tick_text
    fig.update_xaxes(title_text="Frame", range=[0, n_frames], row=row, col=col)
    fig.update_yaxes(
        title_text="Track",
        tickmode="array", tickvals=tick_vals, ticktext=tick_text,
        row=row, col=col,
    )
    return fig


def plot_suppressed_histogram(tracker, fig=None, row=None, col=None):
    """
    Histogram of suppressed track lengths (hit counts).
    A dashed line marks the min_hits threshold.
    """
    if fig is None:
        fig = go.Figure()

    sup = tracker.suppressed_tracks
    if not sup:
        return fig

    hits = [s["hits"] for s in sup]
    fig.add_trace(
        go.Histogram(
            x=hits,
            xbins=dict(start=0.5, end=tracker.min_hits + 1.5, size=1),
            marker=dict(color="tomato", line=dict(color="white", width=1)),
            opacity=0.85,
            name="suppressed",
            hovertemplate="hits %{x}<br>count %{y}<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.add_vline(
        x=tracker.min_hits, line_dash="dash", line_color="black", line_width=1.2,
        annotation_text=f"min_hits = {tracker.min_hits}",
        annotation_position="top right",
        row=row, col=col,
    )
    fig.update_xaxes(title_text="Hits before track was suppressed", row=row, col=col)
    fig.update_yaxes(title_text="Number of tracks", row=row, col=col)
    return fig


def plot_coverage_comparison(df_wide, fig=None, row=None, col=None):
    """
    Bar chart: mean frame coverage per track ID.
    """
    if fig is None:
        fig = go.Figure()

    id_cols_wide = [c for c in df_wide.columns if c != "frame"]
    cov_before = np.mean(
        [(df_wide[c].notna() & (df_wide[c] != "")).mean() for c in id_cols_wide]
    ) * 100 if id_cols_wide else 0.0

    labels = ["OC-SORT tracks"]
    values = [cov_before]
    colors = ["steelblue"]

    fig.add_trace(
        go.Bar(
            x=labels, y=values,
            marker=dict(color=colors, line=dict(color="white", width=1)),
            opacity=0.9,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.update_xaxes(row=row, col=col)
    fig.update_yaxes(title_text="Mean frame coverage per ID (%)", range=[0, 110], row=row, col=col)

    fig.add_annotation(
        text=f"IDs: {len(id_cols_wide)}  |  Coverage: {cov_before:.1f}%",
        showarrow=False,
        xref=f"x{'' if (row is None or col is None) else f' domain'}",
        yref=f"y{'' if (row is None or col is None) else f' domain'}",
        x=0.5, y=-0.25,
        xanchor="center", yanchor="top",
        font=dict(size=10, color="gray"),
        row=row, col=col,
    )

    return fig


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
# Composite figure builders
# ---------------------------------------------------------------------------

def _build_xy_figure(df_wide, df_ordered, vial_rois, df_relinked=None):
    """Build the XY trajectory composite figure (1×1, 1×2, or 1×3)."""
    panels = [("Raw OC-SORT IDs", lambda fig, r, c: plot_xy_trajectories(
        df_wide, vial_rois=vial_rois, fig=fig, row=r, col=c))]

    if df_relinked is not None:
        panels.append(("Relinked IDs", lambda fig, r, c: plot_xy_trajectories_relinked(
            df_relinked, vial_rois=vial_rois, fig=fig, row=r, col=c)))

    if df_ordered is not None:
        panels.append(("Ordered track IDs", lambda fig, r, c: plot_xy_trajectories_ordered(
            df_ordered, vial_rois=vial_rois, fig=fig, row=r, col=c)))

    n_cols = len(panels)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.06,
    )
    for col_i, (_, plot_fn) in enumerate(panels, start=1):
        plot_fn(fig, 1, col_i)
        fig.update_yaxes(autorange="reversed", row=1, col=col_i)
        scaleanchor = "y" if col_i == 1 else f"y{col_i}"
        fig.update_xaxes(scaleanchor=scaleanchor, scaleratio=1, row=1, col=col_i)

    title_parts = [p[0] for p in panels]
    title = "XY trajectories: " + " | ".join(title_parts) if n_cols > 1 else "XY trajectories (○=start ×=end)"
    width = 800 * n_cols

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13)),
        height=700, width=width,
        hovermode="closest",
        margin=dict(l=60, r=30, t=70, b=50),
    )
    return fig


def _build_pipeline_figure(tracker, df_wide, fps):
    """Build the 4-row pipeline diagnostics figure."""
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[2/10, 4/10, 2/10, 2/10],
        vertical_spacing=0.08,
        subplot_titles=(
            "Stage 1→2: Detections vs emitted tracks per frame",
            "Stage 2: Tracklet timeline (blue=emitted, red=suppressed)",
            "Stage 2: Suppressed track length distribution",
            "Stage 2: Mean frame coverage per ID",
        ),
    )
    plot_detection_log(tracker, fps=fps, fig=fig, row=1, col=1)
    plot_tracklet_timeline(df_wide, suppressed_tracks=tracker.suppressed_tracks, fig=fig, row=2, col=1)
    plot_suppressed_histogram(tracker, fig=fig, row=3, col=1)
    plot_coverage_comparison(df_wide, fig=fig, row=4, col=1)

    fig.update_layout(
        title=dict(text="Tracking pipeline diagnostics", x=0.5, xanchor="center",
                   font=dict(size=13)),
        height=1400, width=1400,
        hovermode="closest",
        barmode="overlay",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(tracker, df_wide, df_ordered=None,
                    df_relinked=None,
                    n_expected=None, fps=30, vial_rois=None,
                    config=None, output_dir=None,
                    show_plots=True):
    """
    Run all diagnostics, display interactive Plotly figures, and optionally save a report.

    Parameters
    ----------
    tracker    : OCSort object returned by export_tracks_xy_tuple_csv_one_config
    df_wide    : wide-format OC-SORT tracks DataFrame
    df_ordered : ordered_tracks DataFrame (optional, shows post-tracking trajectories)
    n_expected : total number of flies expected (e.g. n_vials × flies_per_vial)
    fps        : video frame rate (used for axis labels)
    vial_rois  : optional dict {vial_id: [x1, y1, x2, y2]} for XY plot background
    config     : optional dict — the full config used for this run
    output_dir : optional path — if given, saves HTML + markdown + PNGs inside that directory
    show_plots : bool, default True — if False, skip Plotly ``.show()`` (unattended batch).
    """
    cfg_report = _config_for_report(config)
    if cfg_report is not None:
        print("=" * 55)
        print("  RUN CONFIGURATION")
        print("=" * 55)
        print(json.dumps(cfg_report, indent=2, default=str))
        print("=" * 55)
        print()

    stats = compute_tracker_stats(tracker, df_wide)
    summary_text = print_tracker_summary(stats, tracker, n_expected)

    relink_text = ""
    if df_relinked is not None:
        relink_stats = compute_relink_stats(df_wide, df_relinked)
        relink_text  = print_relink_comparison(relink_stats)

    fig_xy       = _build_xy_figure(df_wide, df_ordered, vial_rois, df_relinked=df_relinked)
    fig_pipeline = _build_pipeline_figure(tracker, df_wide, fps)

    if output_dir is not None:
        _save_report(output_dir, summary_text, cfg_report, fig_xy, fig_pipeline,
                     relink_text=relink_text)

    if show_plots:
        fig_xy.show()
        fig_pipeline.show()


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _config_for_report(config):
    """Copy of config for HTML/MD/console: drop inference endpoint noise."""
    if config is None:
        return None
    c = copy.deepcopy(config)
    rf = c.get("roboflow")
    if isinstance(rf, dict):
        rf = {k: v for k, v in rf.items() if k != "inference_api_url"}
        if rf:
            c["roboflow"] = rf
        else:
            c.pop("roboflow", None)
    return c


def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def _image_data_uri(img_path: Path) -> str:
    suffix = img_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _html_scalar(value) -> str:
    if value is None:
        return '<span class="config-empty">null</span>'
    if isinstance(value, bool):
        return "true" if value else "false"
    return _html_escape(str(value))


def _config_type_label(value) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if value is None:
        return "null"
    return type(value).__name__


def _config_table_rows(value, depth=0, label=None) -> str:
    rows = []
    indent = 16 + depth * 18
    label_html = _html_escape("" if label is None else str(label))

    if isinstance(value, dict):
        if label is not None:
            rows.append(
                f'<tr class="config-group-row"><td class="config-group" colspan="3" '
                f'style="padding-left:{indent}px">{label_html}</td></tr>'
            )
        child_depth = depth + (1 if label is not None else 0)
        for key, child in value.items():
            rows.append(_config_table_rows(child, child_depth, key))
        return "".join(rows)

    if isinstance(value, list):
        if value and any(isinstance(item, (dict, list)) for item in value):
            rows.append(
                f'<tr class="config-group-row"><td class="config-group" colspan="3" '
                f'style="padding-left:{indent}px">{label_html}'
                f'<span class="config-meta">{_config_type_label(value)}</span></td></tr>'
            )
            child_depth = depth + 1
            child_indent = 16 + child_depth * 18
            for idx, item in enumerate(value):
                child_label = f"[{idx}]"
                if isinstance(item, (dict, list)):
                    rows.append(_config_table_rows(item, child_depth, child_label))
                else:
                    rows.append(
                        "<tr>"
                        f'<td class="config-key" style="padding-left:{child_indent}px">{_html_escape(child_label)}</td>'
                        f'<td class="config-value">{_html_scalar(item)}</td>'
                        f'<td class="config-type">{_config_type_label(item)}</td>'
                        "</tr>"
                    )
            return "".join(rows)

        scalar_items = ", ".join(_html_escape(str(item)) for item in value)
        if not scalar_items:
            scalar_items = '<span class="config-empty">empty</span>'
        return (
            "<tr>"
            f'<td class="config-key" style="padding-left:{indent}px">{label_html}</td>'
            f'<td class="config-value">{scalar_items}</td>'
            f'<td class="config-type">{_config_type_label(value)}</td>'
            "</tr>"
        )

    return (
        "<tr>"
        f'<td class="config-key" style="padding-left:{indent}px">{label_html}</td>'
        f'<td class="config-value">{_html_scalar(value)}</td>'
        f'<td class="config-type">{_config_type_label(value)}</td>'
        "</tr>"
    )


def _config_table_html(config) -> str:
    if config is None:
        return ""
    rows = _config_table_rows(config)
    return (
        '<details open><summary>Configuration</summary>'
        '<table class="config-table">'
        '<thead><tr><th>Section / Key</th><th>Value</th><th>Type</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></details>"
    )


def _dup_stats_html(dup_stats) -> str:
    if dup_stats is None:
        return "<p><em>Stitched output not provided — duplicate check skipped.</em></p>"
    n_ids    = dup_stats["n_duplicate_ids"]
    n_frames = dup_stats["n_duplicate_frames"]
    if n_ids == 0:
        return "<p><strong>0 duplicate (frame, stitched_id) pairs — clean.</strong></p>"
    rows = "".join(
        f"<tr><td>{d['stitched_id']}</td><td>{d['vial_id']}</td><td>{d['n_frames']}</td></tr>"
        for d in dup_stats["details"]
    )
    return (
        f"<p><strong>{n_ids} stitched_id(s) had duplicates across {n_frames} frame(s).</strong></p>"
        f"<table><thead><tr><th>stitched_id</th><th>vial</th><th>affected frames</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _funny_image_html() -> str:
    img_path = Path(__file__).parent.parent / "data" / "media" / "ty4ids_flies.png"
    if not img_path.exists():
        return ""
    return (
        '<hr><div style="text-align:center;margin-top:24px;">'
        f'<img src="{_image_data_uri(img_path)}" '
        'style="max-width:480px;width:100%;height:auto;" alt="ty4ids flies"></div>'
    )


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HERO_VIDEO_SRC = _REPO_ROOT / "data" / "media" / "funny" / "tsa_line_video.mp4"
_HERO_VIDEO_NAME = _HERO_VIDEO_SRC.name


def _ensure_hero_video(output_dir: Path) -> Path | None:
    """Copy hero MP4 beside metrics_report.html so relative src works when opened locally."""
    if not _HERO_VIDEO_SRC.is_file():
        return None
    dest = output_dir / _HERO_VIDEO_NAME
    shutil.copy2(_HERO_VIDEO_SRC, dest)
    return dest


def _landing_page_hero_html(hero_video: Path | None) -> str:
    if hero_video is None or not hero_video.is_file():
        return ""
    name = _html_escape(hero_video.name)
    return (
        '<div class="report-hero">'
        '<video class="report-hero-video" controls muted playsinline loop autoplay>'
        f'<source src="{name}" type="video/mp4">'
        "</video></div>"
    )


def _random_funny_image_html() -> str:
    funny_dir = Path(__file__).parent.parent / "data" / "media" / "funny"
    if not funny_dir.exists():
        return ""
    candidates = [
        path for path in funny_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    ]
    if not candidates:
        return ""
    img_path = random.choice(candidates)
    alt = _html_escape(img_path.stem.replace("_", " "))
    return (
        '<div class="report-funny-image">'
        f'<img src="{_image_data_uri(img_path)}" alt="{alt}"></div>'
    )


def _objectives_html(objectives) -> str:
    if objectives is None:
        return "<p><em>Not computed for this run.</em></p>"
    return (
        "<table><thead><tr><th>Objective</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>vial_count_error</td><td>{objectives['vial_count_error']:.1f}</td></tr>"
        f"<tr><td>per_id_coverage_loss</td><td>{objectives['per_id_coverage_loss']:.1f} frames/fly</td></tr>"
        f"<tr><td>short_track_count</td><td>{objectives['short_track_count']}</td></tr>"
        f"<tr><td>per_frame_id_variance</td><td>{objectives['per_frame_id_variance']:.3f}</td></tr>"
        "</tbody></table>"
    )


_HTML_STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         max-width: 1700px; margin: 2em auto; padding: 0 1em; color: #222; }
  h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2em; }
  pre { background: #f6f6f6; padding: 0.8em; border-radius: 4px; overflow-x: auto;
        font-size: 12px; line-height: 1.3; }
  details { margin: 0.8em 0; }
  summary { cursor: pointer; font-weight: 600; }
  table { border-collapse: collapse; margin: 0.5em 0; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 4px 10px; font-size: 13px; text-align: left; }
  th { background: #eee; }
  .report-hero { margin: 0 0 1.5em; }
  .report-hero img { display: block; width: 100%; height: auto; border-radius: 8px; }
  .report-hero-video { display: block; width: 100%; height: auto; border-radius: 8px; }
  .report-two-col { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 0.5em 0 1.5em; }
  .report-two-col td, .report-two-col th { border: none; padding: 0; vertical-align: top; }
  .report-two-col .report-left-col { width: 68%; padding-right: 24px; }
  .report-two-col .report-right-col { width: 32%; }
  .report-funny-image { width: 100%; padding-top: 2.3em; }
  .report-funny-image img { display: block; width: 100%; height: auto; max-width: 420px; margin: 0 auto; }
  .config-table { table-layout: fixed; }
  .config-table th:nth-child(1) { width: 36%; }
  .config-table th:nth-child(2) { width: 48%; }
  .config-table th:nth-child(3) { width: 16%; }
  .config-group-row td { background: #f4f6f8; font-weight: 600; border-top: 2px solid #d8dde3; }
  .config-group { color: #1f2937; }
  .config-key { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }
  .config-value { word-break: break-word; }
  .config-type { color: #5b6470; white-space: nowrap; }
  .config-meta { margin-left: 8px; font-size: 12px; font-weight: 500; color: #5b6470; }
  .config-empty { color: #7a7a7a; font-style: italic; }
</style>
"""


def _save_report(output_dir, summary_text, config, fig_xy, fig_pipeline,
                 relink_text=""):
    """
    Write:
      metrics_report.html          — interactive, all figures embedded
      metrics_report.md            — markdown with PNG refs
      metrics_xy_trajectories.png  — static XY plot
      metrics_pipeline.png         — static pipeline plot
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)
    hero_video_path = _ensure_hero_video(out_path)

    xy_png       = os.path.join(output_dir, "metrics_xy_trajectories.png")
    pipeline_png = os.path.join(output_dir, "metrics_pipeline.png")
    report_md    = os.path.join(output_dir, "metrics_report.md")
    report_html  = os.path.join(output_dir, "metrics_report.html")

    # Static PNG exports (kaleido)
    try:
        fig_xy.write_image(xy_png,             width=1600, height=700,  scale=2)
        fig_pipeline.write_image(pipeline_png, width=1400, height=1400, scale=2)
    except Exception as exc:
        print(f"[metrics] PNG export failed ({exc}); install kaleido to enable.")

    # Interactive HTML divs — load plotly.js from CDN once (first fig only)
    xy_div       = fig_xy.to_html(include_plotlyjs="cdn",  full_html=False,
                                  div_id="fig-xy")
    pipeline_div = fig_pipeline.to_html(include_plotlyjs=False, full_html=False,
                                        div_id="fig-pipeline")

    config_block = _config_table_html(config)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Metrics Report</title>{_HTML_STYLE}</head>
<body>
{_landing_page_hero_html(hero_video_path)}

{config_block}

<table class="report-two-col" role="presentation">
<tr>
<td class="report-left-col">
<h2>Tracker Summary</h2>
<pre>{_html_escape(summary_text)}</pre>

{f'<h2>Relinking Comparison</h2><pre>{_html_escape(relink_text)}</pre>' if relink_text else ''}
</td>
<td class="report-right-col">
{_random_funny_image_html()}
</td>
</tr>
</table>

<h2>XY Trajectories</h2>
{xy_div}

<h2>Pipeline Diagnostics</h2>
{pipeline_div}

{_funny_image_html()}

</body></html>
"""
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(html)

    # Markdown (static PNG refs) — mirrors the HTML structure
    md = ["# Metrics Report\n",
          f"> Interactive version: [metrics_report.html](metrics_report.html)\n"]

    if config is not None:
        md.append("## Configuration\n")
        md.append("```json")
        md.append(json.dumps(config, indent=2, default=str))
        md.append("```\n")

    md.append("## Tracker Summary\n")
    md.append("```")
    md.append(summary_text)
    md.append("```\n")

    if relink_text:
        md.append("## Relinking Comparison\n")
        md.append("```")
        md.append(relink_text)
        md.append("```\n")

    md.append("## XY Trajectories\n")
    md.append("![XY Trajectories](metrics_xy_trajectories.png)\n")

    md.append("## Pipeline Diagnostics\n")
    md.append("![Pipeline Diagnostics](metrics_pipeline.png)\n")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Report saved: {report_html}")
    print(f"           +  {report_md}")


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
