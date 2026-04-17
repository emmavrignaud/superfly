#!/usr/bin/env python
"""
scripts/run_grid_search.py

Grid search over stitching weights for a single cached wide CSV.
Tracking is never re-run — only stitching is repeated for each config.

Search space
------------
  link_score_weights  : extrap / direction / behavioral, steps of 0.2, sum-to-1
  direction_weights   : heading_vs_gap / overall_vs_overall, binary {0, 1}, exclude (0,0)
  behavioral_weights  : 7 features, binary {0, 1}, exclude all-zeros

Objectives (all lower-is-better)
-----------
  vial_count_error       sum_vials |n_stitched - expected_per_vial|
  per_id_coverage_loss   num_frames - mean(stitched_track_length)
  short_track_count      tracks shorter than short_frac * num_frames
  per_frame_id_variance  sum_vials std(id_count_at_frame_t)

Outputs (in outputs\\grid_search\\<short_name>\\)
-------
  grid_search_results.csv   one row per config; all weights + all objectives
  grid_search_plot.html     interactive Plotly line plot

Usage
-----
  python scripts\\run_grid_search.py ^
      --wide-csv  outputs\\run_5\\tracks_wide_format.csv ^
      --roi-json  outputs\\run_5\\vial_rois.json

  # re-plot from an existing results CSV without re-running the search:
  python scripts\\run_grid_search.py ^
      --wide-csv  outputs\\run_5\\tracks_wide_format.csv ^
      --roi-json  outputs\\run_5\\vial_rois.json ^
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

# Add the repo root to the Python path so we can import from src/ and utils.py
# regardless of which directory the script is called from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stitching import wide_to_long, build_tracklets, stitch, link_score
from src.metrics import compute_stitching_objectives


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# These must exactly match the key names in config.yaml under stitching:
_DIR_KEYS = ["heading_vs_gap", "overall_vs_overall"]

_BEH_KEYS = [
    "median_velocity",
    "pause_fraction",
    "mean_turning_angle",
    "mean_angular_velocity",
    "mean_acceleration",
    "n_large_displacements",
    "tortuosity",
]


# ---------------------------------------------------------------------------
# Search space builders
# ---------------------------------------------------------------------------

def _link_score_grid(step: float = 0.2):
    """
    Generate all (extrap, direction, behavioral) weight triples that sum to 1,
    where each value is a multiple of `step`.

    Why sum-to-1? Because the three terms are already on compatible scales after
    their individual normalisations in link_score(). Constraining the sum to 1
    means we're choosing a mixture, not an arbitrary scale — which keeps the
    search interpretable and reduces the space from 3D to 2D (the simplex).

    Example with step=0.2: produces (0.0, 0.0, 1.0), (0.0, 0.2, 0.8), ..., (1.0, 0.0, 0.0)
    — 21 combinations total.

    The `step/2` offset in arange ensures 1.0 is included despite floating-point
    rounding (np.arange(0, 1.0, 0.2) sometimes stops at 0.8 without it).
    The -1e-9 / +1e-9 tolerance on b handles the same floating-point issue when
    checking whether the computed third weight lands exactly in [0, 1].
    """
    vals = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    combos = []
    for e in vals:
        for d in vals:
            # Third weight is fully determined once the first two are chosen
            b = round(1.0 - float(e) - float(d), 10)
            if -1e-9 <= b <= 1.0 + 1e-9:
                combos.append((round(float(e), 2), round(float(d), 2), round(float(b), 2)))
    return combos


def _binary_combos(keys: list, exclude_all_zero: bool = True) -> list:
    """
    Generate all binary {0, 1} combinations for the given keys as a list of dicts.

    itertools.product([0, 1], repeat=n) generates all 2^n binary strings of length n.
    For example, with keys=["a", "b"]: [(0,0), (0,1), (1,0), (1,1)].
    We zip each tuple with the key names to get {"a": 0, "b": 1} etc.

    Why exclude all-zeros? A config where all behavioral weights are 0 means
    the behavioral term contributes nothing — the score degenerates. Similarly
    for direction weights. These are degenerate configs, not meaningful.

    For _BEH_KEYS (7 features): 2^7 - 1 = 127 meaningful combinations.
    For _DIR_KEYS (2 features): 2^2 - 1 = 3 meaningful combinations.
    """
    raw = list(itertools.product([0, 1], repeat=len(keys)))
    if exclude_all_zero:
        raw = [c for c in raw if any(c)]
    return [dict(zip(keys, c)) for c in raw]


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _config_key(d: dict) -> str:
    """
    Build a stable string that uniquely identifies a weight configuration.

    Used to check whether a config has already been evaluated (resume support).
    We only include the weight columns — not the objective columns — so this
    works both when called on a freshly-built row_meta dict (no objectives yet)
    and when called on a row read back from the results CSV (which has objectives).

    Example output: "extrap=0.4|direction=0.4|behavioral=0.2|dir_heading_vs_gap=1|..."
    """
    weight_cols = (
        ["extrap", "direction", "behavioral"]
        + [f"dir_{k}" for k in _DIR_KEYS]
        + [f"beh_{k}" for k in _BEH_KEYS]
    )
    return "|".join(f"{k}={d[k]}" for k in weight_cols if k in d)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results_df: pd.DataFrame, short_name: str, out_html: str) -> None:
    """
    Plotly line plot: configs on x-axis sorted by vial_count_error (primary objective),
    one line per objective showing raw values.
    """
    # Baseline row (pre-stitch) pinned to x=0; configs sorted by vial_count_error after it.
    baseline = results_df[results_df["extrap"].isna()].copy()
    configs  = results_df[results_df["extrap"].notna()].sort_values("vial_count_error").reset_index(drop=True)
    df = pd.concat([baseline, configs], ignore_index=True)

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

    def _hover(i: int, row: pd.Series) -> str:
        if pd.isna(row["extrap"]):
            return (
                f"<b>PRE-STITCH (no stitching)</b><br>"
                f"count_err={row['vial_count_error']:.1f}  "
                f"cov_loss={row['per_id_coverage_loss']:.1f}  "
                f"short={row['short_track_count']}  "
                f"id_var={row['per_frame_id_variance']:.3f}"
            )
        return (
            f"rank={i}<br>"
            f"link: extrap={row['extrap']}  dir={row['direction']}  beh={row['behavioral']}<br>"
            f"dir: h_gap={row['dir_heading_vs_gap']}  overall={row['dir_overall_vs_overall']}<br>"
            f"beh: vel={row['beh_median_velocity']}  pause={row['beh_pause_fraction']}  "
            f"turn={row['beh_mean_turning_angle']}  ang_vel={row['beh_mean_angular_velocity']}  "
            f"accel={row['beh_mean_acceleration']}  n_ld={row['beh_n_large_displacements']}  "
            f"tort={row['beh_tortuosity']}<br>"
            f"count_err={row['vial_count_error']:.1f}  "
            f"cov_loss={row['per_id_coverage_loss']:.1f}  "
            f"short={row['short_track_count']}  "
            f"id_var={row['per_frame_id_variance']:.3f}"
        )

    fig = go.Figure()

    for col in obj_cols:
        hover = [_hover(i, row) for i, row in df.iterrows()]
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df[col],
            mode="lines",
            name=labels[col],
            line=dict(color=colors[col], width=1.5),
            hovertext=hover,
            hoverinfo="text+name",
        ))

    fig.update_layout(
        title=(
            f"Stitching weight grid search — {short_name}<br>"
            f"<sup>x=0 is pre-stitch baseline; x>0 sorted by vial_count_error (ascending).  "
            f"y-axis: raw objective values (lower is better)</sup>"
        ),
        xaxis_title="Config rank (x=0: pre-stitch baseline; x>0: sorted by vial_count_error)",
        yaxis_title="Objective value (lower is better)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        template="plotly_white",
    )

    fig.write_html(out_html)
    print(f"Plot saved: {out_html}")


# ---------------------------------------------------------------------------
# Short-name extraction
# ---------------------------------------------------------------------------

def _parse_short_name(wide_csv: str) -> str:
    """
    Extract the human-readable video label from run_params.json.

    run_params.json lives in the same directory as the wide CSV and was written
    by the notebook/run_tracking.py. It stores the original video path under
    config > video. We parse the filename stem to find the age-in-days and
    experiment number, e.g. "..._31d_004-converted" → "31d_n004".

    Falls back to the run directory name (e.g. "run_5") if run_params.json
    is missing or the pattern is not found.
    """
    run_dir = Path(wide_csv).parent
    params_path = run_dir / "run_params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        video = params.get("config", {}).get("video", "")
        stem = Path(video).stem
        m = re.search(r"_(\d+d)_(\d{3})", stem)
        if m:
            return f"{m.group(1)}_n{m.group(2)}"
    return run_dir.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over stitching weights (caches wide CSV, re-runs only stitching)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wide-csv",          required=True,            help="Path to tracks_wide_format.csv")
    parser.add_argument("--roi-json",          required=True,            help="Path to vial_rois.json")
    parser.add_argument("--expected-per-vial", type=int,   default=7,    help="Expected flies per vial (for obj2)")
    parser.add_argument("--short-frac",        type=float, default=0.10, help="Short track threshold as fraction of num_frames (for obj4)")
    parser.add_argument("--step",              type=float, default=0.2,  help="link_score_weights grid step")
    parser.add_argument("--plot-only",         action="store_true",      help="Skip search, re-plot from existing results CSV")
    parser.add_argument("--output-dir",        default=None,             help="Use this folder directly (skips auto-increment; required for --plot-only on a specific run)")
    args = parser.parse_args()

    short_name = _parse_short_name(args.wide_csv)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        _base    = Path("outputs") / "grid_search" / short_name
        out_dir  = _base
        _counter = 2
        while out_dir.exists():
            out_dir  = _base.parent / f"{_base.name}_{_counter}"
            _counter += 1
        out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "grid_search_results.csv"
    plot_html   = out_dir / "grid_search_plot.html"

    if args.plot_only:
        if not results_csv.exists():
            print(f"No results file found at {results_csv}. Run without --plot-only first.")
            sys.exit(1)
        plot_results(pd.read_csv(results_csv), short_name, str(plot_html))
        return

    # --- Load inputs ---
    print(f"Video label  : {short_name}")
    print(f"Wide CSV     : {args.wide_csv}")
    print(f"ROI JSON     : {args.roi_json}")
    print(f"Output dir   : {out_dir}\n")

    df_wide    = pd.read_csv(args.wide_csv)
    num_frames = int(df_wide["frame"].max()) + 1

    with open(args.roi_json) as f:
        vial_rois = {k: tuple(v) for k, v in json.load(f).items()}

    # Build long format and tracklets ONCE, before the grid search loop.
    # This is the expensive part — feature extraction, kinematic computation, etc.
    # Tracklets only depend on (x, y, frame) from the wide CSV, not on any
    # stitching weight. So we compute them once and pass the same list into
    # every stitch() call throughout the loop.
    print("Building long format and tracklets (once)...")
    long_df   = wide_to_long(df_wide)
    tracklets = build_tracklets(long_df)
    print(f"  {len(tracklets)} tracklets  |  {num_frames} frames\n")

    # --- Weights injection check ---
    ordered = sorted(tracklets, key=lambda t: t.start_frame)
    _A = ordered[0]
    _B = next((t for t in ordered[1:] if t.start_frame > _A.end_frame), None)
    if _B is not None:
        _dummy_dir = {"heading_vs_gap": 1, "overall_vs_overall": 0}
        _dummy_beh = {k: 1 for k in _BEH_KEYS}
        _w1 = {"link_score_weights": {"extrap": 1.0, "direction": 0.0, "behavioral": 0.0},
               "direction_weights": _dummy_dir, "behavioral_weights": _dummy_beh}
        _w2 = {"link_score_weights": {"extrap": 0.0, "direction": 0.0, "behavioral": 1.0},
               "direction_weights": _dummy_dir, "behavioral_weights": _dummy_beh}
        _s1 = link_score(_A, _B, vial_rois, tracklets, _w1)
        _s2 = link_score(_A, _B, vial_rois, tracklets, _w2)
        print(f"Weights check — same pair, two extreme configs:")
        print(f"  extrap=1 dir=0 beh=0  →  score = {_s1:.4f}")
        print(f"  extrap=0 dir=0 beh=1  →  score = {_s2:.4f}")
        if abs(_s1 - _s2) < 1e-9:
            print("  WARNING: scores are identical — weights are not taking effect.")
        else:
            print("  OK: scores differ.")
    else:
        print("  Weights check skipped — no non-overlapping tracklet pair found.")
    print()

    # --- Build search space ---
    link_combos = _link_score_grid(args.step)
    dir_combos  = _binary_combos(_DIR_KEYS, exclude_all_zero=True)
    beh_combos  = _binary_combos(_BEH_KEYS, exclude_all_zero=True)

    # When d=0 the direction term is zeroed out, making all dir_combos equivalent.
    # When b=0 the behavioral term is zeroed out, making all beh_combos equivalent.
    # In those cases we only run one representative combo instead of all of them.
    total = sum(
        (len(dir_combos) if d > 0 else 1) * (len(beh_combos) if b > 0 else 1)
        for _, d, b in link_combos
    )

    print(
        f"Search space : {len(link_combos)} link × "
        f"{len(dir_combos)} direction × "
        f"{len(beh_combos)} behavioral = {total} effective configs "
        f"(degenerate configs skipped)\n"
    )

    n_before = long_df["orig_id"].nunique()

    # --- Pre-stitch baseline ---
    # Compute the four objectives on the raw tracklets (no stitching: orig_id = stitched_id).
    # Written as the first row in the CSV and shown as x=0 in the plot.
    _pre_df   = long_df.assign(stitched_id=long_df["orig_id"])
    _pre_objs = compute_stitching_objectives(
        df_stitched       = _pre_df,
        vial_rois         = vial_rois,
        num_frames        = num_frames,
        expected_per_vial = args.expected_per_vial,
        short_frac        = args.short_frac,
    )
    _nan = float("nan")
    _baseline_row = {
        "extrap":    _nan, "direction": _nan, "behavioral": _nan,
        **{f"dir_{k}": _nan for k in _DIR_KEYS},
        **{f"beh_{k}": _nan for k in _BEH_KEYS},
        **_pre_objs,
    }
    print(
        f"Pre-stitch baseline:  vial_count_error={_pre_objs['vial_count_error']:.1f}  "
        f"cov_loss={_pre_objs['per_id_coverage_loss']:.1f}  "
        f"short={_pre_objs['short_track_count']}  "
        f"id_var={_pre_objs['per_frame_id_variance']:.3f}\n"
    )

    # --- Resume: skip configs already in the results CSV ---
    # If the script was interrupted (laptop closed, crash, etc.), re-running it
    # automatically picks up from where it left off. We read all previously
    # completed config keys into a set and skip them in the loop below.
    # We open the file in append mode ("a") so new rows are added after existing ones.
    completed: set = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        completed = {_config_key(row) for _, row in existing.iterrows()}
        print(f"Resuming — {len(completed)} configs already done, {total - len(completed)} remaining\n")
        out_f        = open(results_csv, "a", newline="", encoding="utf-8")
        write_header = False
    else:
        out_f        = open(results_csv, "w", newline="", encoding="utf-8")
        pd.DataFrame([_baseline_row]).to_csv(out_f, header=True, index=False)
        write_header = False

    n_resumed = len(completed)   # configs already on disk from a previous run
    n_done    = 0                # configs evaluated in THIS run
    t0        = time.time()

    try:
        for e, d, b in link_combos:
            # When d=0, direction weights don't contribute — only run first combo.
            # When b=0, behavioral weights don't contribute — only run first combo.
            dir_iter = dir_combos if d > 0 else dir_combos[:1]
            beh_iter = beh_combos if b > 0 else beh_combos[:1]

            for dir_w in dir_iter:
                for beh_w in beh_iter:
                    weights = {
                        "link_score_weights": {"extrap": e, "direction": d, "behavioral": b},
                        "direction_weights":  dir_w,
                        "behavioral_weights": beh_w,
                    }

                    # Build the metadata dict for this config — all weight values.
                    # We prefix direction keys with "dir_" and behavioral keys with
                    # "beh_" so columns are unambiguous in the results CSV.
                    row_meta = {
                        "extrap":     e,
                        "direction":  d,
                        "behavioral": b,
                        **{f"dir_{k}": v for k, v in dir_w.items()},
                        **{f"beh_{k}": v for k, v in beh_w.items()},
                    }

                    # Skip if this config was already evaluated in a previous run
                    if _config_key(row_meta) in completed:
                        continue

                    with contextlib.redirect_stdout(io.StringIO()):
                        stitched_df = stitch(
                            long_df    = long_df,
                            vial_rois  = vial_rois,
                            tracklets  = tracklets,
                            output_dir = None,
                            weights    = weights,
                        )
                    n_after = stitched_df["stitched_id"].nunique()
                    print(f"  [{n_done + n_resumed + 1}/{total}]  {n_before} -> {n_after} IDs")

                    objs = compute_stitching_objectives(
                        df_stitched       = stitched_df,
                        vial_rois         = vial_rois,
                        num_frames        = num_frames,
                        expected_per_vial = args.expected_per_vial,
                        short_frac        = args.short_frac,
                    )

                    row = {**row_meta, **objs}
                    completed.add(_config_key(row))
                    n_done += 1

                    # Write this row immediately and flush to disk.
                    # If the script crashes after this line, the row is already saved.
                    # write_header is True only for the very first row of a fresh run.
                    pd.DataFrame([row]).to_csv(out_f, header=write_header, index=False)
                    out_f.flush()
                    write_header = False

                    if n_done % 100 == 0:
                        elapsed   = time.time() - t0
                        total_done = n_done + n_resumed
                        rate      = n_done / elapsed if elapsed > 0 else 1e-9
                        eta_h     = (total - total_done) / rate / 3600
                        print(
                            f"  {total_done}/{total}  |  "
                            f"elapsed {elapsed/60:.1f} min  |  "
                            f"ETA {eta_h:.1f} h"
                        )

    finally:
        # Always close the file — even if the loop is interrupted by Ctrl+C or an error.
        out_f.close()

    print(f"\nGrid search complete — {n_done} new configs evaluated ({n_resumed} resumed from disk).")

    all_results = pd.read_csv(results_csv)
    print(f"\nTop 5 configs by vial_count_error:")
    print(all_results.sort_values("vial_count_error").head(5)[
        ["extrap", "direction", "behavioral", "vial_count_error", "per_id_coverage_loss", "short_track_count", "per_frame_id_variance"]
    ].to_string(index=False))

    plot_results(all_results, short_name, str(plot_html))
    print(f"\nResults : {results_csv}")
    print(f"Plot    : {plot_html}")


if __name__ == "__main__":
    main()