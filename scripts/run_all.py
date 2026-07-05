#!/usr/bin/env python
"""
scripts/run_all.py

Two-stage pipeline:

  Stage 1 -- Track every video under data/raw/ (or explicit --video paths)
             using the current config.yaml parameters.
             ROIs are loaded from roi_library.json; missing ones open a GUI.
             Overlay MP4s are written only when visualization.enabled is true
             (or when --overlay is passed).
             Every video is re-tracked by default so each run is fresh. Set
             pipeline.skip_tracked: true in config.yaml to instead skip videos
             that already have ordered_tracks.csv.

  Stage 2 -- Load all successfully tracked runs, extract behavioural features
             (including new episode features), run Kruskal-Wallis significance
             tests, and write a per-feature HTML report to
             outputs/analysis/significance_report/.

Usage
-----
    # Track everything + analyse
    python scripts/run_all.py

    # Analyse only (all existing tracked runs, skip tracking)
    python scripts/run_all.py --skip-tracking

    # Track specific videos then analyse
    python scripts/run_all.py --video data/raw/13\ DPE/001/...mp4 --video data/raw/13\ DPE/003/...mp4

    # Skip already-tracked videos (opt in via config: pipeline.skip_tracked: true)
    python scripts/run_all.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    """Print a boxed section header to the console.

    Inputs
    ------
    msg : str
        The heading text to frame between rule lines.

    Outputs
    -------
    None
        Prints to stdout.
    """
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")


def _discover_videos(data_root: Path) -> list[Path]:
    """Find every source video under a folder, excluding pipeline-generated copies.

    Recursively globs ``*-converted.mp4`` and drops the ``_pp`` and
    ``_raw_cropped`` derivatives so only original inputs are returned for batch
    tracking.

    Inputs
    ------
    data_root : pathlib.Path
        Root folder to search recursively.

    Outputs
    -------
    list[pathlib.Path]
        Sorted paths to source videos (excluding _pp / _raw_cropped copies).
    """
    videos = []
    for p in sorted(data_root.rglob("*-converted.mp4")):
        name = p.name
        if name.endswith("_pp.mp4") or "_raw_cropped" in name:
            continue
        videos.append(p)
    return videos


def _output_dir_for(video: Path, outputs_root: Path) -> Path:
    """Derive the output directory for a video (same rule as run_tracking.py).

    Uses ``make_run_output_dir`` so a video maps to the same folder here, in
    run_tracking.py, and in the notebook.

    Inputs
    ------
    video : pathlib.Path
        The source video.
    outputs_root : pathlib.Path
        Root under which per-run output directories live.

    Outputs
    -------
    pathlib.Path
        The output directory for this video.
    """
    from utils import make_run_output_dir
    return Path(make_run_output_dir(str(video), outputs_root=str(outputs_root)))


def _is_tracked(output_dir: Path) -> bool:
    """Report whether a run directory already holds a finished tracking result.

    Presence of ordered_tracks.csv is the completion marker used to skip
    already-tracked videos when pipeline.skip_tracked is true.

    Inputs
    ------
    output_dir : pathlib.Path
        Candidate run directory.

    Outputs
    -------
    bool
        True if ``ordered_tracks.csv`` exists in ``output_dir``.
    """
    return (output_dir / "ordered_tracks.csv").exists()


# ---------------------------------------------------------------------------
# Stage 1 -- Tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """Every output file for one tracked video."""
    output_dir: str
    pp_out: str
    raw_cropped_out: str
    roi_json: str
    ocsort_csv: str
    det_csv: str
    long_csv: str
    ordered_csv: str
    tracker_log: str


def build_paths(output_dir: Path, video: Path) -> RunPaths:
    """Derive every fast-mode output file path for one video.

    Inputs
    ------
    output_dir : pathlib.Path
        The run's output directory (from _output_dir_for).
    video : pathlib.Path
        The source video; its stem names the preprocessed copies.

    Outputs
    -------
    RunPaths
        Frozen bundle of output paths (pp / raw-cropped videos, vial_rois.json,
        track CSVs, long CSV, ordered CSV, tracker log). No overlay paths — this
        script runs in fast mode.
    """
    stem = video.stem
    return RunPaths(
        output_dir      = str(output_dir),
        pp_out          = str(output_dir / f"{stem}_pp.mp4"),
        raw_cropped_out = str(output_dir / f"{stem}_raw_cropped.mp4"),
        roi_json        = str(output_dir / "vial_rois.json"),
        ocsort_csv      = str(output_dir / "ocsort_tracks.csv"),
        det_csv         = str(output_dir / "detections_raw.csv"),
        long_csv        = str(output_dir / "ocsort_tracks_long.csv"),
        ordered_csv     = str(output_dir / "ordered_tracks.csv"),
        tracker_log     = str(output_dir / "tracker_log.json"),
    )


def record_run_metadata(video: Path, paths: RunPaths, cfg, model_id: str) -> tuple[float, str]:
    """Keep the raw video beside its outputs, snapshot config, and probe meta.

    Makes each batch run self-contained: the input video and the config are
    copied into the output folder and the video's fps / frame count are written
    to run_params.json. Returns the video stem, which keys the ROI library shared
    with run_tracking.py and the notebook.

    Inputs
    ------
    video : pathlib.Path
        The source video to hardlink/copy beside its outputs and probe.
    paths : RunPaths
        Run paths; ``paths.output_dir`` is used.
    cfg : Config
        Loaded configuration; supplies cfg.video.fallback_fps.
    model_id : str
        Roboflow model id, recorded into run_params.json for provenance.

    Outputs
    -------
    tuple[float, str]
        fps : the video's fps (or cfg.video.fallback_fps when none reported).
        video_key : the video stem, used as the ROI-library key.
    """
    import cv2
    from utils import link_or_copy, save_config_snapshot, save_run_params

    output_dir = Path(paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    link_or_copy(video, output_dir / video.name)
    save_config_snapshot(paths.output_dir, str(REPO_ROOT / "config.yaml"))

    cap = cv2.VideoCapture(str(video))
    fps_actual = float(cap.get(cv2.CAP_PROP_FPS) or cfg.video.fallback_fps)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    save_run_params(paths.output_dir, "video", {
        "path": str(video), "fps": fps_actual,
        "frames": n_frames, "model_id": model_id,
    })

    # Video stem keys the ROI library (same key run_tracking.py and the notebook
    # use), so a stored ROI is shared across all three entrypoints.
    return fps_actual, video.stem


def stage_preprocess(video: Path, paths: RunPaths, cfg, library: dict,
                     video_key: str) -> tuple[str, Path | None]:
    """Background subtraction using a stored crop, or the GUI when none exists.

    Reuses the library's crop for this video when available (and use_saved_roi is
    on); otherwise opens the GUI and caches the new crop back into the library.
    Always writes the preprocessed and raw-cropped videos.

    Inputs
    ------
    video : pathlib.Path
        The source video.
    paths : RunPaths
        Run paths (pp_out, raw_cropped_out).
    cfg : Config
        Loaded configuration (preprocessing.*, roi.use_saved_roi).
    library : dict
        The in-memory ROI library; a new crop is written into it under video_key.
    video_key : str
        The video stem keying this video's entry in the library.

    Outputs
    -------
    tuple[str, pathlib.Path | None]
        video_path : path to the preprocessed video to track.
        raw_cropped_path : path to the raw-cropped copy (used as ROI substrate).
    """
    from src.preprocessing import preprocess_bgsub_gui

    use_saved_roi = cfg.roi.use_saved_roi
    _stored_crop = library.get(video_key, {}).get("preprocessing") if use_saved_roi else None

    if use_saved_roi and _stored_crop is not None:
        print(f"  [preprocess] Using stored crop for {video_key}")
        video_path, _ = preprocess_bgsub_gui(
            str(video), paths.pp_out, paths.raw_cropped_out,
            crop_params=_stored_crop,
            gain=cfg.preprocessing.bg_gain,
            white_level=cfg.preprocessing.bg_white_level,
            bg_percentile=cfg.preprocessing.bg_percentile,
            bg_sample_stride=cfg.preprocessing.bg_sample_stride,
            codec=cfg.preprocessing.codec,
        )
    else:
        print(f"  [preprocess] No stored crop -- opening GUI for {video.name}")
        video_path, crop_params = preprocess_bgsub_gui(
            str(video), paths.pp_out, paths.raw_cropped_out,
            gain=cfg.preprocessing.bg_gain,
            white_level=cfg.preprocessing.bg_white_level,
            bg_percentile=cfg.preprocessing.bg_percentile,
            bg_sample_stride=cfg.preprocessing.bg_sample_stride,
            codec=cfg.preprocessing.codec,
        )
        library.setdefault(video_key, {})["preprocessing"] = crop_params

    return video_path, Path(paths.raw_cropped_out)


def stage_vial_rois(video: Path, paths: RunPaths, cfg, library: dict, video_key: str,
                    video_path: str, raw_cropped_path: Path | None) -> None:
    """Draw vial ROIs, or reuse the library's stored geometry, writing vial_rois.json.

    On reuse the stored box geometry is written straight to this run's
    vial_rois.json (n_flies is absent, so expected counts later fall back to
    config); otherwise the GUI draws on the raw-cropped substrate and the new
    geometry is cached to the library.

    Inputs
    ------
    video : pathlib.Path
        The source video (used for GUI prompts / naming).
    paths : RunPaths
        Run paths (roi_json).
    cfg : Config
        Loaded configuration (roi.use_saved_roi).
    library : dict
        The in-memory ROI library; new geometry is written under video_key.
    video_key : str
        The video stem keying this video's entry in the library.
    video_path : str
        Preprocessed video path, used as the ROI substrate when no raw-cropped
        copy exists.
    raw_cropped_path : pathlib.Path | None
        Raw-cropped video path; the preferred ROI substrate when it exists.

    Outputs
    -------
    None
        Writes ``vial_rois.json`` and may update ``library`` in place.
    """
    from src.roi import draw_and_save_vial_rois, load_vial_rois

    use_saved_roi = cfg.roi.use_saved_roi
    _stored_roi = library.get(video_key, {}).get("vial_rois") if use_saved_roi else None
    if use_saved_roi and _stored_roi is not None:
        # Library stores only ROI geometry; n_flies is a per-run label and is
        # absent on reuse (expected counts then fall back to config).
        print(f"  [roi] Using stored ROI for {video_key}")
        with open(paths.roi_json, "w") as _f:
            json.dump(_stored_roi, _f, indent=2)
    else:
        print(f"  [roi] Opening ROI GUI for {video.name}")
        overlay_src = raw_cropped_path if raw_cropped_path and raw_cropped_path.exists() else video_path
        draw_and_save_vial_rois(str(overlay_src), paths.roi_json)
        bbox_dict, _ = load_vial_rois(paths.roi_json)
        library.setdefault(video_key, {})["vial_rois"] = {k: list(v) for k, v in bbox_dict.items()}


def stage_track(video: Path, paths: RunPaths, cfg, api_key: str, model_id: str,
                video_path: str) -> tuple:
    """Run RF-DETR + OC-SORT with every parameter from config, writing the tracks.

    The fast-mode counterpart to run_tracking.py's stage_track: forwards all
    tracker params from config.yaml, resolves per-vial expected counts to gate
    ghost detection, and dumps detection / ghost / top-exit bookkeeping to
    tracker_log.json.

    Inputs
    ------
    video : pathlib.Path
        The source video (used for log messages).
    paths : RunPaths
        Run paths (ocsort_csv, det_csv, roi_json, tracker_log).
    cfg : Config
        Loaded configuration (tracker.*, tracker.ghost_detection.*, roboflow.*,
        watershed.*, pipeline.expected_per_vial).
    api_key : str
        Roboflow API key.
    model_id : str
        Roboflow model id, "<workspace>/<version>".
    video_path : str
        Video to track (raw or preprocessed, from stage_preprocess).

    Outputs
    -------
    tuple[pandas.DataFrame, OCSort]
        df_wide : wide tracks table (frame + id{N} columns).
        tracker : the OCSort instance (carries detection_log, ghost_log, etc.).
    """
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.roi import load_vial_rois, resolve_vial_expected_counts

    print(f"  [track] Running RF-DETR + OC-SORT on {video.name}...")
    _gd = cfg.tracker.ghost_detection
    _vial_rois, _n_flies = load_vial_rois(paths.roi_json)
    vial_expected_counts = resolve_vial_expected_counts(
        _n_flies, _vial_rois, cfg.pipeline.expected_per_vial
    )
    df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=paths.ocsort_csv,
        api_key=api_key,
        model_id=model_id,
        inference_api_url=cfg.roboflow.inference_api_url,
        confidence=cfg.tracker.confidence,
        detection_confidence_rfdetr=cfg.tracker.detection_confidence_rfdetr,
        lost_track_buffer=cfg.tracker.lost_track_buffer,
        minimum_matching_threshold=cfg.tracker.minimum_matching_threshold,
        minimum_consecutive_frames=cfg.tracker.minimum_consecutive_frames,
        asso_func=cfg.tracker.asso_func,
        brownian_pos_noise=cfg.tracker.brownian_pos_noise,
        det_log_csv=paths.det_csv,
        vial_rois=_vial_rois,
        aspect_weight=cfg.tracker.aspect_weight,
        behavioral_weights=dict(cfg.tracker.behavioral_weights),
        overlap_weight_scale=cfg.tracker.overlap_weight_scale,
        inertia=cfg.tracker.inertia,
        delta_t=cfg.tracker.delta_t,
        jump_factor=cfg.tracker.jump_factor,
        jump_iou_threshold=cfg.tracker.jump_iou_threshold,
        jump_inertia=cfg.tracker.jump_inertia,
        overlap_iou_scale=cfg.tracker.overlap_iou_scale,
        edge_fraction=cfg.tracker.edge_fraction,
        expected_count=cfg.tracker.expected_count,
        w_under=cfg.tracker.w_under,
        w_over=cfg.tracker.w_over,
        ghost_detection_enabled=_gd.enabled,
        ghost_offset_fraction=_gd.offset_fraction,
        ghost_confidence=_gd.confidence,
        ghost_occlusion_max_gap=_gd.occlusion_max_gap,
        ghost_top_exit_px=_gd.top_exit_px,
        watershed_cfg=dict(cfg.watershed),
        vial_expected_counts=vial_expected_counts,
    )

    with open(paths.tracker_log, "w") as _f:
        json.dump({
            "detection_log":      tracker.detection_log,
            "suppressed_tracks":  tracker.suppressed_tracks,
            "min_hits":           tracker.min_hits,
            "max_age":            tracker.max_age,
            "ghost_log":          getattr(tracker, "ghost_log",          []),
            "top_exit_events":    getattr(tracker, "top_exit_events",    []),
            "top_reentry_events": getattr(tracker, "top_reentry_events", []),
        }, _f)

    return df_wide, tracker


def stage_assign(paths: RunPaths, df_wide, fps: float):
    """Map tracks to vials and relabel to ordered IDs, writing ordered_tracks.csv.

    Converts wide tracks to long form (saved for reuse), assigns vials, and writes
    the ordered-id table which downstream analysis consumes.

    Inputs
    ------
    paths : RunPaths
        Run paths (long_csv, roi_json, ordered_csv).
    df_wide : pandas.DataFrame
        Wide tracks table from stage_track.
    fps : float
        Video frame rate (passed through for time columns).

    Outputs
    -------
    pandas.DataFrame
        df_ord : the ordered-tracks table read back from ordered_tracks.csv.
    """
    import pandas as pd
    from src.stitching import wide_to_long
    from src.roi import assign_vials_and_ordered_ids

    print("  [assign] Assigning vials + ordered IDs...")
    df_long = wide_to_long(df_wide)
    df_long.to_csv(paths.long_csv, index=False)
    assign_vials_and_ordered_ids(paths.long_csv, paths.roi_json, paths.ordered_csv, fps=fps)
    df_ord = pd.read_csv(paths.ordered_csv)
    print(f"  [done] {len(df_ord['ordered_id'].unique())} tracks -> {paths.ordered_csv}")
    return df_ord


def render_detection_overlay(video: Path, paths: RunPaths, cfg, video_path: str) -> None:
    """Render the raw RF-DETR detection overlay for visual QC.

    Inputs
    ------
    video : pathlib.Path
        Source video; its stem names the output MP4.
    paths : RunPaths
        Run paths (det_csv, output_dir).
    cfg : Config
        Loaded configuration (visualization.fps_out).
    video_path : str
        Video substrate the detections were computed on.

    Outputs
    -------
    None
        Writes ``<stem>_detections_RF-DETR.mp4`` into the output dir.
    """
    from src.visualization import render_detections_video

    print("  [overlay] RF-DETR detection overlay...")
    render_detections_video(
        video_path=video_path,
        det_log_csv=paths.det_csv,
        out_mp4=os.path.join(paths.output_dir, f"{video.stem}_detections_RF-DETR.mp4"),
        fps_out=cfg.visualization.fps_out,
    )


def stage_overlays(cfg, paths: RunPaths, video_path: str, vials: dict) -> None:
    """Render raw OC-SORT and ordered-track overlay videos.

    Inputs
    ------
    cfg : Config
        Loaded configuration (visualization.overlay_source, visualization.fps_out).
    paths : RunPaths
        Run paths (long_csv, ordered_csv, det_csv, output_dir).
    video_path : str
        Fallback substrate when the configured overlay source is absent.
    vials : dict
        vial id -> (x0, y0, x1, y1) ROIs drawn on both overlays.

    Outputs
    -------
    None
        Writes overlay_raw_ocsort.mp4 and overlay_ordered.mp4 into the output dir.
    """
    from src.visualization import render_vial_overlay_video, render_raw_overlay_video
    from utils import resolve_overlay_video, save_run_params

    print("  [overlay] Track overlays...")
    _overlay_mode = cfg.visualization.overlay_source
    overlay_video = resolve_overlay_video(paths.output_dir, _overlay_mode) or video_path
    det_log_arg = paths.det_csv if os.path.exists(paths.det_csv) else None

    raw_overlay_mp4 = os.path.join(paths.output_dir, "overlay_raw_ocsort.mp4")
    render_raw_overlay_video(
        video_path=overlay_video,
        csv_path=paths.long_csv,
        out_mp4=raw_overlay_mp4,
        vial_rois=vials,
        det_log_csv=det_log_arg,
        fps_out=cfg.visualization.fps_out,
    )

    ordered_overlay_mp4 = os.path.join(paths.output_dir, "overlay_ordered.mp4")
    render_vial_overlay_video(
        video_path=overlay_video,
        csv_path=paths.ordered_csv,
        out_mp4=ordered_overlay_mp4,
        vial_rois=vials,
        det_log_csv=det_log_arg,
        fps_out=cfg.visualization.fps_out,
    )

    save_run_params(paths.output_dir, "outputs", {
        "raw_overlay": raw_overlay_mp4,
        "ordered_overlay": ordered_overlay_mp4,
    })


def stage_diagnostics(paths: RunPaths, cfg, tracker, df_wide, df_ord, fps: float) -> None:
    """Write the metrics report for one run; never abort the batch on failure.

    Computes total expected flies and calls run_diagnostics; any exception is
    caught and logged so one bad run cannot stop the whole batch.

    Inputs
    ------
    paths : RunPaths
        Run paths (roi_json, output_dir).
    cfg : Config
        Loaded configuration (pipeline.expected_per_vial, report settings).
    tracker : OCSort
        The tracker from stage_track.
    df_wide : pandas.DataFrame
        Wide tracks table.
    df_ord : pandas.DataFrame
        Ordered tracks table from stage_assign.
    fps : float
        Video frame rate.

    Outputs
    -------
    None
        Writes metrics_report.md into the output dir; swallows and logs errors.
    """
    from src.roi import load_vial_rois, resolve_vial_expected_counts
    from src.metrics import run_diagnostics

    try:
        bbox_dict, n_flies_dict = load_vial_rois(paths.roi_json)
        n_expected = sum(resolve_vial_expected_counts(
            n_flies_dict, bbox_dict, cfg.pipeline.expected_per_vial).values())
        run_diagnostics(
            tracker=tracker,
            df_wide=df_wide,
            df_ordered=df_ord,
            n_expected=n_expected,
            fps=fps,
            vial_rois=bbox_dict,
            config=cfg,
            output_dir=paths.output_dir,
            show_plots=False,
        )
    except Exception as e:
        print(f"  [warn] Diagnostics failed: {e}")


def _save_library(library: dict, roi_lib_path: Path) -> None:
    """Persist the ROI library to disk (pretty-printed).

    Inputs
    ------
    library : dict
        The in-memory ROI library (video-stem -> stored geometry).
    roi_lib_path : pathlib.Path
        Destination JSON path.

    Outputs
    -------
    None
        Writes ``roi_lib_path`` as a side effect.
    """
    with open(roi_lib_path, "w") as f:
        json.dump(library, f, indent=2)


def draw_first_prepass(videos: list[Path], cfg, library: dict,
                       roi_lib_path: Path) -> None:
    """Front-load all human input: capture the crop + vial ROIs for every video.

    Opens the crop GUI and the vial-ROI GUI for each video up front and stores
    the geometry in the ROI library (keyed by video stem), so the slow,
    unattended RF-DETR + tracking loop that follows never stops for a person.
    The crop is written to the library before the ROI GUI runs, because
    ``draw_and_save_vial_rois`` reads that crop back from disk to draw ROIs in
    the same cropped coordinate space the tracker later uses. Videos that already
    carry both a crop and vial ROIs are skipped, so an interrupted pass resumes
    safely.

    Inputs
    ------
    videos : list[pathlib.Path]
        Videos to collect input for.
    cfg : Config
        Loaded configuration (only used to parse per-video display context).
    library : dict
        The in-memory ROI library; updated in place under each video's stem.
    roi_lib_path : pathlib.Path
        Where the library is persisted after each capture (crash-safe).

    Outputs
    -------
    None
        Updates ``library`` and rewrites ``roi_lib_path`` as it goes.
    """
    import tempfile
    from src.preprocessing import capture_crop_params_gui
    from src.roi import draw_and_save_vial_rois
    from src.ui_context import parse_video_context

    _banner(f"Draw-first -- collecting crop + ROIs for {len(videos)} video(s)")
    for i, video in enumerate(videos, start=1):
        stem  = video.stem
        entry = library.setdefault(stem, {})
        if entry.get("preprocessing") and entry.get("vial_rois"):
            print(f"  [{i}/{len(videos)}] skip (already have crop + ROIs): {stem}")
            continue

        print(f"  [{i}/{len(videos)}] {video.name}")
        vctx = parse_video_context(str(video))

        # 1) Crop geometry (GUI only, no video written). Persist immediately:
        #    draw_and_save_vial_rois reads this crop back from roi_library.json
        #    so the vial ROIs are drawn on the cropped frame.
        if not entry.get("preprocessing"):
            entry["preprocessing"] = capture_crop_params_gui(str(video), video_context=vctx)
            entry["video_path"]    = str(video)
            _save_library(library, roi_lib_path)

        # 2) Vial ROIs (drawn on the cropped frame, matching the tracked video).
        if not entry.get("vial_rois"):
            with tempfile.TemporaryDirectory() as tmp:
                roi_dict = draw_and_save_vial_rois(
                    video_path=str(video),
                    roi_json_path=str(Path(tmp) / "vial_rois.json"),
                    video_context=vctx,
                )
            entry["vial_rois"] = {k: list(v) for k, v in roi_dict.items()}
            _save_library(library, roi_lib_path)

    _banner("Draw-first complete -- tracking will now run hands-off")


def track_one(video: Path, output_dir: Path, api_key: str, model_id: str,
              cfg, library: dict, *, write_overlays: bool) -> bool:
    """Track one video end-to-end; optionally render overlay MP4s.

    Mirrors run_tracking.py's stage sequence — metadata, preprocess, ROIs, track,
    assign, diagnostics — and writes overlay videos when ``write_overlays`` is
    true (from config.visualization.enabled or CLI --overlay).

    Inputs
    ------
    video : pathlib.Path
        The source video to track.
    output_dir : pathlib.Path
        Where this run's outputs are written.
    api_key : str
        Roboflow API key.
    model_id : str
        Roboflow model id, "<workspace>/<version>".
    cfg : Config
        Loaded configuration.
    library : dict
        The in-memory ROI library, potentially updated by the ROI / preprocess
        stages.
    write_overlays : bool
        When true, render detection + track overlay MP4s after tracking.

    Outputs
    -------
    bool
        True on success. Exceptions propagate to the caller, which records the
        failure and continues the batch.
    """
    paths = build_paths(output_dir, video)
    fps, video_key = record_run_metadata(video, paths, cfg, model_id)

    video_path, raw_cropped_path = stage_preprocess(video, paths, cfg, library, video_key)
    stage_vial_rois(video, paths, cfg, library, video_key, video_path, raw_cropped_path)
    df_wide, tracker = stage_track(video, paths, cfg, api_key, model_id, video_path)
    if write_overlays:
        render_detection_overlay(video, paths, cfg, video_path)
    df_ord = stage_assign(paths, df_wide, fps)
    stage_diagnostics(paths, cfg, tracker, df_wide, df_ord, fps)
    if write_overlays:
        from src.roi import load_vial_rois
        vials, _ = load_vial_rois(paths.roi_json)
        stage_overlays(cfg, paths, video_path, vials)
    return True


# ---------------------------------------------------------------------------
# Stage 2 -- Classification analysis
# ---------------------------------------------------------------------------

# Fixed left-to-right vial layout used in every video of this experiment
# (vial1 .. vial6). Most runs in all_runs_final/ were stripped down to just
# ordered_tracks.csv — no run_params.json / video filename — so the genotype
# can't be parsed per run. The layout is constant across the dataset (verified
# against every run that still carries its filename), so we fall back to it.
VIAL_GENOTYPE_LAYOUT = ["WT", "A90V", "G287S", "G294A", "A315T", "M337V"]


def _load_run_with_genotype(rd):
    """Load a run's ordered_tracks.csv with a ``genotype`` column attached.

    Prefers the per-run video filename (via ``map_vial_to_genotype``) when
    ``run_params.json`` is present; otherwise assigns genotype from the fixed
    left-to-right vial layout. This lets stripped-down runs (ordered_tracks.csv
    only) still be analysed.

    Inputs
    ------
    rd : pathlib.Path
        A run directory containing ordered_tracks.csv (and optionally
        run_params.json).

    Outputs
    -------
    pandas.DataFrame
        The run's ordered tracks with an added ``genotype`` column.

    Raises
    ------
    Exception
        Only if ordered_tracks.csv itself cannot be read.
    """
    import os
    import pandas as pd
    from src.classification import map_vial_to_genotype

    if os.path.exists(os.path.join(str(rd), "run_params.json")):
        try:
            return map_vial_to_genotype(str(rd))
        except Exception as e:
            print(f"    [genotype] {rd.name}: filename parse failed ({e}); "
                  f"using fixed vial layout")

    csv_path = os.path.join(str(rd), "ordered_tracks.csv")
    df = pd.read_csv(csv_path)
    layout = {f"vial{i + 1}": g for i, g in enumerate(VIAL_GENOTYPE_LAYOUT)}
    df["genotype"] = df["vial_id"].map(layout)
    return df


def _parse_age_dpe(run_tag: str) -> float:
    """Pull the days-post-eclosion (DPE) number out of a run name.

    Inputs
    ------
    run_tag : str
        The run name / tag, e.g. ``..._13DPE_...``.

    Outputs
    -------
    float
        The DPE integer as a float, or ``float("nan")`` when no ``N DPE`` token
        is present.
    """
    import re
    m = re.search(r"(\d+)\s*DPE", run_tag, flags=re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def _add_vial_relative_xy(d):
    """Add rel_x, rel_y: each fly's position normalised to [0, 1] within its vial.

    Absolute x/y are meaningless across vials — different genotypes sit at
    different screen positions — so position is re-expressed relative to the
    fly's own vial bounding box before it can feed the embedding.

    Inputs
    ------
    d : pandas.DataFrame
        Per-frame track rows with ``x`` / ``y`` columns (and ideally ``vial_id``).

    Outputs
    -------
    pandas.DataFrame
        A copy of ``d`` with added ``rel_x`` and ``rel_y`` columns in [0, 1]
        (normalised per vial when ``vial_id`` is present, else globally).
    """
    import numpy as np
    d = d.copy()
    grouper = d.groupby("vial_id") if "vial_id" in d.columns else None
    for ax in ("x", "y"):
        if grouper is not None:
            lo = grouper[ax].transform("min")
            rng = grouper[ax].transform("max") - lo
        else:
            lo = d[ax].min()
            rng = d[ax].max() - lo
        d["rel_" + ax] = np.where(rng > 0, (d[ax] - lo) / rng, 0.0)
    return d


def run_analysis(run_dirs: list[Path], out_dir: Path) -> None:
    """Stage 2 — extract behavioural features across runs and write the report.

    Loads every tracked run (with genotype), computes per-fly behavioural and
    vial-relative position features, runs Kruskal-Wallis significance tests across
    genotypes, and writes an HTML significance report.

    Inputs
    ------
    run_dirs : list[pathlib.Path]
        Tracked run directories (each with ordered_tracks.csv) to include.
    out_dir : pathlib.Path
        Destination directory for the significance report.

    Outputs
    -------
    None
        Writes the significance report to ``out_dir``; returns early (with a
        printed message) if no runs could be loaded.
    """
    import pandas as pd
    from src.features import (
        extract_behavioral_features,
        aggregate_per_fly_features,
        classification_feature_columns,
    )
    from src.statistics import feature_significance_report, write_significance_report

    _banner("Stage 2 -- Feature extraction & significance analysis")

    parts = []
    for rd in run_dirs:
        try:
            d = _load_run_with_genotype(rd)
            run_tag = rd.name
            d["run"]        = run_tag
            d["age_dpe"]    = _parse_age_dpe(run_tag)
            d = _add_vial_relative_xy(d)   # rel_x, rel_y within each vial
            d["ordered_id"] = run_tag + "::" + d["ordered_id"].astype(str)
            parts.append(d)
            print(f"  [load] {run_tag}: {d['ordered_id'].nunique()} flies")
        except Exception as e:
            print(f"  [skip] {rd.name}: {e}")

    if not parts:
        print("  No runs loaded -- aborting analysis.")
        return

    df_raw = pd.concat(parts, ignore_index=True)
    print(f"\n  Combined: {df_raw.shape[0]} frames, "
          f"{df_raw['ordered_id'].nunique()} flies, "
          f"{df_raw['genotype'].nunique()} genotypes")

    print("\n  Extracting behavioural features...")
    df_feat = extract_behavioral_features(df_raw)
    df_agg  = aggregate_per_fly_features(df_feat)
    meta    = (df_raw.drop_duplicates("ordered_id")
                     .set_index("ordered_id")[["genotype", "run", "age_dpe"]])
    df_agg  = df_agg.join(meta, on="ordered_id").dropna(subset=["genotype"])

    # Per-fly vial-relative position summaries — fair spatial features for the
    # embedding (mean/spread of where the fly sat inside its own vial).
    rel = (df_raw.groupby("ordered_id")[["rel_x", "rel_y"]]
                 .agg(["mean", "std"]))
    rel.columns = ["rel_x_mean", "rel_x_std", "rel_y_mean", "rel_y_std"]
    df_agg = df_agg.join(rel, on="ordered_id")
    print(f"  Aggregated: {df_agg.shape[0]} flies x {df_agg.shape[1]} columns")

    FEATURES = classification_feature_columns()
    REL_FEATURES = ["rel_x_mean", "rel_y_mean", "rel_x_std", "rel_y_std"]
    print(f"\n  Running significance tests on {len(FEATURES)} features...")
    results, vol_fig, bar_fig = feature_significance_report(df_agg, FEATURES)

    sig_n = (results["kw_p_adj"] < 0.05).sum()
    print(f"  Significant (FDR < 0.05): {sig_n}/{len(FEATURES)}")
    print(results[["feature", "kw_p_adj", "max_abs_delta"]].head(10).to_string(index=False))

    report_path = write_significance_report(
        results, vol_fig, bar_fig,
        df=df_agg, features=FEATURES,
        out_dir=str(out_dir),
        n_runs=len(run_dirs),
        embed_features=FEATURES + REL_FEATURES,
    )
    print(f"\n  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for batch tracking + analysis.

    Inputs
    ------
    None

    Outputs
    -------
    argparse.ArgumentParser
        Parser exposing --video, --data-root, --outputs-root, --analysis-root,
        --analysis-out, --skip-tracking, --skip-analysis, and --draw-first.
    """
    p = argparse.ArgumentParser(
        description="Track all videos + run behavioural significance analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", action="append", dest="videos", metavar="PATH",
        help="Explicit video(s) to track. Repeat for multiple. "
             "If omitted, all *-converted.mp4 under data/raw/ are used.",
    )
    p.add_argument(
        "--data-root", default="data/raw",
        help="Root folder to search for videos when --video is not given.",
    )
    p.add_argument(
        "--outputs-root", default="data/outputs",
        help="Where tracked run directories are written (Stage 1 output).",
    )
    p.add_argument(
        "--analysis-root", default="all_runs_final",
        help="Folder of tracked run dirs to analyse in Stage 2. Defaults to "
             "all_runs_final/ (the canonical, most-recent set of runs).",
    )
    p.add_argument(
        "--analysis-out", default="data/outputs/analysis/significance_report",
        help="Where the significance report is written.",
    )
    p.add_argument(
        "--skip-tracking", action="store_true",
        help="Skip tracking; run analysis on all existing tracked runs.",
    )
    p.add_argument(
        "--skip-analysis", action="store_true",
        help="Track only, skip the analysis stage.",
    )
    p.add_argument(
        "--draw-first", action="store_true",
        help="Collect the crop + vial ROIs for every video up front (one GUI "
             "pass), then track them all hands-off. Requires roi.use_saved_roi=true.",
    )
    p.add_argument(
        "--overlay", dest="overlay", action="store_true",
        help="Write overlay MP4s (overrides config.visualization.enabled).",
    )
    p.add_argument(
        "--no-overlay", dest="overlay", action="store_false",
        help="Skip overlay MP4s (overrides config.visualization.enabled).",
    )
    p.set_defaults(overlay=None)
    return p


def main() -> None:
    """Entry point: track all requested videos (Stage 1), then analyse (Stage 2).

    Stage 1 tracks each discovered / explicit video (fresh every run by default;
    set pipeline.skip_tracked to skip finished ones), optionally writing overlay
    MP4s per visualization.enabled / --overlay.
    Stage 2 loads the tracked runs and writes the behavioural significance report.

    Inputs
    ------
    None
        Reads config.yaml / creds_config.yaml and the process command-line args.

    Outputs
    -------
    None
        Runs the two stages for their side effects (tracked run dirs + report).
    """
    from utils import load_config, resolve_overlay_enabled

    args  = build_parser().parse_args()
    cfg   = load_config(REPO_ROOT / "config.yaml")
    write_overlays = resolve_overlay_enabled(cfg, args.overlay)

    outputs_root = (REPO_ROOT / args.outputs_root).resolve()
    analysis_out = (REPO_ROOT / args.analysis_out).resolve()

    # ------------------------------------------------------------------ #
    # Stage 1 -- Tracking                                                  #
    # ------------------------------------------------------------------ #
    if not args.skip_tracking:
        from utils import load_creds
        api_key, model_id = load_creds(cfg, REPO_ROOT / "creds_config.yaml")

        if args.videos:
            videos = [Path(v).expanduser().resolve() for v in args.videos]
        else:
            data_root = (REPO_ROOT / args.data_root).resolve()
            videos    = _discover_videos(data_root)

        if not videos:
            print(f"No videos found under {args.data_root}")
            sys.exit(1)

        _banner(f"Stage 1 -- Tracking {len(videos)} video(s)"
                + (" (overlays on)" if write_overlays else " (overlays off)"))

        # Load ROI library
        roi_lib_path = REPO_ROOT / "roi_library.json"
        library: dict = {}
        if roi_lib_path.exists():
            with open(roi_lib_path) as f:
                library = json.load(f)

        # Optional GUI pre-pass: gather every crop + ROI first, so the tracking
        # loop below runs unattended. Reuse only kicks in when use_saved_roi is
        # on, so warn loudly if it isn't (otherwise tracking re-opens the GUIs).
        if args.draw_first:
            if not cfg.roi.use_saved_roi:
                print("  [warn] roi.use_saved_roi is false in config.yaml -- tracking "
                      "will re-open the GUIs. Set it true for a hands-off run.")
            draw_first_prepass(videos, cfg, library, roi_lib_path)

        tracked_dirs: list[Path] = []
        failed: list[Path] = []

        for video in videos:
            out_dir = _output_dir_for(video, outputs_root)
            if cfg.pipeline.skip_tracked and _is_tracked(out_dir):
                print(f"\n[skip] Already tracked (pipeline.skip_tracked): {out_dir.name}")
                tracked_dirs.append(out_dir)
                continue

            print(f"\n[track] {video.name}  ->  {out_dir.name}")
            try:
                ok = track_one(video, out_dir, api_key, model_id, cfg, library,
                               write_overlays=write_overlays)
                if ok:
                    tracked_dirs.append(out_dir)
                    # Persist updated ROI library
                    with open(roi_lib_path, "w") as f:
                        json.dump(library, f, indent=2)
            except Exception:
                print(f"  [FAIL] {video.name}")
                traceback.print_exc()
                failed.append(video)

        print(f"\nTracking complete: {len(tracked_dirs)} succeeded, {len(failed)} failed.")
        if failed:
            for v in failed:
                print(f"  FAILED: {v}")

    # ------------------------------------------------------------------ #
    # Stage 2 -- Analysis                                                  #
    # ------------------------------------------------------------------ #
    if not args.skip_analysis:
        analysis_root = (REPO_ROOT / args.analysis_root).resolve()
        if not analysis_root.exists():
            print(f"Analysis root not found: {analysis_root} -- skipping analysis.")
            return
        # Collect all run dirs that have ordered_tracks.csv
        run_dirs = sorted([
            d for d in analysis_root.iterdir()
            if d.is_dir() and (d / "ordered_tracks.csv").exists()
        ])
        if not run_dirs:
            print(f"No tracked runs found under {analysis_root} -- skipping analysis.")
            return
        # Never overwrite a previous report: version the output dir.
        if analysis_out.exists():
            i = 2
            while analysis_out.with_name(f"{analysis_out.name}_{i}").exists():
                i += 1
            analysis_out = analysis_out.with_name(f"{analysis_out.name}_{i}")
        print(f"\nFound {len(run_dirs)} tracked run(s) under {analysis_root.name} for analysis.")
        print(f"Writing report to {analysis_out}")
        run_analysis(run_dirs, analysis_out)


if __name__ == "__main__":
    main()
