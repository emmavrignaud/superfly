#!/usr/bin/env python
"""
scripts/run_tracking.py

CLI: one raw video -> ocsort_tracks.csv + ordered_tracks.csv + overlay videos.

This is the single-video, full-render counterpart to run_all.py (which batches
many videos in fast mode). Both read all behaviour from config.yaml.

Config is the single source of truth. You change a run by editing config.yaml
(and creds_config.yaml for secrets) — not this script. The only thing you may
pass on the command line is the video to track, which overrides
config.video.raw_path for this one invocation.

Stages (one function each; main() just orders them)
---------------------------------------------------
1. (optional) Background-subtraction GUI   -- config.preprocessing.enabled
2. Interactive vial ROI drawing            -- draws & saves vial_rois.json
3. RF-DETR + OC-SORT tracking              -- writes ocsort_tracks.csv + detections_raw.csv
3b. RF-DETR detection overlay video
4. Vial assignment + ordered IDs           -- writes ordered_tracks.csv
5. Overlay videos                          -- overlay_raw_ocsort.mp4 + overlay_ordered.mp4

Usage
-----
    # Track the video named in config.yaml (video.raw_path)
    python scripts/run_tracking.py

    # Track a specific video instead (overrides config.video.raw_path)
    python scripts/run_tracking.py --video data/raw/my_experiment.mp4
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import (
    config_for_run,
    inference_requires_api_key,
    inference_tracking_kwargs,
    link_or_copy,
    load_config,
    load_creds,
    make_run_output_dir,
    resolve_overlay_enabled,
    resolve_overlay_video,
    save_config_snapshot,
    save_run_params,
)

CONFIG_PATH      = REPO_ROOT / "config.yaml"
CREDS_PATH       = REPO_ROOT / "creds_config.yaml"
ROI_LIBRARY_PATH = REPO_ROOT / "roi_library.json"


# ===========================================================================
# Run paths — every output file location, derived once from the video.
# ===========================================================================

@dataclass(frozen=True)
class RunPaths:
    """Every output file location for one tracked video, derived once from it.

    A single frozen bundle passed to each stage so the notebook-style scattering
    of ``os.path.join`` calls can't drift apart between stages.
    """
    output_dir: str
    stem: str
    pp_out: str
    raw_cropped_out: str
    roi_json: str
    crop_roi_json: str
    ocsort_csv: str
    ocsort_long: str
    ordered_csv: str
    tracker_log_json: str
    det_log_csv: str


def build_paths(config, raw_video: Path) -> RunPaths:
    """Auto-derive the run's output folder and every file path inside it.

    Centralising path construction here means every stage receives one RunPaths
    object instead of recomputing names, so outputs can never drift apart.
    ``det_log_csv`` points at a cached detections_raw.csv (skipping RF-DETR) when
    config.tracker.cached_detections is set and exists; otherwise it is the fresh
    detections file for this run.

    Inputs
    ------
    config : Config
        Loaded configuration; reads config.tracker.cached_detections.
    raw_video : pathlib.Path
        The video being tracked; its stem names the output folder and files.

    Outputs
    -------
    RunPaths
        Frozen dataclass holding output_dir plus every derived file path
        (preprocessed video, ROI JSONs, track CSVs, tracker log, detection cache).
    """
    output_dir = make_run_output_dir(str(raw_video), outputs_root=str(REPO_ROOT / "data" / "outputs"))
    stem = raw_video.stem

    cached = config.tracker.cached_detections
    det_log_csv = (
        str(REPO_ROOT / cached)
        if cached and (REPO_ROOT / cached).exists()
        else os.path.join(output_dir, "detections_raw.csv")
    )

    return RunPaths(
        output_dir       = output_dir,
        stem             = stem,
        pp_out           = os.path.join(output_dir, f"{stem}_pp.mp4"),
        raw_cropped_out  = os.path.join(output_dir, f"{stem}_raw_cropped.mp4"),
        roi_json         = os.path.join(output_dir, "vial_rois.json"),
        crop_roi_json    = os.path.join(output_dir, "crop_roi.json"),
        ocsort_csv       = os.path.join(output_dir, "ocsort_tracks.csv"),
        ocsort_long      = os.path.join(output_dir, "ocsort_tracks_long.csv"),
        ordered_csv      = os.path.join(output_dir, "ordered_tracks.csv"),
        tracker_log_json = os.path.join(output_dir, "tracker_log.json"),
        det_log_csv      = det_log_csv,
    )


# ===========================================================================
# Small helpers
# ===========================================================================

def _load_roi_library() -> dict:
    """Read the shared ROI library JSON, or return an empty dict if absent.

    The library (``roi_library.json`` at the repo root) caches per-video crop and
    vial-ROI geometry so a video set up once can be reused across runs.

    Inputs
    ------
    None

    Outputs
    -------
    dict
        video-stem -> stored ROI data (keys like "preprocessing", "vial_rois",
        "video_path"); empty when the file does not exist yet.
    """
    if ROI_LIBRARY_PATH.exists():
        with open(ROI_LIBRARY_PATH) as f:
            return json.load(f)
    return {}


def _save_roi_library(library: dict) -> None:
    """Write the ROI library back to ``roi_library.json`` (pretty-printed).

    Inputs
    ------
    library : dict
        The full ROI library to persist (video-stem -> stored ROI data).

    Outputs
    -------
    None
        Writes ``roi_library.json`` as a side effect, creating parents if needed.
    """
    ROI_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_LIBRARY_PATH, "w") as f:
        json.dump(library, f, indent=2)


def resolve_raw_video(args, config) -> Path:
    """Pick the video to track: CLI ``--video`` if given, else config.video.raw_path.

    The command line has the final say for a single invocation, so a one-off run
    on a different video needs no edit to config.yaml.

    Inputs
    ------
    args : argparse.Namespace
        Parsed CLI args; only ``args.video`` (str | None) is read.
    config : Config
        Loaded configuration; supplies config.video.raw_path as the fallback.

    Outputs
    -------
    pathlib.Path
        Path to an existing video (expanded/joined to the repo root).

    Raises
    ------
    SystemExit
        If the resolved video does not exist on disk.
    """
    raw_video = Path(args.video).expanduser() if args.video else REPO_ROOT / config.video.raw_path
    if not raw_video.exists():
        raise SystemExit(f"Video not found: {raw_video}")
    return raw_video


def record_run_metadata(config, paths: RunPaths, raw_video: Path) -> float:
    """Keep the raw video beside its outputs, snapshot config, and probe video meta.

    Makes each run self-contained and reproducible: the exact input video and the
    config used are placed in the output folder, and the video's geometry/fps are
    recorded to run_params.json.

    Inputs
    ------
    config : Config
        Loaded configuration; supplies config.video.fallback_fps.
    paths : RunPaths
        Run paths; only ``paths.output_dir`` is used here.
    raw_video : pathlib.Path
        The input video to hardlink/copy beside the outputs and probe.

    Outputs
    -------
    float
        The video's fps (OpenCV-reported, or config.video.fallback_fps when the
        video reports none). Consumed by later stages.
    """
    import cv2

    link_or_copy(raw_video, os.path.join(paths.output_dir, raw_video.name))
    save_config_snapshot(
        paths.output_dir, CONFIG_PATH,
        raw_video=raw_video, repo_root=REPO_ROOT,
    )

    cap = cv2.VideoCapture(str(raw_video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.video.fallback_fps)
    save_run_params(paths.output_dir, "video", {
        "path":   str(raw_video),
        "fps":    fps,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    })
    cap.release()
    return fps


# ===========================================================================
# Stage 1 — background subtraction (optional)
# ===========================================================================

def stage_preprocess(config, paths: RunPaths, raw_video: Path, video_context) -> str:
    """Stage 1 — optionally background-subtract / crop the video before tracking.

    When config.preprocessing.enabled is set, opens the bg-subtraction GUI (or
    reuses a stored crop) and writes a preprocessed ``_pp.mp4``; the crop params
    are cached in the ROI library and this run's crop_roi.json. When disabled, the
    raw video is used untouched. The choice is recorded to run_params.json either
    way.

    Inputs
    ------
    config : Config
        Loaded configuration (preprocessing.*, roi.use_saved_roi).
    paths : RunPaths
        Run paths (pp_out, raw_cropped_out, crop_roi_json, output_dir, stem).
    raw_video : pathlib.Path
        The input video to preprocess.
    video_context : VideoContext
        Parsed video metadata passed through to the preprocessing GUI.

    Outputs
    -------
    str
        Path to the video tracking should use: the ``_pp.mp4`` when preprocessing
        ran, otherwise ``str(raw_video)`` unchanged.
    """
    from src.preprocessing import preprocess_bgsub_gui

    video_path       = str(raw_video)
    raw_cropped_path = None
    crop_params      = None
    video_key        = paths.stem
    use_saved_roi    = config.roi.use_saved_roi

    if config.preprocessing.enabled:
        print("\n=== Stage 1: Background subtraction ===")
        raw_cropped_path = paths.raw_cropped_out
        library = _load_roi_library()

        _stored_crop = library.get(video_key, {}).get("preprocessing")
        if use_saved_roi and _stored_crop is not None:
            print(f"Found stored preprocessing params for: {video_key}")
        else:
            print(f"No stored preprocessing params for: {video_key}: opening GUI...")

        video_path, crop_params = preprocess_bgsub_gui(
            video_path=video_path,
            out_mp4=paths.pp_out,
            out_raw_mp4=paths.raw_cropped_out,
            gain=config.preprocessing.bg_gain,
            white_level=config.preprocessing.bg_white_level,
            bg_sample_stride=config.preprocessing.bg_sample_stride,
            bg_percentile=config.preprocessing.bg_percentile,
            codec=config.preprocessing.codec,
            crop_params=_stored_crop if (use_saved_roi and _stored_crop is not None) else None,
            video_context=video_context,
        )
        print(f"Preprocessed video: {video_path}")

        library.setdefault(video_key, {})
        library[video_key]["preprocessing"] = crop_params
        library[video_key]["video_path"] = str(raw_video)
        _save_roi_library(library)

        with open(paths.crop_roi_json, "w") as _f:
            json.dump(crop_params, _f, indent=2)
        print(f"Saved crop params: {paths.crop_roi_json}")

    save_run_params(paths.output_dir, "preprocessing",
                    {"video_pp": video_path, "video_raw_cropped": raw_cropped_path, "crop_params": crop_params})
    return video_path


# ===========================================================================
# Stage 2 — draw / load vial ROIs
# ===========================================================================

def stage_vial_rois(config, paths: RunPaths, raw_video: Path, video_context) -> tuple[dict, dict]:
    """Stage 2 — obtain the vial ROIs, drawing them or reusing stored geometry.

    When config.roi.use_saved_roi is set and the library holds ROIs for this
    video, they are written to this run's vial_rois.json and loaded; otherwise the
    ROI GUI is opened and the freshly drawn geometry is cached to the library. The
    library stores only box geometry (reusable across runs); ``n_flies`` is a
    per-run label read back from this run's JSON.

    Inputs
    ------
    config : Config
        Loaded configuration (roi.use_saved_roi).
    paths : RunPaths
        Run paths (roi_json, output_dir, stem).
    raw_video : pathlib.Path
        The input video (the ROI GUI draws on it; crop is applied internally).
    video_context : VideoContext
        Parsed video metadata passed through to the ROI GUI.

    Outputs
    -------
    tuple[dict, dict]
        vials : vial id -> (x0, y0, x1, y1) bounding box.
        n_flies_dict : vial id -> fly count drawn this run (empty on ROI reuse).
    """
    from src.roi import draw_and_save_vial_rois, load_vial_rois, load_vial_colors

    print("\n=== Stage 2: Draw vial ROIs ===")
    video_key     = paths.stem
    use_saved_roi = config.roi.use_saved_roi
    library       = _load_roi_library()

    _stored_vials = library.get(video_key, {}).get("vial_rois")
    if use_saved_roi and _stored_vials is not None:
        print(f"Found stored vial ROIs for: {video_key}")
        with open(paths.roi_json, "w") as f:
            json.dump(_stored_vials, f, indent=2)
        vials, n_flies_dict = load_vial_rois(paths.roi_json)
        print(f"Loaded {len(vials)} vials from library.")
    else:
        if not use_saved_roi:
            print("use_saved_roi=False: opening GUI...")
        else:
            print(f"No stored vial ROIs for: {video_key}: opening GUI...")
        vials = draw_and_save_vial_rois(
            video_path=str(raw_video),
            roi_json_path=paths.roi_json,
            video_context=video_context,
        )
        # Load n_flies from the JSON the GUI just wrote (per-run only).
        _, n_flies_dict = load_vial_rois(paths.roi_json)

        # The library caches the ROI geometry AND the per-vial colour (both are
        # persistent identity, reused across runs); n_flies is a per-run label and
        # stays only in this run's vial_rois.json.
        _colors = load_vial_colors(paths.roi_json)
        library.setdefault(video_key, {})
        library[video_key]["vial_rois"] = {
            k: ({"bbox": list(v), "color": _colors[k]} if k in _colors else list(v))
            for k, v in vials.items()
        }
        _save_roi_library(library)

    save_run_params(paths.output_dir, "roi", {k: list(v) for k, v in vials.items()})
    return vials, n_flies_dict


# ===========================================================================
# Stage 3 — RF-DETR + OC-SORT tracking (+ 3b detection overlay)
# ===========================================================================

def stage_track(config, paths: RunPaths, video_path: str, api_key: str, model_id: str,
                vials: dict, n_flies_dict: dict) -> tuple:
    """Stage 3 — run RF-DETR + OC-SORT and write the tracks + tracker log.

    Forwards every tracker parameter from config.yaml to the tracking entry point
    (config is the single source of truth), resolves per-vial expected counts to
    gate ghost detection, and dumps the tracker's detection / ghost / top-exit
    bookkeeping to tracker_log.json.

    Inputs
    ------
    config : Config
        Loaded configuration (tracker.*, tracker.ghost_detection.*, roboflow.*,
        watershed.*, pipeline.expected_per_vial).
    paths : RunPaths
        Run paths (ocsort_csv, det_log_csv, tracker_log_json, output_dir).
    video_path : str
        Video to track (raw or preprocessed, from stage_preprocess).
    api_key : str
        Roboflow API key.
    model_id : str
        Roboflow model id, "<workspace>/<version>".
    vials : dict
        vial id -> (x0, y0, x1, y1) ROIs.
    n_flies_dict : dict
        vial id -> fly count drawn this run (used to resolve expected counts).

    Outputs
    -------
    tuple[pandas.DataFrame, OCSort]
        df_wide : wide tracks table (frame + id{N} columns).
        tracker : the OCSort instance (carries detection_log, ghost_log, etc.).
    """
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.roi import resolve_vial_expected_counts

    print("\n=== Stage 3: RF-DETR + OC-SORT tracking ===")
    tracker_cfg = config.tracker
    ghost_cfg   = tracker_cfg.ghost_detection
    vial_expected_counts = resolve_vial_expected_counts(
        n_flies_dict, vials, config.pipeline.expected_per_vial
    )
    df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=paths.ocsort_csv,
        api_key=api_key,
        model_id=model_id,
        inference_api_url=config.roboflow.inference_api_url,
        **inference_tracking_kwargs(config, REPO_ROOT),
        detection_confidence_rfdetr=tracker_cfg.detection_confidence_rfdetr,
        confidence=tracker_cfg.confidence,
        lost_track_buffer=tracker_cfg.lost_track_buffer,
        minimum_matching_threshold=tracker_cfg.minimum_matching_threshold,
        minimum_consecutive_frames=tracker_cfg.minimum_consecutive_frames,
        asso_func=tracker_cfg.asso_func,
        brownian_pos_noise=tracker_cfg.brownian_pos_noise,
        det_log_csv=paths.det_log_csv,
        vial_rois=vials,
        aspect_weight=tracker_cfg.aspect_weight,
        behavioral_weights=dict(tracker_cfg.behavioral_weights),
        overlap_weight_scale=tracker_cfg.overlap_weight_scale,
        jump_factor=tracker_cfg.jump_factor,
        jump_iou_threshold=tracker_cfg.jump_iou_threshold,
        jump_inertia=tracker_cfg.jump_inertia,
        inertia=tracker_cfg.inertia,
        delta_t=tracker_cfg.delta_t,
        overlap_iou_scale=tracker_cfg.overlap_iou_scale,
        edge_fraction=tracker_cfg.edge_fraction,
        expected_count=tracker_cfg.expected_count,
        w_under=tracker_cfg.w_under,
        w_over=tracker_cfg.w_over,
        watershed_cfg=dict(config.watershed),
        vial_expected_counts=vial_expected_counts,
        ghost_detection_enabled=ghost_cfg.enabled,
        ghost_offset_fraction=ghost_cfg.offset_fraction,
        ghost_confidence=ghost_cfg.confidence,
        ghost_occlusion_max_gap=ghost_cfg.occlusion_max_gap,
        ghost_top_exit_px=ghost_cfg.top_exit_px,
    )
    print(f"  shape: {df_wide.shape}")
    save_run_params(paths.output_dir, "tracker_output", {
        "ocsort_csv": paths.ocsort_csv,
        "frames": int(df_wide.shape[0]),
        "track_count": int(df_wide.shape[1] - 1),
    })

    with open(paths.tracker_log_json, "w") as _f:
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


def render_detection_overlay(config, paths: RunPaths, video_path: str) -> None:
    """Stage 3b — render a video of the raw RF-DETR detections for visual QC.

    Draws the cached detection boxes back onto the video so detection quality can
    be judged independently of the tracker.

    Inputs
    ------
    config : Config
        Loaded configuration (visualization.fps_out).
    paths : RunPaths
        Run paths (det_log_csv, output_dir).
    video_path : str
        Video the detections were computed on (used as the overlay substrate).

    Outputs
    -------
    None
        Writes ``<stem>_detections_RF-DETR.mp4`` into the output dir.
    """
    from src.visualization import render_detections_video

    print("\n=== Stage 3b: RF-DETR detection overlay video ===")
    render_detections_video(
        video_path=video_path,
        det_log_csv=paths.det_log_csv,
        out_mp4=os.path.join(paths.output_dir, f"{Path(video_path).stem}_detections_RF-DETR.mp4"),
        fps_out=config.visualization.fps_out,
    )


# ===========================================================================
# Stage 4 — vial assignment + ordered IDs, then scoring
# ===========================================================================

def stage_assign(paths: RunPaths, df_wide, fps: float):
    """Stage 4 — map tracks to vials and relabel them to stable ordered IDs.

    Converts the wide tracks to long form (also saved, for the raw-overlay
    renderer), then assigns each track to a vial and gives it a left-to-right
    ordered id, writing ordered_tracks.csv.

    Inputs
    ------
    paths : RunPaths
        Run paths (ocsort_long, roi_json, ordered_csv, output_dir).
    df_wide : pandas.DataFrame
        Wide tracks table from stage_track.
    fps : float
        Video frame rate (passed through for time columns in the assignment).

    Outputs
    -------
    pandas.DataFrame
        df_ordered : long table with vial_id and ordered_id columns.
    """
    from src.stitching import wide_to_long
    from src.roi import assign_vials_and_ordered_ids

    print("\n=== Stage 4: Vial assignment + ordered IDs ===")
    long_df = wide_to_long(df_wide)
    # save OC-SORT long format — used by the raw OC-SORT overlay renderer
    long_df.to_csv(paths.ocsort_long, index=False)

    df_ordered = assign_vials_and_ordered_ids(
        ocsort_csv=paths.ocsort_long,
        roi_json=paths.roi_json,
        out_csv=paths.ordered_csv,
        fps=fps,
    )
    print(f"  ordered_tracks saved: {paths.ordered_csv}  shape: {df_ordered.shape}")
    save_run_params(paths.output_dir, "ordered", {
        "csv": paths.ordered_csv,
        "rows": int(df_ordered.shape[0]),
        "track_count": int(df_ordered["ordered_id"].nunique()),
    })
    return df_ordered


def score_run_metrics(config, paths: RunPaths, tracker, df_wide, df_ordered,
                      vials: dict, fps: float) -> None:
    """Stage 4b — score the run (HOTA if ground truth exists) and write the report.

    Computes the total expected flies, optionally runs HOTA scoring (writing
    hota.json), then generates metrics_report.md. Both scoring and reporting are
    wrapped so a failure here never aborts the pipeline.

    Inputs
    ------
    config : Config
        Loaded configuration (pipeline.expected_per_vial, plus report settings).
    paths : RunPaths
        Run paths (roi_json, output_dir).
    tracker : OCSort
        The tracker from stage_track (source of detection / suppression logs).
    df_wide : pandas.DataFrame
        Wide tracks table.
    df_ordered : pandas.DataFrame
        Ordered / long tracks table from stage_assign.
    vials : dict
        vial id -> (x0, y0, x1, y1) ROIs.
    fps : float
        Video frame rate.

    Outputs
    -------
    None
        Writes hota.json (when GT exists) and metrics_report.md into the output
        dir. Never raises on scoring / report failure.
    """
    from src.metrics import run_diagnostics
    from src.roi import load_vial_rois, load_vial_colors, resolve_vial_expected_counts

    _, n_flies_dict = load_vial_rois(paths.roi_json)
    n_expected = sum(resolve_vial_expected_counts(
        n_flies_dict, vials, config.pipeline.expected_per_vial).values())

    # HOTA scoring (no-op if no ground truth exists for this video).
    # Run BEFORE the metrics report so the report can include the HOTA section
    # from <output_dir>/hota.json. Never break the pipeline if scoring fails.
    try:
        from parameter_tuning.score_run import score_run
        _hota = score_run(paths.output_dir, print_results=False)
        if _hota is not None:
            _row = next((r for r in _hota["summary"] if r["video"] != "COMBINED_SEQ"),
                        _hota["summary"][0])
            print(f"  HOTA: {_row['HOTA']:.3f}  (DetA {_row['DetA']:.3f}, "
                  f"AssA {_row['AssA']:.3f}, LocA {_row['LocA']:.3f})")
        else:
            print("  HOTA: no GT available for this video; skipped")
    except Exception as e:
        print(f"  HOTA scoring skipped: {e}")

    # Metrics report (picks up hota.json automatically if score_run wrote one)
    run_diagnostics(
        tracker=tracker,
        df_wide=df_wide,
        df_ordered=df_ordered,
        n_expected=n_expected,
        fps=fps,
        vial_rois=vials,
        config=config,
        output_dir=paths.output_dir,
        show_plots=False,
        vial_colors=load_vial_colors(paths.roi_json),
    )
    print(f"  Metrics report: {os.path.join(paths.output_dir, 'metrics_report.md')}")


# ===========================================================================
# Stage 5 — overlay videos
# ===========================================================================

def stage_overlays(config, paths: RunPaths, video_path: str, vials: dict) -> None:
    """Stage 5 — render the raw OC-SORT overlay and the ordered-tracks overlay.

    Draws tracks onto the configured substrate video (raw, preprocessed, or
    cropped, per config.visualization.overlay_source): one overlay of the raw
    OC-SORT ids and one of the final ordered ids.

    Inputs
    ------
    config : Config
        Loaded configuration (visualization.overlay_source, visualization.fps_out).
    paths : RunPaths
        Run paths (ocsort_long, ordered_csv, det_log_csv, output_dir).
    video_path : str
        Fallback substrate video when the configured overlay source is absent.
    vials : dict
        vial id -> (x0, y0, x1, y1) ROIs, drawn on both overlays.

    Outputs
    -------
    None
        Writes overlay_raw_ocsort.mp4 and overlay_ordered.mp4 into the output dir.
    """
    from src.visualization import render_vial_overlay_video, render_raw_overlay_video
    from src.roi import load_vial_colors

    print("\n=== Stage 5: Overlay videos ===")
    _overlay_mode = config.visualization.overlay_source
    overlay_video = resolve_overlay_video(paths.output_dir, _overlay_mode) or video_path
    print(f"  overlay_source={_overlay_mode} -> substrate: {overlay_video}")

    det_log_arg = paths.det_log_csv if os.path.exists(paths.det_log_csv) else None
    # Per-vial colours picked in the setup window (empty when none were chosen).
    vial_colors = load_vial_colors(paths.roi_json) if os.path.exists(paths.roi_json) else {}

    # 5a — raw OC-SORT overlay
    raw_overlay_mp4 = os.path.join(paths.output_dir, "overlay_raw_ocsort.mp4")
    render_raw_overlay_video(
        video_path=overlay_video,
        csv_path=paths.ocsort_long,
        out_mp4=raw_overlay_mp4,
        vial_rois=vials,
        det_log_csv=det_log_arg,
        fps_out=config.visualization.fps_out,
    )
    print(f"  Raw OC-SORT overlay: {raw_overlay_mp4}")

    # 5b — ordered/relinked tracks overlay
    ordered_overlay_mp4 = os.path.join(paths.output_dir, "overlay_ordered.mp4")
    render_vial_overlay_video(
        video_path=overlay_video,
        csv_path=paths.ordered_csv,
        out_mp4=ordered_overlay_mp4,
        vial_rois=vials,
        det_log_csv=det_log_arg,
        fps_out=config.visualization.fps_out,
        vial_colors=vial_colors,
    )
    print(f"  Ordered tracks overlay: {ordered_overlay_mp4}")

    save_run_params(paths.output_dir, "outputs", {
        "raw_overlay": raw_overlay_mp4,
        "ordered_overlay": ordered_overlay_mp4,
    })


# ===========================================================================
# CLI + orchestration
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (``--video``, ``--overlay`` / ``--no-overlay``).

    Inputs
    ------
    None

    Outputs
    -------
    argparse.ArgumentParser
        Parser exposing optional ``--video`` and overlay override flags.
    """
    p = argparse.ArgumentParser(
        description="Fly tracking: one raw video -> ocsort_tracks.csv + ordered_tracks.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", default=None,
        help="Video to track. Overrides config.video.raw_path for this run. "
             "If omitted, config.video.raw_path is used.",
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
    p.add_argument(
        "--reuse-roi", action="store_true",
        help="Reuse this video's saved crop + vial ROIs from roi_library.json "
             "and run headless (no drawing GUI). Transient override of "
             "config.roi.use_saved_roi for this run only; used by scripts/app.py "
             "after it captures ROIs, and handy for scripted CLI submissions.",
    )
    return p


def main():
    """Entry point: load config, resolve paths, and run every stage in order.

    Orchestrates the pipeline — resolve the video, derive paths, record metadata,
    then preprocess -> vial ROIs -> track -> detection overlay -> assign -> score
    -> overlays. Holds no behaviour of its own beyond ordering the stages.

    Inputs
    ------
    None
        Reads config.yaml / creds_config.yaml and the process command-line args.

    Outputs
    -------
    None
        Runs the pipeline for its side effects (files written under the run's
        output dir).
    """
    config = load_config(CONFIG_PATH)
    args   = build_parser().parse_args()

    if getattr(getattr(config, "inference", None), "mode", "hosted").strip().lower() == "local":
        import torch  # load before cv2/Qt — avoids fbgemm.dll conflicts on Windows

    # Transient, in-memory override (never written back to config.yaml): when the
    # GUI has just captured ROIs (or a CLI user opts in), reuse saved geometry and
    # skip every drawing GUI for this run.
    if args.reuse_roi:
        config.roi["use_saved_roi"] = True

    raw_video = resolve_raw_video(args, config)
    run_cfg   = config_for_run(config, raw_video, repo_root=REPO_ROOT)
    paths     = build_paths(config, raw_video)
    os.makedirs(paths.output_dir, exist_ok=True)
    print(f"Video:      {raw_video}")
    print(f"Output dir: {paths.output_dir}")

    from src.ui_context import parse_video_context
    api_key, model_id = load_creds(
        config, CREDS_PATH, require_api_key=inference_requires_api_key(config)
    )
    video_context     = parse_video_context(str(raw_video))
    fps               = record_run_metadata(config, paths, raw_video)

    video_path = stage_preprocess(config, paths, raw_video, video_context)
    vials, n_flies_dict = stage_vial_rois(config, paths, raw_video, video_context)
    df_wide, tracker = stage_track(config, paths, video_path, api_key, model_id, vials, n_flies_dict)
    write_overlays = resolve_overlay_enabled(config, args.overlay)
    if write_overlays:
        render_detection_overlay(config, paths, video_path)
    else:
        print("\n=== Overlays skipped (visualization.enabled=false; pass --overlay to force) ===")
    df_ordered = stage_assign(paths, df_wide, fps)
    score_run_metrics(run_cfg, paths, tracker, df_wide, df_ordered, vials, fps)
    if write_overlays:
        stage_overlays(config, paths, video_path, vials)

    print("\nDone.")


if __name__ == "__main__":
    main()
