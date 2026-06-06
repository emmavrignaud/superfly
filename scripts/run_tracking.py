#!/usr/bin/env python
"""
scripts/run_tracking.py

CLI: raw video -> ocsort_tracks.csv + ordered_tracks.csv + overlay videos

Stages
------
1. (optional) Background subtraction GUI  -- --preprocess flag
2. Interactive vial ROI drawing           -- draws & saves vial_rois.json
3. RF-DETR + OC-SORT tracking            -- writes ocsort_tracks.csv + detections_raw.csv
3b. RF-DETR detection overlay video
4. Vial assignment + ordered IDs         -- writes ordered_tracks.csv
5. Overlay videos                        -- overlay_raw_ocsort.mp4 + overlay_ordered.mp4

Usage
-----
python scripts\\run_tracking.py ^
    --video      data\\my_experiment.mp4 ^
    --output-dir outputs\\my_run ^
    --api-key    YOUR_ROBOFLOW_KEY ^
    --model-id   YOUR_MODEL_ID

All parameters have defaults from config.yaml.  Use --help for full list.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (
    load_config,
    make_run_output_dir,
    resolve_overlay_video,
    save_config_snapshot,
    save_run_params,
)

ROI_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "roi_library.json"


def _load_roi_library() -> dict:
    if ROI_LIBRARY_PATH.exists():
        with open(ROI_LIBRARY_PATH) as f:
            return json.load(f)
    return {}


def _save_roi_library(library: dict) -> None:
    ROI_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_LIBRARY_PATH, "w") as f:
        json.dump(library, f, indent=2)


def _n_expected_from_rois(roi_json: str, cfg) -> int:
    """
    Return total expected fly count.  Uses per-vial n_flies from the JSON when
    available (new format); falls back to pipeline.expected_per_vial × n_vials
    for old-format files.
    """
    from src.roi import load_vial_rois
    bbox_dict, n_flies_dict = load_vial_rois(roi_json)
    total = sum(n_flies_dict.values())
    if total > 0:
        return total
    return cfg.pipeline.expected_per_vial * len(bbox_dict)


def build_parser(cfg) -> argparse.ArgumentParser:
    t = cfg.tracker
    p = argparse.ArgumentParser(
        description="Fly tracking: raw video -> ocsort_tracks.csv + ordered_tracks.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--video", required=True, help="Path to the raw input video")
    p.add_argument("--output-dir", default=None,
                   help="Directory for all outputs. If omitted, auto-generated as "
                        "outputs/run_N_<DPE>DPE_n<NNN> from the video path.")
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--model-id", required=True, help="Roboflow model ID (e.g. flies-123/1)")

    p.add_argument("--preprocess", action="store_true",
                   help="Run interactive background-subtraction GUI before tracking")
    p.add_argument("--no-overlay", action="store_true", help="Skip overlay video rendering")

    p.add_argument("--confidence", type=float, default=t.confidence)
    p.add_argument("--lost-track-buffer", type=int, default=t.lost_track_buffer)
    p.add_argument("--min-matching-threshold", type=float, default=t.minimum_matching_threshold)
    p.add_argument("--min-consecutive-frames", type=int, default=t.minimum_consecutive_frames)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit number of frames to process (None = all)")
    p.add_argument("--asso-func", type=str, default=t.asso_func,
                   help="OC-SORT association function: diou, hmiou, or iou")
    p.add_argument("--brownian-pos-noise", type=float, default=t.brownian_pos_noise,
                   help="Scale factor on Kalman Q[cx], Q[cy]")
    p.add_argument("--fps-out", type=int, default=cfg.visualization.fps_out)

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(config_path)
    args = build_parser(cfg).parse_args()
    use_saved_roi = cfg.roi.use_saved_roi

    if args.output_dir is None:
        args.output_dir = make_run_output_dir(args.video)
        print(f"Auto output-dir: {args.output_dir}")

    import cv2
    import pandas as pd
    from src.preprocessing import preprocess_bgsub_gui
    from src.ui_context import parse_video_context
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.stitching import wide_to_long
    from src.roi import draw_and_save_vial_rois, assign_vials_and_ordered_ids, load_vial_rois
    from src.visualization import render_detections_video, render_vial_overlay_video, render_raw_overlay_video
    from src.metrics import run_diagnostics

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy/hardlink original video into the run folder
    dest_video = os.path.join(args.output_dir, Path(args.video).name)
    if not os.path.exists(dest_video):
        try:
            os.link(args.video, dest_video)
        except OSError:
            shutil.copy2(args.video, dest_video)

    video_path = args.video
    _video_key = Path(args.video).stem
    _library   = _load_roi_library()
    video_context = parse_video_context(args.video)

    save_config_snapshot(args.output_dir, config_path)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or cfg.video.fallback_fps)
    save_run_params(args.output_dir, "video", {
        "path": video_path,
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    })
    cap.release()

    # ------------------------------------------------------------------
    # Stage 1 (optional): background subtraction
    # ------------------------------------------------------------------
    crop_params = None
    raw_cropped_path = None
    if args.preprocess:
        print("\n=== Stage 1: Background subtraction ===")
        pp_out = os.path.join(args.output_dir, Path(args.video).stem + "_pp.mp4")
        raw_cropped_path = os.path.join(args.output_dir, Path(args.video).stem + "_raw_cropped.mp4")

        _stored_crop = _library.get(_video_key, {}).get("preprocessing")
        if use_saved_roi and _stored_crop is not None:
            print(f"Found stored preprocessing params for: {_video_key}")
        else:
            print(f"No stored preprocessing params for: {_video_key}: opening GUI...")

        video_path, crop_params = preprocess_bgsub_gui(
            video_path=video_path,
            out_mp4=pp_out,
            out_raw_mp4=raw_cropped_path,
            gain=cfg.preprocessing.bg_gain,
            white_level=cfg.preprocessing.bg_white_level,
            bg_sample_stride=cfg.preprocessing.bg_sample_stride,
            bg_percentile=cfg.preprocessing.bg_percentile,
            crop_params=_stored_crop if (use_saved_roi and _stored_crop is not None) else None,
            video_context=video_context,
        )
        print(f"Preprocessed video: {video_path}")

        if _video_key not in _library:
            _library[_video_key] = {}
        _library[_video_key]["preprocessing"] = crop_params
        _library[_video_key]["video_path"] = args.video
        _save_roi_library(_library)

        crop_roi_json = os.path.join(args.output_dir, "crop_roi.json")
        with open(crop_roi_json, "w") as _f:
            json.dump(crop_params, _f, indent=2)
        print(f"Saved crop params: {crop_roi_json}")

    save_run_params(args.output_dir, "preprocessing",
                    {"video_pp": video_path, "video_raw_cropped": raw_cropped_path, "crop_params": crop_params})

    # ------------------------------------------------------------------
    # Stage 2: draw vial ROIs
    # ------------------------------------------------------------------
    roi_json = os.path.join(args.output_dir, "vial_rois.json")
    print("\n=== Stage 2: Draw vial ROIs ===")

    _stored_vials = _library.get(_video_key, {}).get("vial_rois")
    if use_saved_roi and _stored_vials is not None:
        print(f"Found stored vial ROIs for: {_video_key}")
        with open(roi_json, "w") as f:
            json.dump(_stored_vials, f, indent=2)
        vials, n_flies_dict = load_vial_rois(roi_json)
        print(f"Loaded {len(vials)} vials from library.")
    else:
        if not use_saved_roi:
            print("use_saved_roi=False: opening GUI...")
        else:
            print(f"No stored vial ROIs for: {_video_key}: opening GUI...")
        vials = draw_and_save_vial_rois(
            video_path=args.video,
            roi_json_path=roi_json,
            video_context=video_context,
        )
        # Load n_flies from the JSON the GUI just wrote
        _, n_flies_dict = load_vial_rois(roi_json)

        if _video_key not in _library:
            _library[_video_key] = {}
        # Save the full new-format JSON content (includes n_flies when available)
        with open(roi_json) as _f:
            _library[_video_key]["vial_rois"] = json.load(_f)
        _save_roi_library(_library)

    save_run_params(args.output_dir, "roi", {k: list(v) for k, v in vials.items()})

    # ------------------------------------------------------------------
    # Stage 3: RF-DETR + OC-SORT tracking
    # ------------------------------------------------------------------
    ocsort_csv  = os.path.join(args.output_dir, "ocsort_tracks.csv")
    det_log_csv = os.path.join(args.output_dir, "detections_raw.csv")
    print("\n=== Stage 3: RF-DETR + OC-SORT tracking ===")
    _gd = cfg.tracker.ghost_detection
    df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=ocsort_csv,
        api_key=args.api_key,
        model_id=args.model_id,
        inference_api_url=cfg.roboflow.inference_api_url,
        detection_confidence_rfdetr=cfg.tracker.detection_confidence_rfdetr,
        confidence=args.confidence,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.min_matching_threshold,
        minimum_consecutive_frames=args.min_consecutive_frames,
        max_frames=args.max_frames,
        asso_func=args.asso_func,
        brownian_pos_noise=args.brownian_pos_noise,
        det_log_csv=det_log_csv,
        vial_rois=vials,
        watershed_cfg=dict(cfg.watershed) if hasattr(cfg, "watershed") else None,
        vial_expected_counts=n_flies_dict if any(n_flies_dict.values()) else None,
        ghost_detection_enabled=_gd.enabled,
        ghost_offset_fraction=_gd.offset_fraction,
        ghost_confidence=_gd.confidence,
        ghost_occlusion_max_gap=_gd.occlusion_max_gap,
        ghost_top_exit_px=_gd.top_exit_px,
    )
    print(f"  shape: {df_wide.shape}")
    save_run_params(args.output_dir, "tracker_output", {
        "ocsort_csv": ocsort_csv,
        "frames": int(df_wide.shape[0]),
        "track_count": int(df_wide.shape[1] - 1),
    })

    tracker_log_json = os.path.join(args.output_dir, "tracker_log.json")
    with open(tracker_log_json, "w") as _f:
        json.dump({
            "detection_log":      tracker.detection_log,
            "suppressed_tracks":  tracker.suppressed_tracks,
            "min_hits":           tracker.min_hits,
            "max_age":            tracker.max_age,
            "ghost_log":          getattr(tracker, "ghost_log",          []),
            "top_exit_events":    getattr(tracker, "top_exit_events",    []),
            "top_reentry_events": getattr(tracker, "top_reentry_events", []),
        }, _f)

    print("\n=== Stage 3b: RF-DETR detection overlay video ===")
    render_detections_video(
        video_path=video_path,
        det_log_csv=det_log_csv,
        out_mp4=os.path.join(args.output_dir, f"{Path(video_path).stem}_detections_RF-DETR.mp4"),
        fps_out=cfg.visualization.fps_out,
    )

    # ------------------------------------------------------------------
    # Stage 4: vial assignment + ordered IDs
    # ------------------------------------------------------------------
    ordered_csv = os.path.join(args.output_dir, "ordered_tracks.csv")
    print("\n=== Stage 4: Vial assignment + ordered IDs ===")

    long_df = wide_to_long(df_wide)
    # save OC-SORT long format — used by the raw OC-SORT overlay renderer
    ocsort_long = os.path.join(args.output_dir, "ocsort_long.csv")
    long_df.to_csv(ocsort_long, index=False)

    df_ordered = assign_vials_and_ordered_ids(
        ocsort_csv=ocsort_long,
        roi_json=roi_json,
        out_csv=ordered_csv,
        fps=fps,
    )
    print(f"  ordered_tracks saved: {ordered_csv}  shape: {df_ordered.shape}")
    save_run_params(args.output_dir, "ordered", {
        "csv": ordered_csv,
        "rows": int(df_ordered.shape[0]),
        "track_count": int(df_ordered["ordered_id"].nunique()),
    })

    # HOTA scoring (no-op if no ground truth exists for this video).
    # Run BEFORE the metrics report so the report can include the HOTA section
    # from <output_dir>/hota.json. Never break the pipeline if scoring fails.
    try:
        from parameter_tuning.score_run import score_run
        _hota = score_run(args.output_dir, print_results=False)
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
        n_expected=_n_expected_from_rois(roi_json, cfg),
        fps=fps,
        vial_rois=vials,
        config=cfg,
        output_dir=args.output_dir,
        show_plots=False,
    )
    print(f"  Metrics report: {os.path.join(args.output_dir, 'metrics_report.md')}")

    # ------------------------------------------------------------------
    # Stage 5 (optional): overlay videos
    # ------------------------------------------------------------------
    if not args.no_overlay:
        print("\n=== Stage 5: Overlay videos ===")

        _overlay_mode = cfg.visualization.overlay_source
        overlay_video = resolve_overlay_video(args.output_dir, _overlay_mode) or video_path
        print(f"  overlay_source={_overlay_mode} -> substrate: {overlay_video}")

        det_log_arg = det_log_csv if os.path.exists(det_log_csv) else None

        # 5a — raw OC-SORT overlay
        raw_overlay_mp4 = os.path.join(args.output_dir, "overlay_raw_ocsort.mp4")
        render_raw_overlay_video(
            video_path=overlay_video,
            csv_path=ocsort_long,
            out_mp4=raw_overlay_mp4,
            vial_rois=vials,
            det_log_csv=det_log_arg,
            fps_out=args.fps_out,
        )
        print(f"  Raw OC-SORT overlay: {raw_overlay_mp4}")

        # 5b — ordered/relinked tracks overlay
        ordered_overlay_mp4 = os.path.join(args.output_dir, "overlay_ordered.mp4")
        render_vial_overlay_video(
            video_path=overlay_video,
            csv_path=ordered_csv,
            out_mp4=ordered_overlay_mp4,
            vial_rois=vials,
            det_log_csv=det_log_arg,
            fps_out=args.fps_out,
        )
        print(f"  Ordered tracks overlay: {ordered_overlay_mp4}")

        save_run_params(args.output_dir, "outputs", {
            "raw_overlay": raw_overlay_mp4,
            "ordered_overlay": ordered_overlay_mp4,
        })

    print("\nDone.")


if __name__ == "__main__":
    main()
