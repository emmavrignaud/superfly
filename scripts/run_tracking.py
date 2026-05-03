#!/usr/bin/env python
"""
scripts/run_tracking.py

CLI: raw video -> tracks_wide_format.csv

Stages
------
1. (optional) Background subtraction GUI  -- --preprocess flag
2. Interactive vial ROI drawing           -- draws & saves vial_rois.json
3. RF-DETR + OC-SORT tracking            -- writes tracks_wide_format.csv

Then run scripts/run_stitching.py to continue from the wide CSV.

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
import re
import shutil
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import save_run_params

ROI_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "roi_library.json"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_roi_library() -> dict:
    if ROI_LIBRARY_PATH.exists():
        with open(ROI_LIBRARY_PATH) as f:
            return json.load(f)
    return {}


def _save_roi_library(library: dict) -> None:
    ROI_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_LIBRARY_PATH, "w") as f:
        json.dump(library, f, indent=2)


def _auto_output_dir(video_path: str) -> str:
    """Generate outputs/run_N_<day>DPE_n<NNN> from the video path."""
    m = re.search(r"(\d+)\s+DPE[/\\](\d+)", video_path)
    if m:
        short = f"{m.group(1)}DPE_n{m.group(2).zfill(3)}"
    else:
        short = Path(video_path).stem[:20]
    base_tpl = str(Path("outputs") / f"run_{{N}}_{short}")
    n = 0
    while Path(base_tpl.format(N=n)).exists():
        n += 1
    return base_tpl.format(N=n)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    t = cfg.get("tracker", {})

    p = argparse.ArgumentParser(
        description="Fly tracking: raw video -> tracks_wide_format.csv",
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

    p.add_argument("--confidence", type=float, default=t.get("confidence", 0.10))
    p.add_argument("--lost-track-buffer", type=int, default=t.get("lost_track_buffer", 90))
    p.add_argument("--min-matching-threshold", type=float,
                   default=t.get("minimum_matching_threshold", 0.2))
    p.add_argument("--min-consecutive-frames", type=int,
                   default=t.get("minimum_consecutive_frames", 3))
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit number of frames to process (None = all)")
    p.add_argument("--asso-func", type=str, default=t.get("asso_func", "diou"),
                   help="OC-SORT association function: diou, hmiou, or iou")
    p.add_argument("--brownian-pos-noise", type=float,
                   default=t.get("brownian_pos_noise", 1.0),
                   help="Scale factor on Kalman Q[cx], Q[cy] (1.0 = original; higher tolerates saccades)")

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(str(config_path))
    args = build_parser(cfg).parse_args()

    _t = cfg.get("tracker", {})
    _p = cfg.get("preprocessing", {})
    _r = cfg.get("roi", {})
    _rf = cfg.get("roboflow", {})
    inference_api_url = _rf.get("inference_api_url", "https://detect.roboflow.com")
    detection_confidence_rfdetr = _t.get("detection_confidence_rfdetr", 0.4)
    use_saved_roi = _r.get("use_saved_roi", True)

    if args.output_dir is None:
        args.output_dir = _auto_output_dir(args.video)
        print(f"Auto output-dir: {args.output_dir}")

    import cv2
    from src.preprocessing import preprocess_bgsub_gui
    from src.ui_context import parse_video_context
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.roi import draw_and_save_vial_rois
    from src.visualization import render_detections_video

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy/hardlink original video into the run folder (zero disk cost if same filesystem)
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

    # persist config + video metadata
    cap = cv2.VideoCapture(video_path)
    save_run_params(args.output_dir, "config", {
        "video": video_path,
        "video_fps": cap.get(cv2.CAP_PROP_FPS),
        "video_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "video_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "tracker": {
            "detection_confidence_rfdetr": detection_confidence_rfdetr,
            "confidence": args.confidence,
            "lost_track_buffer": args.lost_track_buffer,
            "min_matching_threshold": args.min_matching_threshold,
            "min_consecutive_frames": args.min_consecutive_frames,
            "asso_func": args.asso_func,
            "brownian_pos_noise": args.brownian_pos_noise,
        },
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
            print(f"No stored preprocessing params for: {_video_key} — opening GUI...")

        video_path, crop_params = preprocess_bgsub_gui(
            video_path=video_path,
            out_mp4=pp_out,
            out_raw_mp4=raw_cropped_path,
            gain=_p.get("bg_gain", 1.2),
            white_level=_p.get("bg_white_level", 245),
            bg_sample_stride=_p.get("bg_sample_stride", 1),
            bg_percentile=_p.get("bg_percentile", 85.0),
            crop_params=_stored_crop if (use_saved_roi and _stored_crop is not None) else None,
            video_context=video_context,
        )
        print(f"Preprocessed video: {video_path}")

        # persist to library
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
        vials = {k: tuple(v) for k, v in _stored_vials.items()}
        with open(roi_json, "w") as f:
            json.dump({k: list(v) for k, v in vials.items()}, f, indent=2)
        print(f"Loaded {len(vials)} vials from library.")
    else:
        if not use_saved_roi:
            print("use_saved_roi=False — opening GUI...")
        else:
            print(f"No stored vial ROIs for: {_video_key} — opening GUI...")
        vials = draw_and_save_vial_rois(
            video_path=args.video,
            roi_json_path=roi_json,
            video_context=video_context,
        )

        # persist to library
        if _video_key not in _library:
            _library[_video_key] = {}
        _library[_video_key]["vial_rois"] = {k: list(v) for k, v in vials.items()}
        _save_roi_library(_library)

    save_run_params(args.output_dir, "roi", {k: list(v) for k, v in vials.items()})

    # ------------------------------------------------------------------
    # Stage 3: track
    # ------------------------------------------------------------------
    wide_csv    = os.path.join(args.output_dir, "tracks_wide_format.csv")
    det_log_csv = os.path.join(args.output_dir, "detections_raw.csv")
    print("\n=== Stage 3: RF-DETR + OC-SORT tracking ===")
    df, tracker = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=wide_csv,
        api_key=args.api_key,
        model_id=args.model_id,
        inference_api_url=inference_api_url,
        detection_confidence_rfdetr=detection_confidence_rfdetr,
        confidence=args.confidence,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.min_matching_threshold,
        minimum_consecutive_frames=args.min_consecutive_frames,
        max_frames=args.max_frames,
        asso_func=args.asso_func,
        brownian_pos_noise=args.brownian_pos_noise,
        det_log_csv=det_log_csv,
    )
    print(f"  shape: {df.shape}")
    save_run_params(args.output_dir, "tracker_output", {
        "wide_csv": wide_csv,
        "frames": int(df.shape[0]),
        "track_count": int(df.shape[1] - 1),
    })

    # Save tracker internals so run_stitching.py can generate metrics_report.md
    tracker_log_json = os.path.join(args.output_dir, "tracker_log.json")
    with open(tracker_log_json, "w") as _f:
        json.dump({
            "detection_log":     tracker.detection_log,
            "suppressed_tracks": tracker.suppressed_tracks,
            "min_hits":          tracker.min_hits,
            "max_age":           tracker.max_age,
        }, _f)

    # RF-DETR detection overlay video
    print("\n=== Stage 3b: RF-DETR detection overlay video ===")
    render_detections_video(
        video_path=video_path,
        det_log_csv=det_log_csv,
        out_mp4=os.path.join(args.output_dir, f"{Path(video_path).stem}_detections_RF-DETR.mp4"),
        fps_out=cfg.get("visualization", {}).get("fps_out", 30),
    )

    print("\nDone. Run scripts/run_stitching.py to continue.")


if __name__ == "__main__":
    main()
