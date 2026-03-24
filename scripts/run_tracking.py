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
python scripts/run_tracking.py \
    --video      data/my_experiment.mp4 \
    --output-dir outputs/my_run \
    --api-key    YOUR_ROBOFLOW_KEY \
    --model-id   YOUR_MODEL_ID

All parameters have defaults from config.yaml.  Use --help for full list.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    t = cfg.get("tracker", {})

    p = argparse.ArgumentParser(
        description="Fly tracking: raw video -> tracks_wide_format.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--video", required=True, help="Path to the raw input video")
    p.add_argument("--output-dir", required=True, help="Directory for all outputs")
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--model-id", required=True, help="Roboflow model ID (e.g. flies-123/1)")

    p.add_argument("--preprocess", action="store_true",
                   help="Run interactive background-subtraction GUI before tracking")

    p.add_argument("--confidence", type=float, default=t.get("confidence", 0.10))
    p.add_argument("--lost-track-buffer", type=int, default=t.get("lost_track_buffer", 90))
    p.add_argument("--min-matching-threshold", type=float,
                   default=t.get("minimum_matching_threshold", 0.01))
    p.add_argument("--min-consecutive-frames", type=int,
                   default=t.get("minimum_consecutive_frames", 10))
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit number of frames to process (None = all)")
    p.add_argument("--asso-func", type=str, default=t.get("asso_func", "hmiou"),
                   help="OC-SORT association function: hmiou (recommended) or iou")

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(str(config_path))
    args = build_parser(cfg).parse_args()

    from src.preprocessing import preprocess_bgsub_gui_cv2_median_background
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.roi import draw_and_save_vial_rois

    os.makedirs(args.output_dir, exist_ok=True)

    video_path = args.video

    # ------------------------------------------------------------------
    # Stage 1 (optional): background subtraction
    # ------------------------------------------------------------------
    if args.preprocess:
        print("\n=== Stage 1: Background subtraction ===")
        video_path = preprocess_bgsub_gui_cv2_median_background(
            video_path=video_path,
            out_mp4=None,
        )
        print(f"Preprocessed video: {video_path}")

    # ------------------------------------------------------------------
    # Stage 2: draw vial ROIs
    # ------------------------------------------------------------------
    roi_json = os.path.join(args.output_dir, "vial_rois.json")
    print("\n=== Stage 2: Draw vial ROIs ===")
    draw_and_save_vial_rois(video_path=video_path, roi_json_path=roi_json)

    # ------------------------------------------------------------------
    # Stage 3: track
    # ------------------------------------------------------------------
    wide_csv = os.path.join(args.output_dir, "tracks_wide_format.csv")
    print("\n=== Stage 3: RF-DETR + OC-SORT tracking ===")
    df = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=wide_csv,
        api_key=args.api_key,
        model_id=args.model_id,
        confidence=args.confidence,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.min_matching_threshold,
        minimum_consecutive_frames=args.min_consecutive_frames,
        max_frames=args.max_frames,
        asso_func=args.asso_func,
    )
    print(f"  shape: {df.shape}")
    print("\nDone. Run scripts/run_stitching.py to continue.")


if __name__ == "__main__":
    main()