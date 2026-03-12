#!/usr/bin/env python
"""
scripts/run_tracking.py

CLI: raw video -> compact_tracks.csv + overlay video.

Stages
------
1. (optional) Background subtraction GUI       -- --preprocess flag
2. Interactive vial ROI drawing                -- draws & saves vial_rois.json
3. RF-DETR + OC-SORT tracking                 -- writes tracks_wide_format.csv
4. Hungarian stitching                        -- writes tracks_xy_stitched_long.csv
5. Vial assignment + compact IDs              -- writes compact_tracks.csv
6. Overlay video rendering                    -- writes overlay_vials_shaded.mp4

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

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    t = cfg.get("tracker", {})
    s = cfg.get("stitching", {})
    v = cfg.get("visualization", {})

    p = argparse.ArgumentParser(
        description="Fly tracking pipeline: raw video -> compact_tracks.csv + overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--video", required=True, help="Path to the raw input video")
    p.add_argument("--output-dir", required=True, help="Directory for all outputs")
    p.add_argument("--api-key", required=True, help="Roboflow API key")
    p.add_argument("--model-id", required=True, help="Roboflow model ID (e.g. flies-123/1)")

    # Optional pipeline flags
    p.add_argument("--preprocess", action="store_true",
                   help="Run interactive background-subtraction GUI before tracking")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip overlay video rendering")

    # Tracker params
    p.add_argument("--confidence", type=float, default=t.get("confidence", 0.10))
    p.add_argument("--lost-track-buffer", type=int, default=t.get("lost_track_buffer", 90))
    p.add_argument("--min-matching-threshold", type=float,
                   default=t.get("minimum_matching_threshold", 0.01))
    p.add_argument("--min-consecutive-frames", type=int,
                   default=t.get("minimum_consecutive_frames", 10))
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit number of frames to process (None = all)")

    # Stitching params
    p.add_argument("--gap-penalty", type=float, default=s.get("gap_penalty", 0.05))
    p.add_argument("--max-gap-frames", type=int, default=s.get("max_gap_frames", 90))

    # Overlay params
    p.add_argument("--fps-out", type=int, default=v.get("fps_out", 30))
    p.add_argument("--overlay-radius", type=int, default=v.get("radius", 5))

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(str(config_path))
    args = build_parser(cfg).parse_args()

    # Defer heavy imports until after --help is resolved
    import cv2
    from src.preprocessing import preprocess_bgsub_gui_cv2_avg_background
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.stitching import stitch_wide_csv_to_long
    from src.roi import draw_and_save_vial_rois, assign_vials_and_compact_ids
    from src.visualization import render_vial_overlay_video

    os.makedirs(args.output_dir, exist_ok=True)

    video_path = args.video

    # ------------------------------------------------------------------
    # Stage 1 (optional): background subtraction
    # ------------------------------------------------------------------
    if args.preprocess:
        print("\n=== Stage 1: Background subtraction ===")
        video_path = preprocess_bgsub_gui_cv2_avg_background(
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
    export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=wide_csv,
        api_key=args.api_key,
        model_id=args.model_id,
        confidence=args.confidence,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.min_matching_threshold,
        minimum_consecutive_frames=args.min_consecutive_frames,
        max_frames=args.max_frames,
    )

    # ------------------------------------------------------------------
    # Stage 4: stitch
    # ------------------------------------------------------------------
    stitched_csv = os.path.join(args.output_dir, "tracks_xy_stitched_long.csv")
    print("\n=== Stage 4: Hungarian stitching ===")
    stats = stitch_wide_csv_to_long(
        input_csv=wide_csv,
        output_stitched_long=stitched_csv,
        max_gap=args.max_gap_frames,
        gap_penalty=args.gap_penalty,
    )
    print(f"  tracklets: {stats['n_orig_tracklets']}  links: {stats['n_links']}  "
          f"sigma_step: {stats['sigma_step']:.2f}")

    # ------------------------------------------------------------------
    # Stage 5: vial assignment + compact IDs
    # ------------------------------------------------------------------
    compact_csv = os.path.join(args.output_dir, "compact_tracks.csv")
    print("\n=== Stage 5: Vial assignment + compact IDs ===")
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    assign_vials_and_compact_ids(
        stitched_csv=stitched_csv,
        roi_json=roi_json,
        out_csv=compact_csv,
        fps=fps,
    )
    print(f"  compact_tracks saved: {compact_csv}")

    # ------------------------------------------------------------------
    # Stage 6 (optional): overlay video
    # ------------------------------------------------------------------
    if not args.no_overlay:
        overlay_mp4 = os.path.join(args.output_dir, "overlay_vials_shaded.mp4")
        print("\n=== Stage 6: Overlay video ===")
        render_vial_overlay_video(
            video_path=video_path,
            csv_path=compact_csv,
            out_mp4=overlay_mp4,
            fps_out=args.fps_out,
            radius=args.overlay_radius,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
