#!/usr/bin/env python
"""
scripts/run_stitching.py

CLI: tracks_wide_format.csv -> compact_tracks.csv + overlay video.

Stages
------
4. Hungarian stitching         -- writes tracks_xy_stitched_long.csv
5. Vial assignment + compact IDs -- writes compact_tracks.csv
6. Overlay video rendering     -- writes overlay_vials_shaded.mp4

Run scripts/run_tracking.py first to produce the wide CSV and vial_rois.json.

Usage
-----
python scripts/run_stitching.py \
    --video      data/my_experiment.mp4 \
    --output-dir outputs/run_1

All parameters have defaults from config.yaml.  Use --help for full list.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    s = cfg.get("stitching", {})
    t = cfg.get("tracker", {})
    v = cfg.get("visualization", {})

    p = argparse.ArgumentParser(
        description="Fly stitching: wide CSV -> compact_tracks.csv + overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--video", required=True, help="Path to the input video (used for fps + overlay)")
    p.add_argument("--output-dir", required=True,
                   help="Directory containing tracks_wide_format.csv and vial_rois.json")

    p.add_argument("--no-overlay", action="store_true", help="Skip overlay video rendering")

    p.add_argument("--min-consecutive-frames", type=int,
                   default=t.get("minimum_consecutive_frames", 10),
                   help="Tracklets shorter than this are excluded from feature scale computation")

    p.add_argument("--fps-out", type=int, default=v.get("fps_out", 30))
    p.add_argument("--overlay-radius", type=int, default=v.get("radius", 5))

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(str(config_path))
    args = build_parser(cfg).parse_args()

    import cv2
    import pandas as pd
    from src.stitching import wide_to_long, build_tracklets, stitch
    from src.roi import assign_vials_and_compact_ids
    from src.visualization import render_vial_overlay_video, render_raw_overlay_video

    wide_csv     = os.path.join(args.output_dir, "tracks_wide_format.csv")
    roi_json     = os.path.join(args.output_dir, "vial_rois.json")
    stitched_csv = os.path.join(args.output_dir, "tracks_xy_stitched_long.csv")
    compact_csv  = os.path.join(args.output_dir, "compact_tracks.csv")

    # ------------------------------------------------------------------
    # Stage 4: stitch
    # ------------------------------------------------------------------
    print("\n=== Stage 4: Hungarian stitching ===")

    with open(roi_json) as f:
        vial_rois = {k: tuple(v) for k, v in json.load(f).items()}

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    long_csv  = os.path.join(args.output_dir, "tracks_long_format.csv")
    long_df   = wide_to_long(pd.read_csv(wide_csv), out_csv=long_csv)
    tracklets = build_tracklets(long_df)
    print(f"  Built {len(tracklets)} tracklets from {long_df['orig_id'].nunique()} original IDs")

    stitched_df = stitch(
        long_df    = long_df,
        vial_rois  = vial_rois,
        tracklets  = tracklets,
        output_dir = args.output_dir,
    )
    stitched_df.to_csv(stitched_csv, index=False)
    print(f"  Stitched IDs: {stitched_df['stitched_id'].nunique()} "
          f"(from {stitched_df['orig_id'].nunique()} original)")

    # ------------------------------------------------------------------
    # Stage 5: vial assignment + compact IDs
    # ------------------------------------------------------------------
    print("\n=== Stage 5: Vial assignment + compact IDs ===")

    df = assign_vials_and_compact_ids(
        stitched_csv=stitched_csv,
        roi_json=roi_json,
        out_csv=compact_csv,
        fps=fps,
    )
    print(f"  compact_tracks saved: {compact_csv}  shape: {df.shape}")

    # ------------------------------------------------------------------
    # Stage 6 (optional): overlay videos
    # ------------------------------------------------------------------
    if not args.no_overlay:
        print("\n=== Stage 6: Overlay videos ===")

        # 6a — raw OC-SORT overlay (before stitching)
        raw_overlay_mp4 = os.path.join(args.output_dir, "overlay_raw_ocsort.mp4")
        render_raw_overlay_video(
            video_path=args.video,
            csv_path=long_csv,
            out_mp4=raw_overlay_mp4,
            fps_out=args.fps_out,
            radius=args.overlay_radius,
        )
        print(f"  Raw OC-SORT overlay: {raw_overlay_mp4}")

        # 6b — stitched overlay
        overlay_mp4 = os.path.join(args.output_dir, "overlay_vials_shaded.mp4")
        render_vial_overlay_video(
            video_path=args.video,
            csv_path=compact_csv,
            out_mp4=overlay_mp4,
            fps_out=args.fps_out,
            radius=args.overlay_radius,
        )
        print(f"  Stitched overlay:    {overlay_mp4}")

    print("\nDone.")


if __name__ == "__main__":
    main()
