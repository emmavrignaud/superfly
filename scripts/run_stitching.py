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
    --output-dir outputs/my_run

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
    s = cfg.get("stitching", {})
    v = cfg.get("visualization", {})

    p = argparse.ArgumentParser(
        description="Fly stitching: wide CSV -> compact_tracks.csv + overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--video", required=True, help="Path to the input video (used for fps + overlay)")
    p.add_argument("--output-dir", required=True,
                   help="Directory containing tracks_wide_format.csv and vial_rois.json")

    p.add_argument("--no-overlay", action="store_true", help="Skip overlay video rendering")

    p.add_argument("--gap-penalty", type=float, default=s.get("gap_penalty", 0.05))
    p.add_argument("--max-gap-frames", type=int, default=s.get("max_gap_frames", 90))

    p.add_argument("--fps-out", type=int, default=v.get("fps_out", 30))
    p.add_argument("--overlay-radius", type=int, default=v.get("radius", 5))

    return p


def main():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(str(config_path))
    args = build_parser(cfg).parse_args()

    import cv2
    from src.stitching import stitch_wide_csv_to_long
    from src.roi import assign_vials_and_compact_ids
    from src.visualization import render_vial_overlay_video

    wide_csv     = os.path.join(args.output_dir, "tracks_wide_format.csv")
    roi_json     = os.path.join(args.output_dir, "vial_rois.json")
    stitched_csv = os.path.join(args.output_dir, "tracks_xy_stitched_long.csv")
    compact_csv  = os.path.join(args.output_dir, "compact_tracks.csv")

    # ------------------------------------------------------------------
    # Stage 4: stitch
    # ------------------------------------------------------------------
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
    print("\n=== Stage 5: Vial assignment + compact IDs ===")
    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    df = assign_vials_and_compact_ids(
        stitched_csv=stitched_csv,
        roi_json=roi_json,
        out_csv=compact_csv,
        fps=fps,
    )
    print(f"  compact_tracks saved: {compact_csv}  shape: {df.shape}")

    # ------------------------------------------------------------------
    # Stage 6 (optional): overlay video
    # ------------------------------------------------------------------
    if not args.no_overlay:
        overlay_mp4 = os.path.join(args.output_dir, "overlay_vials_shaded.mp4")
        print("\n=== Stage 6: Overlay video ===")
        render_vial_overlay_video(
            video_path=args.video,
            csv_path=compact_csv,
            out_mp4=overlay_mp4,
            fps_out=args.fps_out,
            radius=args.overlay_radius,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
