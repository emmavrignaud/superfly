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
python scripts\\run_stitching.py ^
    --video      data\\my_experiment.mp4 ^
    --output-dir outputs\\run_1

All parameters have defaults from config.yaml.  Use --help for full list.
"""

import argparse
import json
import os
import sys
import types
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — must precede any pyplot import

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import save_run_params


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
    p.add_argument("--overlay-tick-len", type=int, default=v.get("tick_len", 4),
                   help="Half-length of crosshair arms at each fly (pixels)")

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

    v_cfg = cfg.get("visualization", {})
    overlay_kwargs = dict(
        fps_out=args.fps_out,
        show_ids=v_cfg.get("show_ids", True),
        tick_len=args.overlay_tick_len,
        tick_thick=v_cfg.get("tick_thick", 1),
        chip_font_scale=v_cfg.get("chip_font_scale", 0.4),
        label_offset_x=v_cfg.get("label_offset_x", 10),
        label_offset_y=v_cfg.get("label_offset_y", -10),
        chip_pad=v_cfg.get("chip_pad", 2),
        leader_thick=v_cfg.get("leader_thick", 1),
    )

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
    save_run_params(args.output_dir, "stitching_output", {
        "stitched_csv": stitched_csv,
        "stitched_ids": int(stitched_df["stitched_id"].nunique()),
        "original_ids": int(stitched_df["orig_id"].nunique()),
    })

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
    save_run_params(args.output_dir, "compact", {
        "csv": compact_csv,
        "rows": int(df.shape[0]),
    })

    # --- Stitching quality objectives ---
    from src.metrics import compute_stitching_objectives, print_stitching_objectives
    num_frames  = int(long_df["frame"].max()) + 1
    s = cfg.get("stitching", {})
    objectives = compute_stitching_objectives(
        df_stitched       = stitched_df,
        vial_rois         = vial_rois,
        num_frames        = num_frames,
        expected_per_vial = s.get("expected_per_vial", 7),
        short_frac        = s.get("short_track_frac", 0.10),
    )
    print_stitching_objectives(objectives)
    save_run_params(args.output_dir, "stitching_objectives", {k: float(v) for k, v in objectives.items()})

    # ------------------------------------------------------------------
    # Metrics report (requires tracker_log.json saved by run_tracking.py)
    # ------------------------------------------------------------------
    tracker_log_path = os.path.join(args.output_dir, "tracker_log.json")
    if os.path.exists(tracker_log_path):
        from src.metrics import run_diagnostics
        with open(tracker_log_path) as _f:
            _tl = json.load(_f)
        mock_tracker = types.SimpleNamespace(**_tl)
        df_wide = pd.read_csv(wide_csv)
        run_diagnostics(
            tracker=mock_tracker,
            df_wide=df_wide,
            df_stitched=stitched_df,
            df_compact=df,
            n_expected=s.get("expected_per_vial", 7) * len(vial_rois),
            fps=fps,
            vial_rois=vial_rois,
            config=cfg,
            output_dir=args.output_dir,
            stitching_objectives=objectives,
        )
        print(f"  Metrics report: {os.path.join(args.output_dir, 'metrics_report.md')}")
    else:
        print("  tracker_log.json not found — skipping metrics report (run via run_tracking.py to generate it)")

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
            **overlay_kwargs,
        )
        print(f"  Raw OC-SORT overlay: {raw_overlay_mp4}")

        # 6b — stitched overlay
        overlay_mp4 = os.path.join(args.output_dir, "overlay_vials_shaded.mp4")
        render_vial_overlay_video(
            video_path=args.video,
            csv_path=compact_csv,
            out_mp4=overlay_mp4,
            **overlay_kwargs,
        )
        print(f"  Stitched overlay:    {overlay_mp4}")
        save_run_params(args.output_dir, "outputs", {
            "raw_overlay": raw_overlay_mp4,
            "overlay": overlay_mp4,
        })

    print("\nDone.")


if __name__ == "__main__":
    main()
