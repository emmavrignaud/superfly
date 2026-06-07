#!/usr/bin/env python
"""
scripts/run_all.py

Two-stage pipeline:

  Stage 1 -- Track every video under data/raw/ (or explicit --video paths)
             using the current config.yaml parameters.
             ROIs are loaded from roi_library.json; missing ones open a GUI.
             No overlay videos are written (fast mode).
             Skips videos whose output dir already has ordered_tracks.csv
             unless --force is passed.

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

    # Force re-track even if ordered_tracks.csv already exists
    python scripts/run_all.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")


def _load_creds() -> tuple[str, str]:
    import yaml
    creds_path = REPO_ROOT / "creds_config.yaml"
    if not creds_path.exists():
        raise SystemExit(f"creds_config.yaml not found at {creds_path}")
    with open(creds_path) as f:
        creds = yaml.safe_load(f)
    from utils import load_config
    cfg = load_config(REPO_ROOT / "config.yaml")
    api_key  = creds.get("API_KEY",  "")
    model_id = creds.get("MODEL_ID") or cfg.roboflow.model_id
    if not api_key:
        raise SystemExit("API_KEY missing in creds_config.yaml")
    return api_key, model_id


def _discover_videos(data_root: Path) -> list[Path]:
    """Find all *-converted.mp4 under data_root, excluding _pp and _raw_cropped copies."""
    videos = []
    for p in sorted(data_root.rglob("*-converted.mp4")):
        name = p.name
        if name.endswith("_pp.mp4") or "_raw_cropped" in name:
            continue
        videos.append(p)
    return videos


def _output_dir_for(video: Path, outputs_root: Path) -> Path:
    """Derive output dir name from video path (same logic as make_run_output_dir)."""
    from utils import make_run_output_dir
    return Path(make_run_output_dir(str(video), outputs_root=str(outputs_root)))


def _is_tracked(output_dir: Path) -> bool:
    return (output_dir / "ordered_tracks.csv").exists()


# ---------------------------------------------------------------------------
# Stage 1 -- Tracking
# ---------------------------------------------------------------------------

def track_one(video: Path, output_dir: Path, api_key: str, model_id: str,
              cfg, library: dict) -> bool:
    """
    Track a single video.  Returns True on success.
    Mirrors the logic of run_tracking.py but without overlay rendering.
    """
    import json
    import cv2
    import pandas as pd
    from utils import save_config_snapshot, save_run_params
    from src.preprocessing import preprocess_bgsub_gui
    from src.ui_context import parse_video_context
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.stitching import wide_to_long
    from src.roi import draw_and_save_vial_rois, assign_vials_and_ordered_ids, load_vial_rois
    from src.metrics import run_diagnostics

    output_dir.mkdir(parents=True, exist_ok=True)

    # Link/copy video into output dir
    dest_video = output_dir / video.name
    if not dest_video.exists():
        try:
            os.link(str(video), str(dest_video))
        except Exception:
            import shutil
            shutil.copy2(str(video), str(dest_video))

    save_config_snapshot(str(output_dir), str(REPO_ROOT / "config.yaml"))

    cap = cv2.VideoCapture(str(video))
    fps_actual = float(cap.get(cv2.CAP_PROP_FPS) or cfg.video.fallback_fps)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    save_run_params(str(output_dir), "video", {
        "path": str(video), "fps": fps_actual,
        "frames": n_frames, "model_id": model_id,
    })

    # --- Video key for ROI library ---
    vc = parse_video_context(str(video))
    video_key = vc.video_key if hasattr(vc, "video_key") else video.stem

    # --- Preprocessing ---
    use_saved_roi = cfg.roi.use_saved_roi
    _stored_crop = library.get(video_key, {}).get("preprocessing") if use_saved_roi else None
    raw_cropped_path = None

    if use_saved_roi and _stored_crop is not None:
        print(f"  [preprocess] Using stored crop for {video_key}")
        pp_out         = output_dir / (video.stem + "_pp.mp4")
        raw_cropped_out = output_dir / (video.stem + "_raw_cropped.mp4")
        video_path, _ = preprocess_bgsub_gui(
            str(video), str(pp_out), str(raw_cropped_out),
            crop_params=_stored_crop,
            gain=cfg.preprocessing.bg_gain,
            white_level=cfg.preprocessing.bg_white_level,
            bg_percentile=cfg.preprocessing.bg_percentile,
            bg_sample_stride=cfg.preprocessing.bg_sample_stride,
            codec=cfg.preprocessing.codec,
        )
        raw_cropped_path = raw_cropped_out
    else:
        print(f"  [preprocess] No stored crop -- opening GUI for {video.name}")
        pp_out         = output_dir / (video.stem + "_pp.mp4")
        raw_cropped_out = output_dir / (video.stem + "_raw_cropped.mp4")
        video_path, crop_params = preprocess_bgsub_gui(
            str(video), str(pp_out), str(raw_cropped_out),
            gain=cfg.preprocessing.bg_gain,
            white_level=cfg.preprocessing.bg_white_level,
            bg_percentile=cfg.preprocessing.bg_percentile,
            bg_sample_stride=cfg.preprocessing.bg_sample_stride,
            codec=cfg.preprocessing.codec,
        )
        if video_key not in library:
            library[video_key] = {}
        library[video_key]["preprocessing"] = crop_params
        raw_cropped_path = raw_cropped_out

    # --- Vial ROIs ---
    roi_json = str(output_dir / "vial_rois.json")
    _stored_roi = library.get(video_key, {}).get("vial_rois") if use_saved_roi else None
    if use_saved_roi and _stored_roi is not None:
        print(f"  [roi] Using stored ROI for {video_key}")
        import json as _json
        with open(roi_json, "w") as _f:
            _json.dump(_stored_roi, _f, indent=2)
        _stored_nflies = library.get(video_key, {}).get("n_flies_per_vial")
        if _stored_nflies:
            # Merge n_flies into vial_rois.json format
            combined = {v: {"bbox": b, "n_flies": _stored_nflies.get(v, 0)}
                        for v, b in _stored_roi.items()}
            with open(roi_json, "w") as _f:
                _json.dump(combined, _f, indent=2)
    else:
        print(f"  [roi] Opening ROI GUI for {video.name}")
        overlay_src = raw_cropped_path if raw_cropped_path and raw_cropped_path.exists() else video_path
        draw_and_save_vial_rois(
            str(overlay_src), roi_json,
        )
        bbox_dict, _ = load_vial_rois(roi_json)
        if video_key not in library:
            library[video_key] = {}
        library[video_key]["vial_rois"] = bbox_dict

    # --- Tracking ---
    print(f"  [track] Running RF-DETR + OC-SORT on {video.name}...")
    _gd = cfg.tracker.ghost_detection
    ocsort_csv = str(output_dir / "ocsort_tracks.csv")
    det_csv    = str(output_dir / "detections_raw.csv")
    _vial_rois, _ = load_vial_rois(roi_json)
    df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
        video_path=video_path,
        output_csv=ocsort_csv,
        api_key=api_key,
        model_id=model_id,
        confidence=cfg.tracker.confidence,
        detection_confidence_rfdetr=cfg.tracker.detection_confidence_rfdetr,
        lost_track_buffer=cfg.tracker.lost_track_buffer,
        minimum_matching_threshold=cfg.tracker.minimum_matching_threshold,
        minimum_consecutive_frames=cfg.tracker.minimum_consecutive_frames,
        min_area=cfg.tracker.min_area,
        asso_func=cfg.tracker.asso_func,
        brownian_pos_noise=cfg.tracker.brownian_pos_noise,
        det_log_csv=det_csv,
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
    )

    # Save tracker log
    import json as _json
    tracker_log = str(output_dir / "tracker_log.json")
    with open(tracker_log, "w") as _f:
        _json.dump({
            "detection_log":      tracker.detection_log,
            "suppressed_tracks":  tracker.suppressed_tracks,
            "min_hits":           tracker.min_hits,
            "max_age":            tracker.max_age,
            "ghost_log":          getattr(tracker, "ghost_log", []),
        }, _f)

    # --- Vial assignment -> ordered_tracks.csv ---
    print("  [assign] Assigning vials + ordered IDs...")
    df_long = wide_to_long(df_wide)
    long_csv = str(output_dir / "tracks_long_format.csv")
    df_long.to_csv(long_csv, index=False)
    ordered_csv = str(output_dir / "ordered_tracks.csv")
    assign_vials_and_ordered_ids(long_csv, roi_json, ordered_csv, fps=fps_actual)
    import pandas as _pd
    df_ord = _pd.read_csv(ordered_csv)
    print(f"  [done] {len(df_ord['ordered_id'].unique())} tracks -> {ordered_csv}")

    # --- Diagnostics (no GUI) ---
    try:
        import types
        tracker_ns = types.SimpleNamespace(
            detection_log=tracker.detection_log,
            suppressed_tracks=tracker.suppressed_tracks,
            min_hits=tracker.min_hits,
            max_age=tracker.max_age,
        )
        bbox_dict, n_flies_dict = load_vial_rois(roi_json)
        run_diagnostics(
            df_ord, tracker_ns, str(output_dir),
            vial_rois=bbox_dict, n_flies_dict=n_flies_dict,
            fps=fps_actual, show=False,
        )
    except Exception as e:
        print(f"  [warn] Diagnostics failed: {e}")

    return True


# ---------------------------------------------------------------------------
# Stage 2 -- Classification analysis
# ---------------------------------------------------------------------------

def run_analysis(run_dirs: list[Path], out_dir: Path) -> None:
    import pandas as pd
    from src.classification import map_vial_to_genotype
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
            d = map_vial_to_genotype(str(rd))
            run_tag = rd.name
            d["run"]        = run_tag
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
                     .set_index("ordered_id")[["genotype", "run"]])
    df_agg  = df_agg.join(meta, on="ordered_id").dropna(subset=["genotype"])
    print(f"  Aggregated: {df_agg.shape[0]} flies x {df_agg.shape[1]} columns")

    FEATURES = classification_feature_columns()
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
    )
    print(f"\n  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
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
        "--outputs-root", default="outputs",
        help="Where tracked run directories are written.",
    )
    p.add_argument(
        "--analysis-out", default="outputs/analysis/significance_report",
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
        "--force", action="store_true",
        help="Re-track even if ordered_tracks.csv already exists.",
    )
    return p


def main() -> None:
    import json
    from utils import load_config

    args  = build_parser().parse_args()
    cfg   = load_config(REPO_ROOT / "config.yaml")

    outputs_root = (REPO_ROOT / args.outputs_root).resolve()
    analysis_out = (REPO_ROOT / args.analysis_out).resolve()

    # ------------------------------------------------------------------ #
    # Stage 1 -- Tracking                                                  #
    # ------------------------------------------------------------------ #
    if not args.skip_tracking:
        api_key, model_id = _load_creds()

        if args.videos:
            videos = [Path(v).expanduser().resolve() for v in args.videos]
        else:
            data_root = (REPO_ROOT / args.data_root).resolve()
            videos    = _discover_videos(data_root)

        if not videos:
            print(f"No videos found under {args.data_root}")
            sys.exit(1)

        _banner(f"Stage 1 -- Tracking {len(videos)} video(s)")

        # Load ROI library
        roi_lib_path = REPO_ROOT / "roi_library.json"
        library: dict = {}
        if roi_lib_path.exists():
            with open(roi_lib_path) as f:
                library = json.load(f)

        tracked_dirs: list[Path] = []
        failed: list[Path] = []

        for video in videos:
            out_dir = _output_dir_for(video, outputs_root)
            if not args.force and _is_tracked(out_dir):
                print(f"\n[skip] Already tracked: {out_dir.name}")
                tracked_dirs.append(out_dir)
                continue

            print(f"\n[track] {video.name}  ->  {out_dir.name}")
            try:
                ok = track_one(video, out_dir, api_key, model_id, cfg, library)
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
        # Collect all run dirs that have ordered_tracks.csv
        run_dirs = sorted([
            d for d in outputs_root.iterdir()
            if d.is_dir() and (d / "ordered_tracks.csv").exists()
        ])
        if not run_dirs:
            print("No tracked runs found -- skipping analysis.")
            return
        print(f"\nFound {len(run_dirs)} tracked run(s) for analysis.")
        run_analysis(run_dirs, analysis_out)


if __name__ == "__main__":
    main()
