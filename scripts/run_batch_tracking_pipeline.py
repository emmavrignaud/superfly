#!/usr/bin/env python
"""
Batch tracking pipeline — parity with notebooks/01_tracking_pipeline.ipynb.

**Interactive order (per video):** for each job, preprocessing (if enabled in
``config.yaml``) then vial ROIs — so you finish one video's GUIs before the
next. **Automated tail:** tracking → stitching → compact/metrics → overlays is
run **per video** end-to-end (simple, API-friendly); order within that chain
matches the notebook.

**Tracker for diagnostics:** the live OC-SORT object exists only right after
tracking. For later ``run_diagnostics`` we do **not** pickle the tracker; we
reload ``tracker_log.json`` into a ``SimpleNamespace`` (same approach as
``scripts/run_stitching.py``), which is enough for ``compute_tracker_stats`` and
the pipeline figure.

**Config:** Background subtraction always runs (notebook parity). Whether the
preprocessing **GUI** opens follows ``roi.use_saved_roi`` and ``roi_library.json``
(same as ``run_tracking.py`` / the notebook). ``roboflow.model_id`` in
``config.yaml`` sets the RF-DETR model; optional ``MODEL_ID`` in
``creds_config.yaml`` overrides. ``API_KEY`` stays in creds.

Plotly ``.show()`` is OFF unless ``--show-plots`` (batch-friendly).

Usage (from repo root)::

    python scripts/run_batch_tracking_pipeline.py --dpe-root "path/to/24 DPE"

    python scripts/run_batch_tracking_pipeline.py \\
        --video "rel/a-converted.mp4" --video "rel/b-converted.mp4"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ROI_LIBRARY_PATH = REPO_ROOT / "roi_library.json"


@dataclass
class VideoJob:
    raw_video: Path
    output_dir: Path
    short_name: str
    video_key: str


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
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


def short_name_from_path(video_path: Path) -> str:
    m = re.search(r"(\d+)\s+DPE[/\\](\d+)", str(video_path))
    if m:
        return f"{m.group(1)}DPE_n{m.group(2).zfill(3)}"
    return video_path.stem[:20]


def _pick_converted_mp4(folder: Path) -> Path:
    mp4s = list(folder.glob("*.mp4"))
    chosen: list[Path] = []
    for p in mp4s:
        s = p.stem
        if s.endswith("-converted") and not s.endswith("-converted-video"):
            chosen.append(p)
    if len(chosen) != 1:
        names = [p.name for p in mp4s]
        raise SystemExit(
            f"{folder}: need exactly one '*-converted.mp4' (not '*-converted-video'), "
            f"found {len(chosen)}. Files: {names}"
        )
    return chosen[0].resolve()


def discover_from_dpe_root(dpe_root: Path) -> list[Path]:
    dpe_root = dpe_root.resolve()
    if not dpe_root.is_dir():
        raise SystemExit(f"--dpe-root is not a directory: {dpe_root}")
    subdirs = sorted([p for p in dpe_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise SystemExit(f"No subdirectories under {dpe_root}")
    return [_pick_converted_mp4(sub) for sub in subdirs]


def _max_existing_run_index(outputs_root: Path) -> int:
    if not outputs_root.is_dir():
        return 0
    best = 0
    for d in outputs_root.iterdir():
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        parts = d.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            best = max(best, int(parts[1]))
    return best


def allocate_jobs(videos: list[Path], outputs_root: Path) -> list[VideoJob]:
    outputs_root.mkdir(parents=True, exist_ok=True)
    next_n = _max_existing_run_index(outputs_root) + 1
    jobs: list[VideoJob] = []
    for i, raw in enumerate(videos):
        raw = raw.resolve()
        m = re.search(r"(\d+)\s+DPE[/\\](\d+)", str(raw))
        if m:
            tail = f"{m.group(1)}DPE_n{m.group(2).zfill(3)}"
            out_name = f"run_{next_n + i}_{tail}"
        else:
            out_name = f"run_{next_n + i}"
        jobs.append(
            VideoJob(
                raw_video=raw,
                output_dir=(outputs_root / out_name).resolve(),
                short_name=short_name_from_path(raw),
                video_key=raw.stem,
            )
        )
    return jobs


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch fly tracking pipeline (notebook 01 parity: per-video GUIs, then per-video automated tail).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dpe-root",
        type=str,
        default=None,
        help="Folder like '.../24 DPE' with subfolders 001, 002, … (one *-converted.mp4 each)",
    )
    p.add_argument(
        "--video",
        action="append",
        dest="videos",
        metavar="PATH",
        help="Explicit video path (repeat for multiple). Mutually exclusive with --dpe-root.",
    )
    p.add_argument("--outputs-root", type=str, default="outputs", help="Relative to repo root")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--creds", type=str, default=str(REPO_ROOT / "creds_config.yaml"))
    p.add_argument("--api-key", type=str, default=None, help="Override API key from creds file")
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Open Plotly figures from run_diagnostics (default: save only)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Limit tracking frames (debug)")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if bool(args.dpe_root) == bool(args.videos):
        p.error("Specify exactly one of: --dpe-root OR one or more --video")

    if args.dpe_root:
        raw_list = discover_from_dpe_root(Path(args.dpe_root))
    else:
        raw_list = [Path(v).expanduser().resolve() for v in args.videos]
        for rp in raw_list:
            if not rp.is_file():
                raise SystemExit(f"Video not found: {rp}")

    os.chdir(REPO_ROOT)

    cfg = load_yaml(Path(args.config))
    creds = load_yaml(Path(args.creds))
    api_key = args.api_key or creds.get("API_KEY")
    if not api_key:
        raise SystemExit("No API_KEY in creds file and no --api-key")

    model_id = creds.get("MODEL_ID") or cfg.get("roboflow", {}).get("model_id")
    if not model_id:
        raise SystemExit(
            "No model id: set roboflow.model_id in config.yaml or MODEL_ID in creds_config.yaml"
        )

    outputs_root = (REPO_ROOT / args.outputs_root).resolve()
    jobs = allocate_jobs(raw_list, outputs_root)

    _t = cfg.get("tracker", {})
    _s = cfg.get("stitching", {})
    _p = cfg.get("preprocessing", {})
    _r = cfg.get("roi", {})
    use_saved_roi = _r.get("use_saved_roi", True)
    show_plots = args.show_plots
    diag_fps = float(_s.get("fps", 30))

    detection_confidence_rfdetr = _t.get("detection_confidence_rfdetr", 0.4)
    confidence = _t.get("confidence", 0.1)
    lost_track_buffer = _t.get("lost_track_buffer", 90)
    min_matching_threshold = _t.get("minimum_matching_threshold", 0.2)
    min_consecutive_frames = _t.get("minimum_consecutive_frames", 3)
    asso_func = _t.get("asso_func", "diou")
    brownian_pos_noise = _t.get("brownian_pos_noise", 1.0)
    vial_count_cap = _s.get("vial_count_cap", 7)
    stop_mode = _s.get("stop_mode", "converge")
    w_under = _s.get("w_under", 10.0)
    w_over = _s.get("w_over", 2.0)
    bg_gain = _p.get("bg_gain", 1.2)
    bg_white_level = _p.get("bg_white_level", 245)
    bg_percentile = _p.get("bg_percentile", 85.0)
    bg_sample_stride = _p.get("bg_sample_stride", 1)

    print(f"Using model_id: {model_id}")

    import cv2
    import pandas as pd
    from utils import save_run_params

    from src.preprocessing import preprocess_bgsub_gui
    from src.tracking import export_tracks_xy_tuple_csv_one_config
    from src.stitching import wide_to_long, build_tracklets, stitch
    from src.roi import draw_and_save_vial_rois, assign_vials_and_compact_ids
    from src.visualization import (
        render_vial_overlay_video,
        render_raw_overlay_video,
        render_detections_video,
    )
    from src.metrics import (
        run_diagnostics,
        compute_stitching_objectives,
        print_stitching_objectives,
    )
    from src.ui_context import parse_video_context

    library = _load_roi_library()

    print("=== Planned runs ===")
    for j in jobs:
        print(f"  {j.raw_video}  ->  {j.output_dir}")
    print()

    # ── Stage 0: init each run folder + config snapshot ─────────────────────
    for j in jobs:
        j.output_dir.mkdir(parents=True, exist_ok=True)
        dest_video = j.output_dir / j.raw_video.name
        _link_or_copy(j.raw_video, dest_video)
        cap = cv2.VideoCapture(str(j.raw_video))
        save_run_params(str(j.output_dir), "config", {
            "video": str(j.raw_video),
            "output_dir": str(j.output_dir),
            "short_name": j.short_name,
            "video_fps": cap.get(cv2.CAP_PROP_FPS),
            "video_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "video_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "tracker": {
                "detection_confidence_rfdetr": detection_confidence_rfdetr,
                "confidence": confidence,
                "lost_track_buffer": lost_track_buffer,
                "min_matching_threshold": min_matching_threshold,
                "min_consecutive_frames": min_consecutive_frames,
                "asso_func": asso_func,
                "brownian_pos_noise": brownian_pos_noise,
            },
            "stitching": {
                "stop_mode": stop_mode,
                "w_under": w_under,
                "w_over": w_over,
                "vial_count_cap": vial_count_cap,
            },
            "preprocessing": {
                "bg_gain": bg_gain,
                "bg_white_level": bg_white_level,
                "bg_percentile": bg_percentile,
                "bg_sample_stride": bg_sample_stride,
            },
        })
        cap.release()

    path_to_vid: dict[str, Path] = {}
    raw_cropped_by_job: dict[str, Path | None] = {}

    # ── Phase A: per video — preprocessing (always; GUI vs library like run_tracking) + vial ROIs ─
    for j in jobs:
        print(f"\n=== [{j.short_name}] preprocessing & vial ROIs ===")
        vc = parse_video_context(str(j.raw_video))
        pp_out = j.output_dir / (j.raw_video.stem + "_pp.mp4")
        raw_cropped_out = j.output_dir / (j.raw_video.stem + "_raw_cropped.mp4")
        _stored_crop = library.get(j.video_key, {}).get("preprocessing") if use_saved_roi else None
        if use_saved_roi and _stored_crop is not None:
            print(f"  Using stored preprocessing params for {j.video_key}")
        else:
            if not use_saved_roi:
                print(f"  roi.use_saved_roi=false — opening preprocessing GUI for {j.video_key}")
            else:
                print(f"  Opening preprocessing GUI for {j.video_key}")

        pp_path, crop_params = preprocess_bgsub_gui(
            video_path=str(j.raw_video),
            out_mp4=str(pp_out),
            out_raw_mp4=str(raw_cropped_out),
            gain=bg_gain,
            white_level=bg_white_level,
            bg_sample_stride=bg_sample_stride,
            bg_percentile=bg_percentile,
            crop_params=_stored_crop if (use_saved_roi and _stored_crop is not None) else None,
            video_context=vc,
        )
        path_to_vid[j.video_key] = Path(pp_path)
        raw_cropped_by_job[j.video_key] = Path(raw_cropped_out)

        if j.video_key not in library:
            library[j.video_key] = {}
        library[j.video_key]["preprocessing"] = crop_params
        library[j.video_key]["video_path"] = str(j.raw_video)
        _save_roi_library(library)

        with open(j.output_dir / "crop_roi.json", "w") as f:
            json.dump(crop_params, f, indent=2)

        save_run_params(
            str(j.output_dir),
            "preprocessing",
            {
                "video_pp": str(path_to_vid[j.video_key]),
                "video_raw_cropped": str(raw_cropped_by_job[j.video_key]),
                "crop_params": crop_params,
            },
        )

        roi_json = j.output_dir / "vial_rois.json"
        _stored_vials = library.get(j.video_key, {}).get("vial_rois")
        if use_saved_roi and _stored_vials is not None:
            print(f"  Loaded vial ROIs from library for {j.video_key}")
            vials = {k: tuple(v) for k, v in _stored_vials.items()}
            with open(roi_json, "w") as f:
                json.dump({k: list(v) for k, v in vials.items()}, f, indent=2)
        else:
            print(f"  Opening vial ROI GUI for {j.video_key}")
            vials = draw_and_save_vial_rois(
                video_path=str(j.raw_video),
                roi_json_path=str(roi_json),
                video_context=vc,
            )
            if j.video_key not in library:
                library[j.video_key] = {}
            library[j.video_key]["vial_rois"] = {k: list(v) for k, v in vials.items()}
            _save_roi_library(library)
        save_run_params(str(j.output_dir), "roi", {k: list(v) for k, v in vials.items()})

    v_cfg = cfg.get("visualization", {})
    overlay_kwargs = dict(
        fps_out=v_cfg.get("fps_out", 30),
        show_ids=v_cfg.get("show_ids", True),
        tick_len=v_cfg.get("tick_len", 4),
        tick_thick=v_cfg.get("tick_thick", 1),
        chip_font_scale=v_cfg.get("chip_font_scale", 0.32),
        label_offset_x=v_cfg.get("label_offset_x", 10),
        label_offset_y=v_cfg.get("label_offset_y", -10),
        chip_pad=v_cfg.get("chip_pad", 1),
        leader_thick=v_cfg.get("leader_thick", 1),
    )

    # ── Phase B: per video — track → stitch → compact/metrics → overlays ────
    for j in jobs:
        print(f"\n=== [{j.short_name}] tracking → stitching → metrics → overlays ===")
        ptv = path_to_vid[j.video_key]
        wide_csv = j.output_dir / "tracks_wide_format.csv"
        det_log_csv = j.output_dir / "detections_raw.csv"
        print("  RF-DETR + OC-SORT …")
        df_wide, tracker = export_tracks_xy_tuple_csv_one_config(
            video_path=str(ptv),
            output_csv=str(wide_csv),
            api_key=api_key,
            model_id=model_id,
            detection_confidence_rfdetr=detection_confidence_rfdetr,
            confidence=confidence,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=min_matching_threshold,
            minimum_consecutive_frames=min_consecutive_frames,
            asso_func=asso_func,
            brownian_pos_noise=brownian_pos_noise,
            det_log_csv=str(det_log_csv),
            max_frames=args.max_frames,
        )

        save_run_params(str(j.output_dir), "tracker_output", {
            "wide_csv": str(wide_csv),
            "frames": int(df_wide.shape[0]),
            "track_count": int(df_wide.shape[1] - 1),
        })
        with open(j.output_dir / "tracker_log.json", "w") as f:
            json.dump({
                "detection_log": tracker.detection_log,
                "suppressed_tracks": tracker.suppressed_tracks,
                "min_hits": tracker.min_hits,
                "max_age": tracker.max_age,
            }, f)

        render_detections_video(
            video_path=str(ptv),
            det_log_csv=str(det_log_csv),
            out_mp4=str(j.output_dir / f"{j.short_name}_detections_RF-DETR.mp4"),
            fps_out=v_cfg.get("fps_out", 30),
        )

        run_diagnostics(
            tracker=tracker,
            df_wide=df_wide,
            df_stitched=None,
            n_expected=42,
            fps=diag_fps,
            config=cfg,
            output_dir=None,
            show_plots=show_plots,
        )

        roi_json = j.output_dir / "vial_rois.json"
        stitched_csv = j.output_dir / "tracks_xy_stitched_long.csv"
        long_csv = j.output_dir / "tracks_long_format.csv"

        with open(roi_json) as f:
            vial_rois = {k: tuple(v) for k, v in json.load(f).items()}

        print("  Hungarian stitching …")
        long_df = wide_to_long(pd.read_csv(wide_csv), out_csv=str(long_csv))
        tracklets = build_tracklets(long_df)
        stitched_df = stitch(
            long_df=long_df,
            vial_rois=vial_rois,
            tracklets=tracklets,
            output_dir=str(j.output_dir),
        )
        stitched_df.to_csv(stitched_csv, index=False)
        save_run_params(str(j.output_dir), "stitching_output", {
            "stitched_csv": str(stitched_csv),
            "stitched_ids": int(stitched_df["stitched_id"].nunique()),
            "original_ids": int(stitched_df["orig_id"].nunique()),
        })
        print(f"  stitched -> {stitched_csv}")

        compact_csv = j.output_dir / "compact_tracks.csv"
        print("  compact IDs …")
        df_compact = assign_vials_and_compact_ids(
            stitched_csv=str(stitched_csv),
            roi_json=str(roi_json),
            out_csv=str(compact_csv),
            fps=diag_fps,
        )
        save_run_params(str(j.output_dir), "compact", {
            "csv": str(compact_csv),
            "rows": int(df_compact.shape[0]),
        })

        num_frames = int(df_wide["frame"].max()) + 1
        objectives = compute_stitching_objectives(
            df_stitched=stitched_df,
            vial_rois=vial_rois,
            num_frames=num_frames,
            expected_per_vial=_s.get("expected_per_vial", 7),
            short_frac=_s.get("short_track_frac", 0.10),
        )
        print_stitching_objectives(objectives)
        save_run_params(str(j.output_dir), "stitching_objectives",
                        {k: float(v) for k, v in objectives.items()})

        with open(j.output_dir / "tracker_log.json") as f:
            _tl = json.load(f)
        mock_tracker = types.SimpleNamespace(**_tl)

        n_expected = _s.get("expected_per_vial", 7) * len(vial_rois)
        print("  full diagnostics (tracker from tracker_log.json) …")
        run_diagnostics(
            tracker=mock_tracker,
            df_wide=df_wide,
            df_stitched=stitched_df,
            df_compact=df_compact,
            n_expected=n_expected,
            fps=diag_fps,
            vial_rois=vial_rois,
            config=cfg,
            output_dir=str(j.output_dir),
            stitching_objectives=objectives,
            show_plots=show_plots,
        )

        _overlay_mode = v_cfg.get("overlay_source", "raw_cropped").lower()
        rc = raw_cropped_by_job.get(j.video_key)
        if _overlay_mode == "raw_cropped" and rc is not None and rc.exists():
            overlay_video = str(rc)
        elif _overlay_mode == "raw_cropped":
            overlay_video = str(j.raw_video)
        else:
            overlay_video = str(path_to_vid[j.video_key])
        print(f"  overlays (substrate={_overlay_mode}) …")

        raw_overlay_mp4 = j.output_dir / f"{j.short_name}_overlay_raw_ocsort.mp4"
        overlay_mp4 = j.output_dir / f"{j.short_name}_overlay_vials_stitched.mp4"

        render_raw_overlay_video(
            video_path=overlay_video,
            csv_path=str(long_csv),
            out_mp4=str(raw_overlay_mp4),
            vial_rois=vial_rois,
            det_log_csv=str(det_log_csv) if det_log_csv.exists() else None,
            **overlay_kwargs,
        )
        render_vial_overlay_video(
            video_path=overlay_video,
            csv_path=str(compact_csv),
            out_mp4=str(overlay_mp4),
            vial_rois=vial_rois,
            det_log_csv=str(det_log_csv) if det_log_csv.exists() else None,
            **overlay_kwargs,
        )
        save_run_params(str(j.output_dir), "outputs", {
            "raw_overlay": str(raw_overlay_mp4),
            "overlay": str(overlay_mp4),
        })

    print("\nAll jobs finished.")


if __name__ == "__main__":
    main()
