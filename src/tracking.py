"""
src/tracking.py

Roboflow RF-DETR detection + OC-SORT multi-object tracking.
Writes a wide CSV: rows = frames, columns = track IDs, cells = "(x, y)".

If a detection cache CSV (det_log_csv) already exists on disk, RF-DETR is
skipped entirely and detections are loaded from the cache — useful for
re-running the tracker with different association parameters without paying
the API cost again.

Optional watershed splitting (watershed_cfg) runs after detection collection,
before the tracker sees anything: oversized bboxes that likely contain >1
fly are split via marker-controlled watershed. Post-split detections are
what gets written to the cache.
"""

import logging
import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_xy(cell):
    """Parse a '(x, y)' CSV cell. Returns (float, float) or None."""
    if not isinstance(cell, str) or not cell.strip():
        return None
    try:
        s = cell.strip().lstrip("(").rstrip(")")
        x, y = s.split(",")
        return float(x), float(y)
    except Exception:
        return None


def _run_tracker_pass(det_by_frame, n_frames, img_h, img_w, vial_rois,
                      tracker_kwargs, max_frames=None):
    """
    Run one complete OC-SORT pass over det_by_frame.
    Returns (df_wide, tracker).
    """
    from .ocsort import OCSort
    tracker = OCSort(**tracker_kwargs)
    rows = []
    all_track_ids = set()

    for frame_idx in range(n_frames):
        if max_frames is not None and frame_idx >= max_frames:
            break

        det_array = det_by_frame.get(frame_idx, np.empty((0, 5), dtype=np.float32))

        if vial_rois is not None and len(det_array) > 0:
            cx = (det_array[:, 0] + det_array[:, 2]) / 2.0
            cy = (det_array[:, 1] + det_array[:, 3]) / 2.0
            in_vial = np.zeros(len(det_array), dtype=bool)
            for x0, y0, x1, y1 in vial_rois.values():
                in_vial |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
            det_array = det_array[in_vial]

        frame_row = {"frame": frame_idx}
        if len(det_array) > 0:
            tracks = tracker.update(det_array, [img_h, img_w], [img_h, img_w])
            if tracks is not None and len(tracks) > 0:
                if tracks.ndim == 1:
                    tracks = tracks[None, :]
                xyxy = tracks[:, :4]
                tids = tracks[:, 4].astype(int)
                cx_out = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                cy_out = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                for tid, x, y in zip(tids, cx_out, cy_out):
                    all_track_ids.add(int(tid))
                    frame_row[f"id{int(tid)}"] = f"({x:.2f}, {y:.2f})"
        else:
            tracker.update(np.empty((0, 5)), [img_h, img_w], [img_h, img_w])

        rows.append(frame_row)

    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)
    return df, tracker


def _find_occlusion_gaps(df_wide, vial_rois, vial_expected_counts, det_by_frame,
                         top_exit_px=2.0, occlusion_max_gap=90):
    """
    Analyse Pass-1 output to classify count drops per vial.

    Returns (top_exit_events, top_reentry_events, occlusion_gaps).

    top_exit_events    — tracks whose centroid reached the vial ROI top edge.
    top_reentry_events — tracks that first appeared near the vial ROI top edge.
    occlusion_gaps     — per-track CSV gaps that are short enough, not a top
                         exit, and where the vial count was full just before
                         the gap then dropped.
    """
    id_cols = [c for c in df_wide.columns if c.startswith("id")]
    frames_list = df_wide["frame"].tolist()

    def _vial_of(cx, cy):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return vid
        return None

    # per-track position sequence
    track_positions = {}
    for col in id_cols:
        tid = int(col[2:])
        for i, frame in enumerate(frames_list):
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is not None:
                track_positions.setdefault(tid, []).append((int(frame), pos[0], pos[1]))

    # top exit / reentry
    top_exit_events = []
    top_exit_tids = set()
    top_reentry_events = []

    for tid, positions in track_positions.items():
        for frame, cx, cy in positions:
            vid = _vial_of(cx, cy)
            if vid is None:
                continue
            _, y0, _, _ = vial_rois[vid]
            if cy <= y0 + top_exit_px:
                top_exit_events.append({"track_id": tid, "frame": int(frame), "vial": vid})
                top_exit_tids.add(tid)
                break

        if positions:
            f0, cx0, cy0 = positions[0]
            vid = _vial_of(cx0, cy0)
            if vid is not None:
                _, y0, _, _ = vial_rois[vid]
                if cy0 <= y0 + top_exit_px:
                    top_reentry_events.append(
                        {"track_id": tid, "frame": int(f0), "vial": vid}
                    )

    # median bbox per vial from detection cache
    vial_median_bbox = {}
    for vid, (x0, y0, x1, y1) in vial_rois.items():
        widths, heights = [], []
        for dets in det_by_frame.values():
            if len(dets) == 0:
                continue
            cx_arr = (dets[:, 0] + dets[:, 2]) / 2.0
            cy_arr = (dets[:, 1] + dets[:, 3]) / 2.0
            mask = (cx_arr >= x0) & (cx_arr <= x1) & (cy_arr >= y0) & (cy_arr <= y1)
            if mask.any():
                widths.extend((dets[mask, 2] - dets[mask, 0]).tolist())
                heights.extend((dets[mask, 3] - dets[mask, 1]).tolist())
        vial_median_bbox[vid] = (
            float(np.median(widths)) if widths else 20.0,
            float(np.median(heights)) if heights else 20.0,
        )

    # per-vial per-frame active track set for count verification
    vial_frame_active = {}
    for col in id_cols:
        tid = int(col[2:])
        for i, frame in enumerate(frames_list):
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is None:
                continue
            vid = _vial_of(pos[0], pos[1])
            if vid is None:
                continue
            vial_frame_active.setdefault(vid, {}).setdefault(int(frame), set()).add(tid)

    # find occlusion gaps
    occlusion_gaps = []

    for tid, positions in track_positions.items():
        if tid in top_exit_tids or len(positions) < 2:
            continue

        for i in range(len(positions) - 1):
            f_before, cx_before, cy_before = positions[i]
            f_after  = positions[i + 1][0]
            gap_len  = f_after - f_before - 1

            if gap_len <= 0 or gap_len > occlusion_max_gap:
                continue

            vid = _vial_of(cx_before, cy_before)
            if vid is None:
                continue

            _, y0, _, _ = vial_rois[vid]
            if cy_before <= y0 + top_exit_px:
                continue  # track was near top before gap — treat as exit

            expected_n = vial_expected_counts.get(vid, 0)
            if expected_n <= 0:
                continue

            # Vial must have been full just before the gap
            count_before = len(vial_frame_active.get(vid, {}).get(f_before, set()))
            if count_before < expected_n:
                continue

            # Count must have dropped during the gap
            count_during = len(
                vial_frame_active.get(vid, {}).get(f_before + 1, set())
            )
            if count_during >= expected_n:
                continue

            bbox_w, bbox_h = vial_median_bbox.get(vid, (20.0, 20.0))
            occlusion_gaps.append({
                "vial":             vid,
                "missing_track_id": tid,
                "start_frame":      f_before,
                "end_frame":        f_after,
                "last_known_cx":    cx_before,
                "last_known_cy":    cy_before,
                "bbox_w":           bbox_w,
                "bbox_h":           bbox_h,
            })

    return top_exit_events, top_reentry_events, occlusion_gaps


def _inject_ghost_detections(det_by_frame, occlusion_gaps, df_wide, vial_rois,
                             offset_fraction=0.5, ghost_confidence=0.45):
    """
    For each occlusion gap inject one synthetic detection per gap frame.

    Ghost position = occluder centroid + (offset_fraction × bbox_w) in the
    direction from occluder toward the missing fly's last known position.
    Falls back to last known position when no other track is visible.

    Returns (det_by_frame_with_ghosts, ghost_log).
    """
    det_out = dict(det_by_frame)  # shallow copy; arrays copied on write

    id_cols = [c for c in df_wide.columns if c.startswith("id")]
    frames_list = df_wide["frame"].tolist()

    def _vial_of(cx, cy):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return vid
        return None

    # frame → {tid: (cx, cy)} from Pass-1
    frame_track_pos = {}
    for i, frame in enumerate(frames_list):
        frame = int(frame)
        frame_track_pos[frame] = {}
        for col in id_cols:
            tid = int(col[2:])
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is not None:
                frame_track_pos[frame][tid] = pos

    ghost_log = []

    for gap in occlusion_gaps:
        vid         = gap["vial"]
        missing_tid = gap["missing_track_id"]
        start_frame = gap["start_frame"]
        end_frame   = gap["end_frame"]
        last_cx     = gap["last_known_cx"]
        last_cy     = gap["last_known_cy"]
        bbox_w      = gap["bbox_w"]
        bbox_h      = gap["bbox_h"]
        x0, y0, x1, y1 = vial_rois[vid]

        for frame in range(start_frame + 1, end_frame):
            candidates = {
                tid: pos
                for tid, pos in frame_track_pos.get(frame, {}).items()
                if tid != missing_tid and _vial_of(pos[0], pos[1]) == vid
            }

            if candidates:
                occluder_tid = min(
                    candidates,
                    key=lambda t: (candidates[t][0] - last_cx) ** 2
                                + (candidates[t][1] - last_cy) ** 2,
                )
                occ_cx, occ_cy = candidates[occluder_tid]
                dx   = last_cx - occ_cx
                dy   = last_cy - occ_cy
                dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
                ghost_cx = occ_cx + (dx / dist) * bbox_w * offset_fraction
                ghost_cy = occ_cy + (dy / dist) * bbox_w * offset_fraction
            else:
                occluder_tid = None
                ghost_cx, ghost_cy = last_cx, last_cy

            ghost_cx = float(np.clip(ghost_cx, x0 + bbox_w / 2, x1 - bbox_w / 2))
            ghost_cy = float(np.clip(ghost_cy, y0 + bbox_h / 2, y1 - bbox_h / 2))

            ghost_det = np.array([[
                ghost_cx - bbox_w / 2, ghost_cy - bbox_h / 2,
                ghost_cx + bbox_w / 2, ghost_cy + bbox_h / 2,
                ghost_confidence,
            ]], dtype=np.float32)

            if frame in det_out:
                det_out[frame] = np.vstack([det_out[frame], ghost_det])
            else:
                det_out[frame] = ghost_det

            ghost_log.append({
                "frame":             frame,
                "missing_track_id":  missing_tid,
                "occluder_track_id": occluder_tid,
                "ghost_cx":          ghost_cx,
                "ghost_cy":          ghost_cy,
                "vial":              vid,
            })

    return det_out, ghost_log


def export_tracks_xy_tuple_csv_one_config(
    video_path: str,
    output_csv: str,
    api_key: str,
    model_id: str,
    inference_api_url: str = "https://detect.roboflow.com",
    detection_confidence_rfdetr: float = 0.4,
    confidence: float = 0.10,
    lost_track_buffer: int = 90,
    minimum_matching_threshold: float = 0.01,
    minimum_consecutive_frames: int = 1,
    max_frames: int | None = None,
    fps_assumed: float | None = None,
    min_area: float | None = 40,
    asso_func: str = "hmiou",
    brownian_pos_noise: float = 1.0,
    det_log_csv: str | None = None,
    vial_rois: dict | None = None,
    aspect_weight: float = 0.0,
    behavioral_weights: dict | None = None,
    overlap_weight_scale: float = 6.0,
    jump_factor: float = 2.0,
    jump_iou_threshold: float = 0.05,
    jump_inertia: float = 0.05,
    inertia: float = 0.2,
    delta_t: int = 3,
    expected_count: int | None = None,
    w_under: float = 15.0,
    w_over: float = 2.0,
    overlap_iou_scale: float = 0.1,
    edge_fraction: float = 0.1,
    watershed_cfg: dict | None = None,
    vial_expected_counts: dict | None = None,
    ghost_detection_enabled: bool = False,
    ghost_offset_fraction: float = 0.5,
    ghost_confidence: float = 0.45,
    ghost_occlusion_max_gap: int = 90,
    ghost_top_exit_px: float = 2.0,
) -> tuple[pd.DataFrame, object]:
    """
    Run RF-DETR + OC-SORT for one configuration and write a wide CSV where:
      - rows    = frame index
      - columns = track IDs  (id{N})
      - cells   = "(x, y)" centre of bbox, or empty if not present

    If det_log_csv already exists on disk, RF-DETR inference is skipped and
    detections are loaded from the cache. This lets you re-run the tracker
    with different association parameters at zero API cost.

    Returns (df, tracker):
      df      — the wide-format DataFrame that was saved to output_csv
      tracker — the OCSort object (carries detection_log, suppressed_tracks,
                ghost_log, top_exit_events, top_reentry_events)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if fps_assumed is None:
        fps_assumed = float(fps) if fps and fps > 0 else 30.0

    # ── Phase 1: Detection collection (cache or RF-DETR) ─────────────────────
    use_cache = det_log_csv is not None and os.path.exists(det_log_csv)
    det_by_frame: dict[int, np.ndarray] = {}

    if use_cache:
        print(f"Detection cache found — skipping RF-DETR: {det_log_csv}")
        det_cache = pd.read_csv(det_log_csv)
        det_by_frame = {
            int(frame_idx): grp[["x1", "y1", "x2", "y2", "conf"]].values.astype(np.float32)
            for frame_idx, grp in det_cache.groupby("frame")
        }
        n_frames = int(det_cache["frame"].max()) + 1 if len(det_cache) else 0
        cap.release()
    else:
        from inference_sdk import InferenceHTTPClient
        from inference_sdk.http.entities import InferenceConfiguration
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        client.configure(InferenceConfiguration(confidence_threshold=detection_confidence_rfdetr))

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            result = client.infer(frame, model_id=model_id)
            dets = sv.Detections.from_inference(result)
            if len(dets) > 0:
                det_by_frame[frame_idx] = np.hstack(
                    [dets.xyxy, dets.confidence[:, None]]
                ).astype(np.float32)
            frame_idx += 1
        n_frames = frame_idx
        cap.release()

    # ── Phase 2: Watershed split (optional) ─────────────────────────────────
    if watershed_cfg and watershed_cfg.get("enabled", False):
        from .watershed_split import apply_watershed_splits
        debug_dir = None
        if watershed_cfg.get("debug", False) and det_log_csv is not None:
            debug_dir = os.path.join(os.path.dirname(det_log_csv) or ".", "watershed_debug")
        det_by_frame = apply_watershed_splits(
            det_by_frame=det_by_frame,
            video_path=video_path,
            cfg=watershed_cfg,
            debug_dir=debug_dir,
        )

    # Write det_log_csv when we produced fresh detections (cache miss) or
    # when watershed mutated them. Skip when cache hit and watershed off,
    # so a re-run with the same dets doesn't churn the file.
    watershed_on = bool(watershed_cfg and watershed_cfg.get("enabled", False))
    if det_log_csv is not None and (not use_cache or watershed_on):
        rows = []
        for f in sorted(det_by_frame.keys()):
            for x1, y1, x2, y2, conf in det_by_frame[f]:
                rows.append({"frame": int(f),
                             "x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2),
                             "conf": float(conf)})
        pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf"]).to_csv(
            det_log_csv, index=False
        )
        print(f"Saved detection cache: {det_log_csv}  ({len(rows)} detections)")

    # ── Phase 3: Tracker ─────────────────────────────────────────────────────
    tracker_kwargs = dict(
        det_thresh=confidence,
        max_age=lost_track_buffer,
        min_hits=minimum_consecutive_frames,
        iou_threshold=minimum_matching_threshold,
        delta_t=delta_t,
        asso_func=asso_func,
        inertia=inertia,
        use_byte=False,
        brownian_pos_noise=brownian_pos_noise,
        vial_rois=vial_rois,
        aspect_weight=aspect_weight,
        behavioral_weights=behavioral_weights,
        overlap_weight_scale=overlap_weight_scale,
        jump_factor=jump_factor,
        jump_iou_threshold=jump_iou_threshold,
        jump_inertia=jump_inertia,
        expected_count=expected_count,
        w_under=w_under,
        w_over=w_over,
        overlap_iou_scale=overlap_iou_scale,
        edge_fraction=edge_fraction,
        fps=fps_assumed,
    )

    do_ghost = (
        ghost_detection_enabled
        and vial_rois is not None
        and vial_expected_counts is not None
        and any(v > 0 for v in vial_expected_counts.values())
    )

    if do_ghost:
        print("Ghost detection enabled — running Pass 1 (baseline)...")
        df_pass1, _ = _run_tracker_pass(
            det_by_frame, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )
        n_pass1_tracks = len([c for c in df_pass1.columns if c.startswith("id")])
        print(f"  Pass 1: {n_pass1_tracks} tracks")

        top_exit_events, top_reentry_events, occlusion_gaps = _find_occlusion_gaps(
            df_pass1, vial_rois, vial_expected_counts, det_by_frame,
            top_exit_px=ghost_top_exit_px,
            occlusion_max_gap=ghost_occlusion_max_gap,
        )
        print(f"  Gap analysis: {len(occlusion_gaps)} occlusion gaps, "
              f"{len(top_exit_events)} top exits, {len(top_reentry_events)} top reentries")

        det_with_ghosts, ghost_log = _inject_ghost_detections(
            det_by_frame, occlusion_gaps, df_pass1, vial_rois,
            offset_fraction=ghost_offset_fraction,
            ghost_confidence=ghost_confidence,
        )
        print(f"  Injected {len(ghost_log)} ghost detections — running Pass 2...")

        df, tracker = _run_tracker_pass(
            det_with_ghosts, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )
        n_pass2_tracks = len([c for c in df.columns if c.startswith("id")])
        print(f"  Pass 2: {n_pass2_tracks} tracks")
    else:
        ghost_log = []
        top_exit_events = []
        top_reentry_events = []
        df, tracker = _run_tracker_pass(
            det_by_frame, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )

    # Attach ghost metadata to tracker so callers can save it
    tracker.ghost_log          = ghost_log
    tracker.top_exit_events    = top_exit_events
    tracker.top_reentry_events = top_reentry_events

    id_cols = [c for c in df.columns if c.startswith("id")]
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")

    obs_log_path = output_csv.replace(".csv", "_obs_logs.json")
    _save_observation_logs(tracker, obs_log_path)

    return df, tracker


def _save_observation_logs(tracker, path: str) -> None:
    """Write each tracker's observation_log to a JSON file.

    Each entry: {"id": int, "log": [[frame_idx, [x1,y1,x2,y2], score], ...]}
    Only includes KalmanBoxTracker objects (active at end of video).
    Suppressed tracks are dicts without observation_log so they are skipped.
    """
    import json
    records = []
    for trk in tracker.trackers:
        log = [
            [int(f), [float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(s) if s is not None else None]
            for f, b, s in trk.observation_log
        ]
        if log:
            records.append({"id": int(trk.id) + 1, "log": log})  # 1-based to match CSV
    with open(path, "w") as fh:
        json.dump(records, fh)
    print(f"Saved observation logs: {path}  ({len(records)} trackers)")
