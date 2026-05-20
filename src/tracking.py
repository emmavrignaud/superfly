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
      tracker — the OCSort object (carries detection_log and suppressed_tracks)
    """
    from .ocsort import OCSort

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
    tracker = OCSort(
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

    rows = []
    all_track_ids = set()

    for frame_idx in range(n_frames):
        if max_frames is not None and frame_idx >= max_frames:
            break
        det_array = det_by_frame.get(frame_idx, np.empty((0, 5), dtype=np.float32))

        # Filter detections to vials only — discard anything whose centre
        # falls outside every vial ROI so spurious out-of-vial detections
        # never spawn new tracks.
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
                cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                for tid, x, y in zip(tids, cx, cy):
                    all_track_ids.add(int(tid))
                    frame_row[f"id{int(tid)}"] = f"({x:.2f}, {y:.2f})"
        else:
            tracker.update(np.empty((0, 5)), [img_h, img_w], [img_h, img_w])

        rows.append(frame_row)

    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")

    # Save observation logs — needed for post-hoc relink grid search without
    # re-running the tracker.  Format: list of {id, log: [[frame, [x1,y1,x2,y2], score], ...]}
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
