"""
src/tracking.py

Roboflow RF-DETR detection + OC-SORT multi-object tracking.
Writes a wide CSV: rows = frames, columns = track IDs, cells = "(x, y)".

If a detection cache CSV (det_log_csv) already exists on disk, RF-DETR is
skipped entirely and detections are loaded from the cache — useful for
re-running the tracker with different association parameters without paying
the API cost again.
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
    behavioral_weight: float = 0.0,
    jump_factor: float = 2.0,
    jump_iou_threshold: float = 0.05,
    jump_inertia: float = 0.05,
    expected_count: int | None = None,
    w_under: float = 15.0,
    w_over: float = 2.0,
    overlap_iou_scale: float = 0.1,
    edge_fraction: float = 0.1,
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
      df      — the wide-format DataFrame that was saved to CSV
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

    # ── Detection source ─────────────────────────────────────────────────────
    use_cache = det_log_csv is not None and os.path.exists(det_log_csv)

    if use_cache:
        print(f"Detection cache found — skipping RF-DETR: {det_log_csv}")
        det_cache = pd.read_csv(det_log_csv)
        # group by frame for fast per-frame lookup
        det_by_frame = {
            frame_idx: grp[["x1", "y1", "x2", "y2", "conf"]].values
            for frame_idx, grp in det_cache.groupby("frame")
        }
        n_frames = int(det_cache["frame"].max()) + 1
    else:
        from inference_sdk import InferenceHTTPClient
        from inference_sdk.http.entities import InferenceConfiguration
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        client.configure(InferenceConfiguration(confidence_threshold=detection_confidence_rfdetr))
        det_by_frame = None
        n_frames = None

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = OCSort(
        det_thresh=confidence,
        max_age=lost_track_buffer,
        min_hits=minimum_consecutive_frames,
        iou_threshold=minimum_matching_threshold,
        delta_t=3,
        asso_func=asso_func,
        inertia=0.2,
        use_byte=False,
        brownian_pos_noise=brownian_pos_noise,
        vial_rois=vial_rois,
        aspect_weight=aspect_weight,
        behavioral_weight=behavioral_weight,
        jump_factor=jump_factor,
        jump_iou_threshold=jump_iou_threshold,
        jump_inertia=jump_inertia,
        expected_count=expected_count,
        w_under=w_under,
        w_over=w_over,
        overlap_iou_scale=overlap_iou_scale,
        edge_fraction=edge_fraction,
    )

    rows = []
    all_track_ids = set()
    det_log_rows = []
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        if use_cache:
            if frame_idx >= n_frames:
                break
            det_array = det_by_frame.get(frame_idx, np.empty((0, 5)))
        else:
            ok, frame = cap.read()
            if not ok:
                break
            result = client.infer(frame, model_id=model_id)
            dets = sv.Detections.from_inference(result)
            if len(dets) > 0:
                det_array = np.hstack([dets.xyxy, dets.confidence[:, None]])
                for bbox, conf in zip(dets.xyxy.tolist(), dets.confidence.tolist()):
                    det_log_rows.append({
                        "frame": frame_idx,
                        "x1": bbox[0], "y1": bbox[1],
                        "x2": bbox[2], "y2": bbox[3],
                        "conf": conf,
                    })
            else:
                det_array = np.empty((0, 5))

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
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")

    if not use_cache and det_log_csv is not None:
        pd.DataFrame(det_log_rows).to_csv(det_log_csv, index=False)
        print(f"Saved detection cache: {det_log_csv}  ({len(det_log_rows)} detections)")

    return df, tracker
