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
    behavioral_weight_overlap: float | None = None,
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
    relink_behavioral_weights: dict | None = None,
    relink_min_length: int = 10,
    relink_inconsistency_threshold: float = 0.4,
    relink_swap_threshold: float = 0.2,
    relink_confidence_weight: float = 1.0,
    relinked_csv: str | None = None,
) -> tuple[pd.DataFrame, object, "pd.DataFrame | None"]:
    """
    Run RF-DETR + OC-SORT for one configuration and write a wide CSV where:
      - rows    = frame index
      - columns = track IDs  (id{N})
      - cells   = "(x, y)" centre of bbox, or empty if not present

    If det_log_csv already exists on disk, RF-DETR inference is skipped and
    detections are loaded from the cache. This lets you re-run the tracker
    with different association parameters at zero API cost.

    Returns (df, tracker, df_relinked):
      df           — the wide-format DataFrame that was saved to output_csv
      tracker      — the OCSort object (carries detection_log and suppressed_tracks)
      df_relinked  — long-format relinked DataFrame (frame, x, y, original_id,
                     relinked_id), or None if relinked_csv was not specified
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
        delta_t=delta_t,
        asso_func=asso_func,
        inertia=inertia,
        use_byte=False,
        brownian_pos_noise=brownian_pos_noise,
        vial_rois=vial_rois,
        aspect_weight=aspect_weight,
        behavioral_weight=behavioral_weight,
        behavioral_weight_overlap=behavioral_weight_overlap,
        jump_factor=jump_factor,
        jump_iou_threshold=jump_iou_threshold,
        jump_inertia=jump_inertia,
        expected_count=expected_count,
        w_under=w_under,
        w_over=w_over,
        overlap_iou_scale=overlap_iou_scale,
        edge_fraction=edge_fraction,
        fps=fps_assumed,
        relink_behavioral_weights=relink_behavioral_weights,
        relink_min_length=relink_min_length,
        relink_inconsistency_threshold=relink_inconsistency_threshold,
        relink_swap_threshold=relink_swap_threshold,
        relink_confidence_weight=relink_confidence_weight,
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

    # ── Second-round re-linking ───────────────────────────────────────────────
    swaps = tracker.relink()
    if swaps:
        print(f"Relink: {len(swaps)} swap(s) accepted: {swaps}")
    else:
        print("Relink: no swaps accepted.")

    if relinked_csv is not None:
        # Build long-format relinked CSV applying the swaps to observation_log.
        # For each tracker, walk its observation_log. If a swap involves this
        # tracker, detections from swap_frame onward belong to the partner's ID.
        id_map: dict[int, list] = {}  # 1-based id → list of (frame, x, y, orig_id)
        swap_lookup: dict[int, tuple[int, int]] = {}  # id_a → (id_b, swap_frame)
        for id_a, id_b, swap_frame in swaps:
            swap_lookup[id_a] = (id_b, swap_frame)
            swap_lookup[id_b] = (id_a, swap_frame)

        for trk in tracker.trackers:
            obs = getattr(trk, "observation_log", [])
            orig_id = trk.id + 1  # 1-based
            swap_info = swap_lookup.get(orig_id)
            for frame_i, bbox, *_ in sorted(obs, key=lambda t: t[0]):
                cx = float((bbox[0] + bbox[2]) / 2.0)
                cy = float((bbox[1] + bbox[3]) / 2.0)
                if swap_info is not None:
                    partner_id, swap_frame = swap_info
                    relinked_id = partner_id if frame_i >= swap_frame else orig_id
                else:
                    relinked_id = orig_id
                id_map.setdefault(orig_id, []).append({
                    "frame": frame_i,
                    "x": cx,
                    "y": cy,
                    "original_id": orig_id,
                    "relinked_id": relinked_id,
                })

        relink_rows = [row for rows_list in id_map.values() for row in rows_list]
        df_relinked = pd.DataFrame(relink_rows, columns=["frame", "x", "y", "original_id", "relinked_id"])
        df_relinked = df_relinked.sort_values(["frame", "relinked_id"]).reset_index(drop=True)
        df_relinked.to_csv(relinked_csv, index=False)
        print(f"Saved relinked tracks: {relinked_csv}  ({len(df_relinked)} rows)")
    else:
        df_relinked = None

    return df, tracker, df_relinked
