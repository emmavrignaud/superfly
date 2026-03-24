"""
src/tracking.py

Roboflow RF-DETR detection + OC-SORT multi-object tracking.
Writes a wide CSV: rows = frames, columns = track IDs, cells = "(x, y)".
"""

import logging
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from inference import get_model
from boxmot import OcSort

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def export_tracks_xy_tuple_csv_one_config(
    video_path: str,
    output_csv: str,
    api_key: str,
    model_id: str,
    confidence: float = 0.10,
    track_activation_threshold: float = 0.10,
    lost_track_buffer: int = 90,
    minimum_matching_threshold: float = 0.01,
    minimum_consecutive_frames: int = 10,
    max_frames: int | None = None,
    fps_assumed: float | None = None,
    use_bottom_sensitive: bool = False,
    global_confidence: float = 0.25,
    bottom_confidence: float = 0.10,
    bottom_start_frac: float = 0.75,
    min_area: float | None = 40,
    asso_func: str = "hmiou",
) -> pd.DataFrame:
    """
    Run RF-DETR + OC-SORT for one configuration and write a wide CSV where:
      - rows    = frame index
      - columns = track IDs  (id{N})
      - cells   = "(x, y)" centre of bbox, or empty if not present

    Returns the DataFrame that was saved.
    """
    model = get_model(model_id=model_id, api_key=api_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_assumed is None:
        fps_assumed = float(fps) if fps and fps > 0 else 30.0

    tracker = OcSort(
        det_thresh=confidence,
        max_age=lost_track_buffer,
        min_hits=minimum_consecutive_frames,
        iou_threshold=minimum_matching_threshold,
        delta_t=3,
        asso_func=asso_func,
        inertia=0.2,
        use_byte=False,
    )

    rows = []
    all_track_ids = set()
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        results = model.infer(frame, confidence=confidence)[0]
        dets = sv.Detections.from_inference(results)

        frame_row = {"frame": frame_idx}

        if len(dets) > 0:
            det_array = np.hstack([
                dets.xyxy,
                dets.confidence[:, None],
                dets.class_id[:, None],
            ])

            tracks = tracker.update(det_array, frame)

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

        rows.append(frame_row)
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")
    return df
