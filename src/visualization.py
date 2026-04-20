"""
src/visualization.py

Render overlay video: fly positions coloured by vial, shaded by compact_id.

Expects compact_tracks CSV columns: frame, x, y, vial_id, compact_id.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def _visualization_cfg() -> dict:
    p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f).get("visualization", {})


def _roi_library_path() -> Path:
    return Path(__file__).parent.parent / "roi_library.json"


def _resolve_overlay_source(video_path: str) -> tuple[str, dict | None]:
    """Pick the substrate the overlay should be drawn on.

    Returns (effective_video_path, crop_params or None).

    The caller's path is always used as-is — we never substitute a different
    file. The only thing this function decides is whether to crop each frame
    on the way in:

    - ``overlay_source: processed``: return (video_path, None). Caller reads
      frames as-is. Use this when the caller already passed the processed
      (``_pp``) video and wants the background-subtracted substrate.
    - ``overlay_source: raw_cropped``: look up ``crop_params`` in
      roi_library.json by the video's filename stem. If found, return
      (video_path, crop_params) so the caller can seek+crop on the fly.
      If no entry exists (e.g. caller passed a ``_pp`` path whose stem isn't
      in the library), fall back to no crop and use the file as-is.

    Any missing file / missing library entry falls back silently (with a
    warning) — visualization must never crash the pipeline.
    """
    cfg = _visualization_cfg()
    mode = str(cfg.get("overlay_source", "raw_cropped")).lower()
    if mode == "processed":
        return video_path, None
    if mode != "raw_cropped":
        print(f"[visualization] unknown overlay_source {mode!r} — using file as-is.")
        return video_path, None

    p = Path(video_path)
    if not p.exists():
        print(f"[visualization] video not found at {p} — using path as-is (will likely fail at open).")
        return video_path, None

    lib_path = _roi_library_path()
    if not lib_path.exists():
        print("[visualization] roi_library.json not found — using video as-is (no crop).")
        return video_path, None

    try:
        with open(lib_path, "r") as f:
            library = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[visualization] could not read roi_library.json ({exc}) — using video as-is (no crop).")
        return video_path, None

    entry = library.get(p.stem)
    if not entry or "preprocessing" not in entry:
        print(f"[visualization] no crop_params for stem {p.stem!r} — using video as-is (no crop).")
        return video_path, None

    try:
        crop = entry["preprocessing"]
        crop_params = {
            "x":     int(crop["x"]),
            "y":     int(crop["y"]),
            "w":     int(crop["w"]),
            "h":     int(crop["h"]),
            "start": int(crop["start"]),
            "end":   int(crop["end"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[visualization] malformed preprocessing entry for {p.stem!r} ({exc}) — using video as-is (no crop).")
        return video_path, None

    return video_path, crop_params


def _draw_fly_marker(
    img,
    x: int,
    y: int,
    color: tuple,
    *,
    tick_len: int = 4,
    tick_thick: int = 1,
    label: str | None = None,
    label_offset: tuple = (10, -10),
    chip_pad: int = 2,
    chip_font_scale: float = 0.4,
    leader_thick: int = 1,
):
    """Crosshair tick at (x, y) + optional chip label with leader line.

    Diagonal ticks (NE/SW, NW/SE) leave the fly body itself visible; the
    ID is drawn inside a small filled rectangle offset from the centroid,
    with a thin line pointing back to the fly.
    """
    cv2.line(img, (x - tick_len, y - tick_len), (x - 1, y - 1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x + 1, y + 1), (x + tick_len, y + tick_len), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x - tick_len, y + tick_len), (x - 1, y + 1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x + 1, y - 1), (x + tick_len, y - tick_len), color, tick_thick, cv2.LINE_AA)

    if label is None:
        return

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, chip_font_scale, 1)
    cx = x + label_offset[0]
    cy = y + label_offset[1]
    x0, y0 = cx - chip_pad, cy - th - chip_pad
    x1, y1 = cx + tw + chip_pad, cy + chip_pad

    cv2.line(img, (x, y), (x0, y1), color, leader_thick, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                chip_font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_vial_overlay_video(
    video_path: str,
    csv_path: str,
    out_mp4: str,
    frame_offset: int = 0,
    invert_y: bool = False,
    start: int = 0,
    end: int = -1,
    step: int = 1,
    fps_out: int = 30,
    show_ids: bool = True,
    tick_len: int = 4,
    tick_thick: int = 1,
    chip_font_scale: float = 0.4,
    label_offset_x: int = 10,
    label_offset_y: int = -10,
    chip_pad: int = 2,
    leader_thick: int = 1,
):
    """
    Render an overlay video where flies are coloured by vial and shaded
    by compact_id within each vial.

    Color scheme (HSV hue per vial):
        vial1: blue    vial4: orange
        vial2: green   vial5: pink/magenta
        vial3: yellow  vial6: purple

    Parameters
    ----------
    video_path : str
        Path to the original (or preprocessed) video.
    csv_path : str
        compact_tracks CSV from assign_vials_and_compact_ids().
    out_mp4 : str
        Output path for the annotated video.
    frame_offset : int
        Add this value to the CSV frame index when matching video frames.
    invert_y : bool
        Flip y coordinate (if tracker and video have opposite y-origins).
    start, end : int
        Render frames [start, end] inclusive.  end=-1 means last frame.
    step : int
        Render every N-th frame (1 = every frame).
    fps_out : int
        Output frame rate.
    show_ids : bool
        Overlay compact_id chip next to each fly.
    tick_len, tick_thick : int
        Half-length and stroke width of the crosshair at the fly centroid.
    chip_font_scale : float
        Font scale for the ID chip text.
    label_offset_x, label_offset_y : int
        Chip anchor relative to the fly centroid (pixels).
    chip_pad : int
        Padding inside the chip background rectangle.
    leader_thick : int
        Stroke width of the line connecting fly to chip.
    """

    VIAL_HUE = {
        "vial1": 120,  # blue
        "vial2": 60,   # green
        "vial3": 30,   # yellow
        "vial4": 15,   # orange
        "vial5": 165,  # pink
        "vial6": 135,  # purple
    }

    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(int)
    df["compact_id"] = df["compact_id"].astype(int)
    df["vial_id"] = df["vial_id"].astype(str)

    by_frame = {
        int(f): g[["x", "y", "vial_id", "compact_id"]].to_numpy()
        for f, g in df.groupby("frame")
    }
    max_in_vial = df.groupby("vial_id")["compact_id"].max().to_dict()

    def color_for(vial_id: str, cid: int) -> tuple:
        hue = int(VIAL_HUE.get(vial_id, 0))
        m = int(max_in_vial.get(vial_id, cid))
        v = 235 if m <= 1 else int(120 + (cid - 1) / (m - 1) * (255 - 120))
        hsv = np.uint8([[[hue, 240, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    effective_path, crop_params = _resolve_overlay_source(video_path)

    cap = cv2.VideoCapture(effective_path)
    if not cap.isOpened():
        raise FileNotFoundError(effective_path)

    if crop_params is not None:
        w = crop_params["w"]
        h = crop_params["h"]
        cx0 = crop_params["x"]
        cy0 = crop_params["y"]
        raw_start = crop_params["start"]
        n_frames = crop_params["end"] - crop_params["start"]
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cx0 = cy0 = 0
        raw_start = 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end == -1 or end >= n_frames:
        end = n_frames - 1

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    label_offset = (label_offset_x, label_offset_y)

    for frame_idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_start + frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if crop_params is not None:
            frame_bgr = frame_bgr[cy0:cy0 + h, cx0:cx0 + w]
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                break

        dets = by_frame.get(int(frame_idx + frame_offset))
        if dets is not None:
            for x, y, vial_id, cid in dets:
                xi = int(round(float(x)))
                yi_raw = float(y)
                yi = int(round((h - 1 - yi_raw) if invert_y else yi_raw))

                if 0 <= xi < w and 0 <= yi < h:
                    cid = int(cid)
                    vial_id = str(vial_id)
                    _draw_fly_marker(
                        frame_bgr, xi, yi, color_for(vial_id, cid),
                        tick_len=tick_len, tick_thick=tick_thick,
                        label=str(cid) if show_ids else None,
                        label_offset=label_offset,
                        chip_pad=chip_pad, chip_font_scale=chip_font_scale,
                        leader_thick=leader_thick,
                    )

        writer.write(frame_bgr)

    cap.release()
    writer.release()
    print("Saved overlay video:", out_mp4)


def render_raw_overlay_video(
    video_path: str,
    csv_path: str,
    out_mp4: str,
    frame_offset: int = 0,
    invert_y: bool = False,
    start: int = 0,
    end: int = -1,
    step: int = 1,
    fps_out: int = 30,
    show_ids: bool = True,
    tick_len: int = 4,
    tick_thick: int = 1,
    chip_font_scale: float = 0.4,
    label_offset_x: int = 10,
    label_offset_y: int = -10,
    chip_pad: int = 2,
    leader_thick: int = 1,
):
    """
    Render an overlay video from raw long-format tracks (frame, orig_id, x, y).

    Each orig_id gets a distinct hue, evenly spaced in HSV space.
    Used to inspect raw OC-SORT output before stitching.
    """
    df = pd.read_csv(csv_path)
    df["frame"]   = df["frame"].astype(int)
    df["orig_id"] = df["orig_id"].astype(str)

    unique_ids = sorted(df["orig_id"].unique(), key=lambda v: int(v) if v.isdigit() else v)
    n = len(unique_ids)
    id_to_hue = {oid: int(i * 180 / max(n, 1)) for i, oid in enumerate(unique_ids)}

    by_frame = {
        int(f): g[["x", "y", "orig_id"]].to_numpy()
        for f, g in df.groupby("frame")
    }

    def color_for(oid: str) -> tuple:
        hsv = np.uint8([[[id_to_hue[oid], 230, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    effective_path, crop_params = _resolve_overlay_source(video_path)

    cap = cv2.VideoCapture(effective_path)
    if not cap.isOpened():
        raise FileNotFoundError(effective_path)

    if crop_params is not None:
        w = crop_params["w"]
        h = crop_params["h"]
        cx0 = crop_params["x"]
        cy0 = crop_params["y"]
        raw_start = crop_params["start"]
        n_frames = crop_params["end"] - crop_params["start"]
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cx0 = cy0 = 0
        raw_start = 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end == -1 or end >= n_frames:
        end = n_frames - 1

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    label_offset = (label_offset_x, label_offset_y)

    for frame_idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_start + frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if crop_params is not None:
            frame_bgr = frame_bgr[cy0:cy0 + h, cx0:cx0 + w]
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                break

        dets = by_frame.get(int(frame_idx + frame_offset))
        if dets is not None:
            for x, y, oid in dets:
                xi   = int(round(float(x)))
                yi_r = float(y)
                yi   = int(round((h - 1 - yi_r) if invert_y else yi_r))
                if 0 <= xi < w and 0 <= yi < h:
                    _draw_fly_marker(
                        frame_bgr, xi, yi, color_for(str(oid)),
                        tick_len=tick_len, tick_thick=tick_thick,
                        label=str(oid) if show_ids else None,
                        label_offset=label_offset,
                        chip_pad=chip_pad, chip_font_scale=chip_font_scale,
                        leader_thick=leader_thick,
                    )

        writer.write(frame_bgr)

    cap.release()
    writer.release()
    print("Saved raw overlay video:", out_mp4)


def render_detections_video(
    video_path: str,
    det_log_csv: str,
    out_mp4: str,
    fps_out: int = 30,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.4,
    show_conf: bool = True,
) -> None:
    """
    Render raw RF-DETR detection bounding boxes on the source video.

    Draws one rectangle per detection per frame, before any tracking or
    association. Useful for diagnosing missed detections vs. tracking errors.

    Parameters
    ----------
    det_log_csv : path to CSV written by tracking.py when det_log_csv is given.
                  Columns: frame, x1, y1, x2, y2, conf.
    color       : BGR tuple for the rectangle and confidence text.
    """
    det_df = pd.read_csv(det_log_csv)
    by_frame = {
        int(f): grp[["x1", "y1", "x2", "y2", "conf"]].values.tolist()
        for f, grp in det_df.groupby("frame")
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        for x1, y1, x2, y2, conf in by_frame.get(frame_idx, []):
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(frame_bgr, pt1, pt2, color, thickness)
            if show_conf:
                cv2.putText(frame_bgr, f"{conf:.2f}", (pt1[0], pt1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        writer.write(frame_bgr)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Saved detections video:", out_mp4)
