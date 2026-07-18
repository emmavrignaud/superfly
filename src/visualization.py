"""
src/visualization.py

Render overlay video: fly positions coloured by vial, shaded by ordered_id.

Expects ordered_tracks CSV columns: frame, x, y, vial_id, ordered_id.
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


def _hex_to_hue_sat(hexstr: str) -> tuple[int, int]:
    """Return (hue, saturation) in OpenCV's 0-179 / 0-255 HSV scale for a #rrggbb.

    Used to let a user's per-vial colour drive the overlay's per-vial hue while
    the existing brightness-by-ordered_id shading is kept.
    """
    h = hexstr.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1])


def _roi_library_path() -> Path:
    return Path(__file__).parent.parent / "roi_library.json"


def _resolve_overlay_source(video_path: str) -> tuple[str, dict | None]:
    """Pick the substrate the overlay should be drawn on.

    Returns (effective_video_path, crop_params or None).

    - ``overlay_source: processed`` (or unknown / fallback): returns the caller's
      path unchanged; caller reads frames as-is (current behavior).
    - ``overlay_source: raw_cropped``:
        If ``video_path`` points to an existing ``*_raw_cropped`` file, returns it
        with ``crop_params=None`` (already cropped on disk).
        Otherwise strips a ``_pp`` suffix when present, resolves the raw path,
        loads ``crop_params`` from roi_library.json when needed, and returns the
        path to decode plus crop metadata.

    Any missing file / missing library entry falls back silently (with a warning)
    to the processed substrate — visualization must never crash the pipeline.
    """
    cfg = _visualization_cfg()
    mode = str(cfg.get("overlay_source", "raw_cropped")).lower()
    if mode == "processed":
        return video_path, None
    if mode != "raw_cropped":
        print(f"[visualization] unknown overlay_source {mode!r} — using processed substrate.")
        return video_path, None

    p = Path(video_path).expanduser()
    # Saved cropped-raw artifact: same coordinate system as *_pp; no library crop pass.
    if p.is_file() and p.stem.endswith("_raw_cropped"):
        return str(p.resolve()), None

    raw_path = p.with_name(p.stem[:-3]).with_suffix(p.suffix) if p.stem.endswith("_pp") else p

    if not raw_path.exists():
        print(f"[visualization] raw video not found at {raw_path} — falling back to processed substrate.")
        return video_path, None

    lib_path = _roi_library_path()
    if not lib_path.exists():
        print("[visualization] roi_library.json not found — falling back to processed substrate.")
        return video_path, None

    try:
        with open(lib_path, "r") as f:
            library = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[visualization] could not read roi_library.json ({exc}) — falling back to processed substrate.")
        return video_path, None

    stem = raw_path.stem
    entry = library.get(stem)
    # roi_library keys use the main clip stem (e.g. ``…-converted``); overlay may pass ``…-converted_raw_cropped``.
    if (not entry or "preprocessing" not in entry) and stem.endswith("_raw_cropped"):
        entry = library.get(stem.removesuffix("_raw_cropped"))
    if not entry or "preprocessing" not in entry:
        print(f"[visualization] no preprocessing crop_params for stem {raw_path.stem!r} — falling back to processed substrate.")
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
        print(f"[visualization] malformed preprocessing entry for {raw_path.stem!r} ({exc}) — falling back to processed substrate.")
        return video_path, None

    return str(raw_path), crop_params


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
    chip_pad: int = 1,
    chip_font_scale: float = 0.32,
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


def _compute_unmatched_by_frame(det_log_csv, track_centers_by_frame, k):
    """Per-frame detection bboxes that no tracker centroid covers.

    A detection is "covered" if any track centroid in the same frame falls
    within the bbox padded by (k * std(width), k * std(height)).
    """
    det_df = pd.read_csv(det_log_csv)
    if det_df.empty:
        return {}, 0, 0
    widths = (det_df["x2"] - det_df["x1"]).to_numpy()
    heights = (det_df["y2"] - det_df["y1"]).to_numpy()
    tol_x = float(k) * float(np.std(widths)) if len(widths) > 1 else 0.0
    tol_y = float(k) * float(np.std(heights)) if len(heights) > 1 else 0.0

    unmatched = {}
    total_dets = 0
    total_missed = 0
    for f, grp in det_df.groupby("frame"):
        f = int(f)
        tracks = track_centers_by_frame.get(f, [])
        misses = []
        for x1, y1, x2, y2 in grp[["x1", "y1", "x2", "y2"]].to_numpy():
            total_dets += 1
            pad_x1, pad_y1 = x1 - tol_x, y1 - tol_y
            pad_x2, pad_y2 = x2 + tol_x, y2 + tol_y
            covered = any(pad_x1 <= tx <= pad_x2 and pad_y1 <= ty <= pad_y2
                          for tx, ty in tracks)
            if not covered:
                misses.append((float(x1), float(y1), float(x2), float(y2)))
                total_missed += 1
        if misses:
            unmatched[f] = misses
    return unmatched, total_dets, total_missed


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
    chip_font_scale: float = 0.32,
    label_offset_x: int = 10,
    label_offset_y: int = -10,
    chip_pad: int = 1,
    leader_thick: int = 1,
    vial_rois: dict | None = None,
    det_log_csv: str | None = None,
    vial_colors: dict | None = None,
):
    """
    Render an overlay video where flies are coloured by vial and shaded
    by ordered_id within each vial.

    Color scheme (HSV hue per vial):
        vial1: blue    vial4: orange
        vial2: green   vial5: pink/magenta
        vial3: yellow  vial6: purple

    ``vial_colors`` ({vial_id: "#rrggbb"}), when given, overrides the default hue
    for that vial with the hue/saturation of the chosen colour (picked in the
    setup window). Within-vial brightness shading by ordered_id is preserved, so
    the colour a user picks becomes that vial's family across the overlay.

    Parameters
    ----------
    video_path : str
        Path to the original (or preprocessed) video.
    csv_path : str
        ordered_tracks CSV from assign_vials_and_ordered_ids().
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
        Overlay ordered_id chip next to each fly.
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
        "vial3": 90,   # cyan
        "vial4": 15,   # orange
        "vial5": 165,  # pink
        "vial6": 135,  # purple
    }

    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(int)
    df["ordered_id"] = df["ordered_id"].astype(int)
    df["vial_id"] = df["vial_id"].astype(str)

    by_frame = {
        int(f): g[["x", "y", "vial_id", "ordered_id"]].to_numpy()
        for f, g in df.groupby("frame")
    }
    max_in_vial = df.groupby("vial_id")["ordered_id"].max().to_dict()

    cfg = _visualization_cfg()
    show_unmatched = bool(cfg.get("show_unmatched_detections", False))
    k = float(cfg.get("unmatched_tolerance_k", 2.0))
    unmatched = None
    if show_unmatched and det_log_csv and os.path.exists(det_log_csv):
        centers = {f: [(float(r[0]), float(r[1])) for r in arr]
                   for f, arr in by_frame.items()}
        unmatched, n_dets, n_missed = _compute_unmatched_by_frame(det_log_csv, centers, k)
        print(f"  unmatched detections: {n_missed} / {n_dets}")

    _vial_colors = vial_colors or {}

    def color_for(vial_id: str, cid: int) -> tuple:
        override = _vial_colors.get(vial_id)
        if override:
            hue, sat = _hex_to_hue_sat(override)
        else:
            hue, sat = int(VIAL_HUE.get(vial_id, 0)), 240
        m = int(max_in_vial.get(vial_id, cid))
        v = 235 if m <= 1 else int(120 + (cid - 1) / (m - 1) * (255 - 120))
        hsv = np.uint8([[[hue, sat, v]]])
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
                if not (np.isfinite(float(x)) and np.isfinite(float(y))):
                    continue
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

        if unmatched is not None:
            for x1, y1, x2, y2 in unmatched.get(int(frame_idx + frame_offset), []):
                cv2.rectangle(frame_bgr,
                              (int(round(x1)), int(round(y1))),
                              (int(round(x2)), int(round(y2))),
                              (0, 0, 255), 1, cv2.LINE_AA)

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
    chip_font_scale: float = 0.32,
    label_offset_x: int = 10,
    label_offset_y: int = -10,
    chip_pad: int = 1,
    leader_thick: int = 1,
    vial_rois: dict | None = None,
    det_log_csv: str | None = None,
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

    cfg = _visualization_cfg()
    show_unmatched = bool(cfg.get("show_unmatched_detections", False))
    k = float(cfg.get("unmatched_tolerance_k", 2.0))
    unmatched = None
    if show_unmatched and det_log_csv and os.path.exists(det_log_csv):
        centers = {f: [(float(r[0]), float(r[1])) for r in arr]
                   for f, arr in by_frame.items()}
        unmatched, n_dets, n_missed = _compute_unmatched_by_frame(det_log_csv, centers, k)
        print(f"  unmatched detections: {n_missed} / {n_dets}")

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
                fx, fy = float(x), float(y)
                if not (np.isfinite(fx) and np.isfinite(fy)):
                    continue
                xi   = int(round(fx))
                yi_r = fy
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

        if unmatched is not None:
            for x1, y1, x2, y2 in unmatched.get(int(frame_idx + frame_offset), []):
                cv2.rectangle(frame_bgr,
                              (int(round(x1)), int(round(y1))),
                              (int(round(x2)), int(round(y2))),
                              (0, 0, 255), 1, cv2.LINE_AA)

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
