"""
src/visualization.py

Render overlay video: fly positions coloured by vial, brightness-graded by compact_id.

Expects compact_tracks CSV columns: frame, x, y, vial_id, compact_id.
"""

import json
import os
from pathlib import Path

import colorcet as cc
import cv2
import numpy as np
import pandas as pd
import yaml

_GLASBEY = cc.glasbey_bw_minc_20  # 256 perceptually distinct '#rrggbb' colors


def _glasbey_bgr(idx: int) -> tuple:
    """Index into the glasbey palette, return an OpenCV BGR tuple.

    colorcet palettes come in two shapes depending on version: a list of
    '#rrggbb' strings, or a list of (r, g, b) floats in [0, 1]. Handle both.
    """
    entry = _GLASBEY[idx % len(_GLASBEY)]
    if isinstance(entry, str):
        hex_c = entry.lstrip("#")
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
    else:
        r, g, b = (int(round(c * 255)) for c in entry[:3])
    return (b, g, r)


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
    chip_pad: int = 1,
    chip_font_scale: float = 0.32,
    leader_thick: int = 1,
    show_border: bool = True,
    shadow_text: bool = True,
    anchor_radius: int = 0,
    drop_shadow: bool = True,
):
    cv2.line(img, (x - tick_len, y - tick_len), (x - 1, y - 1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x + 1, y + 1), (x + tick_len, y + tick_len), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x - tick_len, y + tick_len), (x - 1, y + 1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x + 1, y - 1), (x + tick_len, y - tick_len), color, tick_thick, cv2.LINE_AA)

    if anchor_radius > 0:
        cv2.circle(img, (x, y), anchor_radius, color, -1, cv2.LINE_AA)

    if label is None:
        return

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, chip_font_scale, 1)
    cw = tw + 2 * chip_pad
    ch = th + 2 * chip_pad
    h_img, w_img = img.shape[:2]
    x0 = x + label_offset[0] - chip_pad
    y0 = y + label_offset[1]  - th - chip_pad
    x0 = max(0, min(x0, w_img - cw))
    y0 = max(0, min(y0, h_img - ch))
    x1 = x0 + cw
    y1 = y0 + ch

    corner_x = x0 if x < x0 else x1
    corner_y = y0 if y < y0 else y1
    cv2.line(img, (x, y), (corner_x, corner_y), color, leader_thick, cv2.LINE_AA)

    if drop_shadow:
        sxa, sya = max(0, x0 + 2), max(0, y0 + 2)
        sxb, syb = min(w_img, x1 + 2), min(h_img, y1 + 2)
        if sxb > sxa and syb > sya:
            roi = img[sya:syb, sxa:sxb]
            img[sya:syb, sxa:sxb] = (roi.astype(np.int16) * 6 // 10).astype(np.uint8)

    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
    if show_border:
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)

    tx = x0 + chip_pad
    ty = y0 + chip_pad + th
    if shadow_text:
        cv2.putText(img, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    chip_font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                chip_font_scale, (255, 255, 255), 1, cv2.LINE_AA)


VIAL_HUE = {
    "vial1": 120, "vial2": 60, "vial3": 30,
    "vial4": 15,  "vial5": 165, "vial6": 135,
}


def _vial_hsv_color(vial_id: str | None, rank: int, max_rank: int) -> tuple:
    """Color rule used by both overlays. Returns BGR.

    `rank` is the within-vial position; brightness varies across ranks so
    same-vial flies remain distinguishable. None vial → neutral grey
    (a detection that didn't fall into any drawn vial — shouldn't happen).
    """
    if vial_id is None or vial_id not in VIAL_HUE:
        return (140, 140, 140)
    hue = VIAL_HUE[vial_id]
    v = 200 if max_rank <= 1 else int(140 + (rank - 1) / (max_rank - 1) * (220 - 140))
    bgr = cv2.cvtColor(np.uint8([[[hue, 150, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _assign_to_vial(x: float, y: float, vial_rois: dict | None) -> str | None:
    if not vial_rois:
        return None
    for vid, (x0, y0, x1, y1) in vial_rois.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return vid
    return None


def _compute_unmatched_by_frame(
    det_log_csv: str,
    track_centers_by_frame: dict,
    k: float,
) -> tuple[dict, int, int]:
    """For each frame, return detection bboxes that no tracker centroid covers.

    A detection is "covered" if any track centroid in the same frame falls
    within the bbox padded by (k * std(width), k * std(height)) — the std
    is computed once across all detections, so tolerance auto-scales to the
    typical detection-box size.

    Returns (unmatched_by_frame, total_detections, total_unmatched).
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


def _render_overlay_core(
    video_path: str,
    out_mp4: str,
    drawables_by_frame: dict,
    unmatched_by_frame: dict | None,
    *,
    frame_offset: int,
    invert_y: bool,
    start: int,
    end: int,
    step: int,
    fps_out: int,
    tick_len: int,
    tick_thick: int,
    chip_font_scale: float,
    label_offset: tuple,
    chip_pad: int,
    leader_thick: int,
    show_border: bool,
    shadow_text: bool,
    anchor_radius: int,
):
    """Shared video plumbing for overlay rendering.

    `drawables_by_frame[frame] = [(xi, yi, color_bgr, label_or_None), ...]`
    `unmatched_by_frame[frame] = [(x1, y1, x2, y2), ...]` for red boxes (or None).
    """
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
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    if raw_start + start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_start + start)

    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (frame_idx - start) % step == 0:
            if crop_params is not None:
                frame_bgr = frame_bgr[cy0:cy0 + h, cx0:cx0 + w]
                if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                    break

            csv_frame = int(frame_idx + frame_offset)

            if unmatched_by_frame is not None:
                for x1, y1, x2, y2 in unmatched_by_frame.get(csv_frame, []):
                    p1 = (int(round(x1)), int(round((h - 1 - y2) if invert_y else y1)))
                    p2 = (int(round(x2)), int(round((h - 1 - y1) if invert_y else y2)))
                    cv2.rectangle(frame_bgr, p1, p2, (0, 0, 220), 1, cv2.LINE_AA)

            for xf, yf, color, label in drawables_by_frame.get(csv_frame, []):
                xi = int(round(xf))
                yi = int(round((h - 1 - yf) if invert_y else yf))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                _draw_fly_marker(
                    frame_bgr, xi, yi, color,
                    tick_len=tick_len, tick_thick=tick_thick,
                    label=label,
                    label_offset=label_offset,
                    chip_pad=chip_pad, chip_font_scale=chip_font_scale,
                    leader_thick=leader_thick,
                    show_border=show_border,
                    shadow_text=shadow_text,
                    anchor_radius=anchor_radius,
                )

            writer.write(frame_bgr)

        frame_idx += 1

    cap.release()
    writer.release()
    return w, h


def _build_drawables_from_csv(
    df: pd.DataFrame,
    id_col: str,
    vial_rois: dict | None,
    label_col: str,
    show_ids: bool,
):
    """Convert (frame, x, y, id...) rows into per-frame float drawables.

    Drawables are `(xf, yf, color, label_or_None)` — int conversion and
    y-flip happen in the render core, which knows the real frame height.

    Vial assignment uses the track's median (x, y); within each vial the
    IDs are ranked left-to-right by median x, and brightness varies by
    rank so same-vial flies stay distinguishable. Detections outside all
    vials get a neutral grey (rare; surfaces real bugs).

    Also returns `track_centers_by_frame` for the missed-detection check.
    """
    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df[id_col] = df[id_col].astype(str)

    if vial_rois:
        med = df.groupby(id_col).agg(_x=("x", "median"), _y=("y", "median"))
        id_to_vial = {
            tid: _assign_to_vial(row._x, row._y, vial_rois)
            for tid, row in med.iterrows()
        }
    else:
        id_to_vial = {tid: None for tid in df[id_col].unique()}

    rank = {}
    max_per_vial = {}
    for vid in set(id_to_vial.values()):
        members = [tid for tid, v in id_to_vial.items() if v == vid]
        medians = df[df[id_col].isin(members)].groupby(id_col)["x"].median()
        ordered = sorted(members, key=lambda t: medians.get(t, 0.0))
        max_per_vial[vid] = len(ordered)
        for i, tid in enumerate(ordered, start=1):
            rank[tid] = i

    drawables = {}
    centers = {}
    for frame, grp in df.groupby("frame"):
        f = int(frame)
        frame_drawables = []
        frame_centers = []
        for x, y, tid, lbl in grp[["x", "y", id_col, label_col]].to_numpy():
            xf, yf = float(x), float(y)
            if not (np.isfinite(xf) and np.isfinite(yf)):
                continue
            tid = str(tid)
            v = id_to_vial.get(tid)
            color = _vial_hsv_color(v, rank.get(tid, 1), max_per_vial.get(v, 1))
            frame_drawables.append((xf, yf, color, str(lbl) if show_ids else None))
            frame_centers.append((xf, yf))
        drawables[f] = frame_drawables
        centers[f] = frame_centers
    return drawables, centers


def render_vial_overlay_video(
    video_path: str,
    csv_path: str,
    out_mp4: str,
    vial_rois: dict | None = None,
    det_log_csv: str | None = None,
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
    show_border: bool = True,
    shadow_text: bool = True,
    anchor_radius: int = 0,
):
    """Stitched/compact overlay: vial-coloured, brightness-graded by compact_id.

    `vial_rois` maps vial id → (x0, y0, x1, y1). Optional but recommended;
    without it, all flies render in neutral grey. `det_log_csv` enables
    red boxes around detections that no tracker covered.
    """
    df = pd.read_csv(csv_path)
    drawables, centers = _build_drawables_from_csv(
        df, id_col="compact_id", vial_rois=vial_rois,
        label_col="compact_id", show_ids=show_ids,
    )

    cfg = _visualization_cfg()
    show_unmatched = bool(cfg.get("show_unmatched_detections", False))
    k = float(cfg.get("unmatched_tolerance_k", 2.0))
    unmatched, n_dets, n_missed = (None, 0, 0)
    if show_unmatched and det_log_csv and os.path.exists(det_log_csv):
        unmatched, n_dets, n_missed = _compute_unmatched_by_frame(det_log_csv, centers, k)

    _render_overlay_core(
        video_path, out_mp4, drawables, unmatched,
        frame_offset=frame_offset, invert_y=invert_y,
        start=start, end=end, step=step, fps_out=fps_out,
        tick_len=tick_len, tick_thick=tick_thick,
        chip_font_scale=chip_font_scale,
        label_offset=(label_offset_x, label_offset_y),
        chip_pad=chip_pad, leader_thick=leader_thick,
        show_border=show_border, shadow_text=shadow_text,
        anchor_radius=anchor_radius,
    )
    print("Saved overlay video:", out_mp4)
    if unmatched is not None:
        print(f"  unmatched detections: {n_missed} / {n_dets}")


def render_raw_overlay_video(
    video_path: str,
    csv_path: str,
    out_mp4: str,
    vial_rois: dict | None = None,
    det_log_csv: str | None = None,
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
    show_border: bool = True,
    shadow_text: bool = True,
    anchor_radius: int = 0,
):
    """Raw OC-SORT overlay: same vial-aware coloring as the stitched overlay,
    keyed by `orig_id` instead of `compact_id`. Pre-stitching, so multiple
    orig_ids may share a vial — they get distinct brightness shades.
    """
    df = pd.read_csv(csv_path)
    drawables, centers = _build_drawables_from_csv(
        df, id_col="orig_id", vial_rois=vial_rois,
        label_col="orig_id", show_ids=show_ids,
    )

    cfg = _visualization_cfg()
    show_unmatched = bool(cfg.get("show_unmatched_detections", False))
    k = float(cfg.get("unmatched_tolerance_k", 2.0))
    unmatched, n_dets, n_missed = (None, 0, 0)
    if show_unmatched and det_log_csv and os.path.exists(det_log_csv):
        unmatched, n_dets, n_missed = _compute_unmatched_by_frame(det_log_csv, centers, k)

    _render_overlay_core(
        video_path, out_mp4, drawables, unmatched,
        frame_offset=frame_offset, invert_y=invert_y,
        start=start, end=end, step=step, fps_out=fps_out,
        tick_len=tick_len, tick_thick=tick_thick,
        chip_font_scale=chip_font_scale,
        label_offset=(label_offset_x, label_offset_y),
        chip_pad=chip_pad, leader_thick=leader_thick,
        show_border=show_border, shadow_text=shadow_text,
        anchor_radius=anchor_radius,
    )
    print("Saved raw overlay video:", out_mp4)
    if unmatched is not None:
        print(f"  unmatched detections: {n_missed} / {n_dets}")


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
