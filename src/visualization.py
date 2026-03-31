"""
src/visualization.py

Render overlay video: fly positions coloured by vial, shaded by compact_id.

Expects compact_tracks CSV columns: frame, x, y, vial_id, compact_id.
"""

import os
import cv2
import numpy as np
import pandas as pd


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
    radius: int = 5,
    show_ids: bool = True,
    font_scale: float = 0.5,
    text_thick: int = 1,
    outline_thick: int = 2,
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
    radius : int
        Dot radius in pixels.
    show_ids : bool
        Overlay compact_id text next to each dot.
    font_scale, text_thick, outline_thick : float/int
        OpenCV text rendering parameters.
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

    def put_text_outlined(img, text, org):
        for col, thick in [((0, 0, 0), outline_thick), ((255, 255, 255), text_thick)]:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, col, thick, cv2.LINE_AA)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end == -1 or end >= n_frames:
        end = n_frames - 1

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    for frame_idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
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
                    cv2.circle(frame_bgr, (xi, yi), radius, color_for(vial_id, cid), -1)
                    if show_ids:
                        put_text_outlined(frame_bgr, str(cid), (xi + 8, yi - 8))

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
    radius: int = 5,
    show_ids: bool = True,
    font_scale: float = 0.5,
    text_thick: int = 1,
    outline_thick: int = 2,
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

    def put_text_outlined(img, text, org):
        for col, thick in [((0, 0, 0), outline_thick), ((255, 255, 255), text_thick)]:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, col, thick, cv2.LINE_AA)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end == -1 or end >= n_frames:
        end = n_frames - 1

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    for frame_idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        dets = by_frame.get(int(frame_idx + frame_offset))
        if dets is not None:
            for x, y, oid in dets:
                xi   = int(round(float(x)))
                yi_r = float(y)
                yi   = int(round((h - 1 - yi_r) if invert_y else yi_r))
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(frame_bgr, (xi, yi), radius, color_for(str(oid)), -1)
                    if show_ids:
                        put_text_outlined(frame_bgr, str(oid), (xi + 8, yi - 8))

        writer.write(frame_bgr)

    cap.release()
    writer.release()
    print("Saved raw overlay video:", out_mp4)
