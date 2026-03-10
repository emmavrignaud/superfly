
"""
Imported libraries and packages.
"""

import cv2
import numpy as np
import supervision as sv
from inference import get_model
from boxmot import OcSort
import logging
import os
import ast
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import pandas as pd
import json
from IPython.display import Video
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import mannwhitneyu, kruskal
from scipy.spatial import ConvexHull

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC




# -----------------------------------------------------------------------------
# Preprocessing (optional): interactive ROI + frame range, then background-sub mp4
#
# We sometimes run a quick preprocessing pass to make flies pop more clearly.
# The GUI lets us draw the ROI and choose a frame window; we then compute an
# average background and write a "_pp" video (same folder by default).
#
# Notebook usage:
#   preprocess = True
#   PATH_TO_VID = RAW_VIDEO
#   if preprocess:
#       PATH_TO_VID = Path(preprocess_bgsub_gui_cv2_avg_background(
#           video_path=str(RAW_VIDEO), out_mp4=None, default_end=700, bg_sample_stride=1
#       ))
# -----------------------------------------------------------------------------

# -------------------------
# BG-SUB visualization tuning (saved mp4 only)
# -------------------------
BG_GAIN = 1.2          # higher = flies darker/more contrast
BG_WHITE_LEVEL = 245   # higher = brighter background
BG_DEADZONE = 0.0      # >0 suppresses tiny static mismatches

BG_CODEC = "mp4v"      # common fallback codec for OpenCV VideoWriter


def _bgr_to_gray_float32(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> grayscale float32 (OpenCV ordering)."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.1140 * b + 0.5870 * g + 0.2989 * r


def gui_pick_roi_and_range(video_path: str, default_end: int = 700):
    """
    OpenCV GUI to pick ROI + [start, end_excl).
    Controls:
      - Drag mouse to draw ROI
      - Trackbars: start, end_excl, cur frame
      - ENTER: accept
      - ESC: cancel
    Returns:
      (x, y, w, h, start, end_excl)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = 0
    end = min(default_end, n_frames) if n_frames > 0 else default_end
    cur = 0

    roi = None  # (x,y,w,h)
    drawing = False
    x0 = y0 = 0

    WIN = "Pick ROI/range | drag ROI | ENTER=accept | ESC=cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def clamp():
        nonlocal start, end, cur
        if n_frames > 0:
            start = max(0, min(start, n_frames - 1))
            end = max(start + 1, min(end, n_frames))  # end exclusive
            cur = max(start, min(cur, end - 1))
        else:
            start = max(0, start)
            end = max(start + 1, end)
            cur = max(start, min(cur, end - 1))

    def on_start(v):
        nonlocal start
        start = int(v)
        clamp()

    def on_end(v):
        nonlocal end
        end = int(v)
        clamp()

    def on_cur(v):
        nonlocal cur
        cur = int(v)
        clamp()

    cv2.createTrackbar("start", WIN, start, max(n_frames - 1, 1), on_start)
    cv2.createTrackbar("end_excl", WIN, end, max(n_frames, 1), on_end)
    cv2.createTrackbar("cur", WIN, cur, max(n_frames - 1, 1), on_cur)

    def redraw():
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if not ok:
            return np.zeros((h_vid, w_vid, 3), dtype=np.uint8)

        img = frame.copy()
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            img,
            f"frame={cur}  start={start}  end_excl={end}  total={n_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if roi is None:
            cv2.putText(
                img,
                "Draw ROI with mouse drag",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            x, y, w, h = roi
            cv2.putText(
                img,
                f"ROI x={x} y={y} w={w} h={h}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return img

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, x0, y0, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = redraw()
            cv2.rectangle(img, (x0, y0), (x, y), (255, 0, 0), 2)
            cv2.imshow(WIN, img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x, y
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])

            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)

            # clamp to bounds
            x_min = max(0, min(x_min, w_vid - 1))
            y_min = max(0, min(y_min, h_vid - 1))
            w = max(1, min(w, w_vid - x_min))
            h = max(1, min(h, h_vid - y_min))

            roi = (x_min, y_min, w, h)

    cv2.setMouseCallback(WIN, mouse_cb)

    while True:
        cv2.imshow(WIN, redraw())
        k = cv2.waitKey(20) & 0xFF

        if k == 27:  # ESC
            cv2.destroyWindow(WIN)
            cap.release()
            raise RuntimeError("Cancelled ROI/range selection")

        if k in (13, 10):  # ENTER
            if roi is None:
                print("Draw ROI first.")
                continue
            cv2.destroyWindow(WIN)
            cap.release()
            x, y, w, h = roi
            return x, y, w, h, start, end


def preprocess_bgsub_gui_cv2_avg_background(
    video_path: str,
    out_mp4: str | None = None,
    default_end: int = 700,
    gain: float = BG_GAIN,
    white_level: float = BG_WHITE_LEVEL,
    deadzone: float = BG_DEADZONE,
    codec: str = BG_CODEC,
    bg_sample_stride: int = 1,   # use every frame for bg if 1; faster if 2,3,4...
) -> str:
    """
    GUI-driven ROI/range selection + average-background subtraction using ONLY the video.

    Output path behavior:
      - If out_mp4 is None:
          output = "<same folder>/<original_filename>.bgsub.mp4"
        Example:
          video_path="/a/b/c/myvideo.mp4"
          -> out="/a/b/c/myvideo.bgsub.mp4"

      - If out_mp4 is provided, it writes exactly to that path.
        The parent directory is created if needed.

    Returns:
      The output mp4 path as a string.
    """
    video_path = str(video_path)

    # Pick ROI + range
    x, y, w, h, start, end_excl = gui_pick_roi_and_range(video_path, default_end=default_end)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if fps > 0 else 30.0

    if out_mp4 is None:
        p = Path(video_path)
        out_mp4 = str(p.with_name(p.stem + "_pp").with_suffix(p.suffix))

    if n_frames > 0:
        end_excl = min(end_excl, n_frames)

    # -------------------------------------------------------------
    # 1) Compute average background over the selected range (ROI only)
    # ------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    acc = np.zeros((h, w), dtype=np.float64)
    count = 0

    for f in range(start, end_excl):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if bg_sample_stride > 1 and ((f - start) % bg_sample_stride != 0):
            continue

        roi_bgr = frame_bgr[y:y + h, x:x + w]
        if roi_bgr.shape[0] != h or roi_bgr.shape[1] != w:
            cap.release()
            raise ValueError("ROI out of bounds for this video/frame during background computation.")

        gray = _bgr_to_gray_float32(roi_bgr)  # float32
        acc += gray.astype(np.float64)
        count += 1

    if count == 0:
        cap.release()
        raise RuntimeError("No frames were available to compute the average background.")

    bg_gray = (acc / float(count)).astype(np.float32)

    # --------------------------------
    # 2) Write bg-sub video over the selected range
    # -------------------------
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    # Seek again to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for f in range(start, end_excl):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        roi_bgr = frame_bgr[y:y + h, x:x + w]
        if roi_bgr.shape[0] != h or roi_bgr.shape[1] != w:
            cap.release()
            writer.release()
            raise ValueError("ROI out of bounds for this video/frame.")

        gray = _bgr_to_gray_float32(roi_bgr)

        # One-sided difference so dark stationary flies still show
        motion = np.maximum(bg_gray - gray, 0.0)

        if deadzone and deadzone > 0:
            motion = np.maximum(motion - float(deadzone), 0.0)

        vis = float(white_level) - motion * float(gain)
        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)

        out_bgr = cv2.cvtColor(vis_u8, cv2.COLOR_GRAY2BGR)
        writer.write(out_bgr)

    cap.release()
    writer.release()
    print("Saved bgsub video:", out_mp4)
    print(f"Background computed from {count} frames (stride={bg_sample_stride}).")
    return out_mp4




logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)



# RF-DETR + ByteTrack export to (x,y) tuple CSV utility

# -----------------------------------------------------------------------------
# RF-DETR + OC-SORT export: write the wide "(x, y)" tracking CSV
#
# This is the first real pipeline step after choosing PATH_TO_VID.
# We run RF-DETR on each frame, feed detections to OC-SORT, and save a wide CSV
# (one id per column, "(x,y)" per frame).
#
# Notebook usage:
#   export_tracks_xy_tuple_csv_one_config(
#       video_path=PATH_TO_VID,
#       output_csv=os.path.join(OUTPUT_PATH, "tracks_wide_format.csv"),
#       api_key=api_key, model_id=model_id,
#       confidence=confidence,
#       track_activation_threshold=track_activation_threshold,
#       lost_track_buffer=lost_track_buffer,
#       minimum_matching_threshold=minimum_matching_threshold,
#       minimum_consecutive_frames=minimum_consecutive_frames,
#       max_frames=None, use_bottom_sensitive=True
#   )
# -----------------------------------------------------------------------------

def export_tracks_xy_tuple_csv_one_config(
    video_path: str,
    output_csv: str,
    api_key: str,
    model_id: str,
    confidence: float = 0.10,
    track_activation_threshold: float = 0.10,   # 
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
) -> pd.DataFrame:
    """
    Runs RF-DETR detection + OC-SORT for ONE configuration and writes a CSV where:
      - rows = frame index
      - columns = track IDs (id{N})
      - each cell = "(x, y)" (center of bbox) if actively output by the tracker
      - empty cell if not present in that frame

    Returns the DataFrame that was saved.
    """

    # Load detector once
    model = get_model(model_id=model_id, api_key=api_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_assumed is None:
        fps_assumed = float(fps) if fps and fps > 0 else 30.0

    # OC-SORT tracker
    tracker = OcSort(
        det_thresh=confidence,  
        max_age=lost_track_buffer,
        min_hits=minimum_consecutive_frames,
        iou_threshold=minimum_matching_threshold,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    )

    rows = []                 # list of dicts, one dict per frame
    all_track_ids = set()     # track IDs observed at least once
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        # Detector output
        results = model.infer(frame, confidence=confidence)[0]
        dets = sv.Detections.from_inference(results)

        frame_row = {"frame": frame_idx}

        if len(dets) > 0:
            # Convert to OC-SORT format: [x1,y1,x2,y2,conf,class]
            det_array = np.hstack([
                dets.xyxy,
                dets.confidence[:, None],
                dets.class_id[:, None],
            ])

            tracks = tracker.update(det_array, frame)

            if tracks is not None and len(tracks) > 0:
                # Normalize shape: OC-SORT may return (7,) for one track
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

    # Build DataFrame with consistent columns
    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)

    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")

    return df



#
# -----------------------------------------------------------------------------
# Interactive ROI drawing utility for fly vials
# 
# Vial ROIs (one-time run: draw the 6 vial rectangles and save to JSON
#
# This is manual so that different formats of video can be used. Thenvial positions are stable, 
# are stable during the pipeline and we can use them to assign a vial id to each track.
# 
#
# Notebook usage:
#   draw_and_save_vial_rois(
#       video_path=PATH_TO_VID,
#       roi_json_path=os.path.join(OUTPUT_PATH, "vial_rois.json")
#   )
# -----------------------------------------------------------------------------

def draw_and_save_vial_rois(
    video_path: str,
    roi_json_path: str,
    frame_idx: int = 0,
    n_vials: int = 6,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Interactive utility to manually draw rectangular ROIs for fly vials
    on a fixed reference frame of a video.

    We assume:
    - The experimental setup is static.
    - Vial positions do not change across the video.
    - Exactly 6 vials are present (default).
    - A single reference frame (default: frame 0) is sufficient.

    Controls:
    - Mouse drag: draw ROI
    - u: undo last ROI
    - r: reset all ROIs
    - q: finish (only allowed when exactly `n_vials` ROIs are drawn)
    - ESC: cancel

    Parameters
    ----------
    video_path : str
        Absolute path to the experiment video.
        This should be the same video used for the entire tracking run.
    roi_json_path : str
        Where the resulting ROIs will be saved as JSON.
    frame_idx : int, optional
        Frame index used as reference for drawing ROIs.
        Defaults to 0. Should not need to change for static setups.
    n_vials : int, optional
        Number of vials (ROIs). Defaults to 6.

    Returns
    -------
    Dict[str, Tuple[int, int, int, int]]
        Dictionary mapping vial IDs to (x0, y0, x1, y1),
        sorted left → right by center x-coordinate.

    Raises
    ------
    RuntimeError
        If the reference frame cannot be read.
    """

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, base = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Could not read reference frame from video")

    rois = []
    drawing = False
    ix, iy = -1, -1

    WIN = "Draw ROIs: drag=add | u=undo | r=reset | q=finish"

    def redraw():
        img = base.copy()
        for k, (x0, y0, x1, y1) in enumerate(rois, start=1):
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{k}",
                (x0 + 5, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            f"ROIs: {len(rois)}/{n_vials}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    img_show = redraw()

    def mouse_cb(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img_show, rois

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = redraw()
            cv2.rectangle(tmp, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow(WIN, tmp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            x1, y1 = max(ix, x), max(iy, y)
            rois.append((x0, y0, x1, y1))
            img_show = redraw()
            cv2.imshow(WIN, img_show)
            print(f"Added ROI {len(rois)} = ({x0}, {y0}, {x1}, {y1})")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)
    cv2.imshow(WIN, img_show)

    print("Controls:")
    print(" - Drag mouse: add ROI")
    print(" - u: undo last ROI")
    print(" - r: reset all ROIs")
    print(f" - q: finish (requires exactly {n_vials} ROIs)")
    print("Tip: click the ROI window once so it receives keyboard input.")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and rois:
            print("Undo:", rois.pop())
            img_show = redraw()
            cv2.imshow(WIN, img_show)

        elif key == ord("r"):
            rois.clear()
            print("Reset.")
            img_show = redraw()
            cv2.imshow(WIN, img_show)

        elif key == ord("q"):
            if len(rois) == n_vials:
                break
            print(f"Not finishing: {len(rois)}/{n_vials} ROIs")

        elif key == 27:  # ESC
            rois.clear()
            break

    cv2.destroyAllWindows()

    if len(rois) != n_vials:
        raise RuntimeError("ROI selection cancelled or incomplete")

    rois_sorted = sorted(rois, key=lambda r: (r[0] + r[2]) / 2.0)

    roi_dict = {
        f"vial{i}": tuple(r)
        for i, r in enumerate(rois_sorted, start=1)
    }

    with open(roi_json_path, "w") as f:
        json.dump({k: list(v) for k, v in roi_dict.items()}, f, indent=2)

    print("Saved ROIs to:", roi_json_path)
    return roi_dict



# -----------------------------------------------------------------------------
# 
#  Parses long / wide CSV tracking formats to make it readable
# 
# CSV helpers: wide tracking CSV -> long rows (frame, id, x, y)
#
# The tracker export is "wide": one column per id, and each cell is "(x, y)".
# For stitching and analysis it’s easier to work in long format, so we convert it.
#
# Example in practice:
#   df_wide = pd.read_csv("tracks_wide_format.csv")
#   df_long = wide_to_long(df_wide)
# -----------------------------------------------------------------------------

def parse_xy_cell(cell) -> Optional[Tuple[float, float]]:
    """
    Parse a cell containing (x, y) coordinates.

    Accepts:
    - tuple / list of length 2
    - string representation "(x, y)"
    - NaN / None → returns None
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, (tuple, list)) and len(cell) == 2:
        return float(cell[0]), float(cell[1])

    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None

    try:
        xy = ast.literal_eval(s)
        if isinstance(xy, (tuple, list)) and len(xy) == 2:
            return float(xy[0]), float(xy[1])
    except Exception:
        return None

    return None


def wide_to_long(df_wide: pd.DataFrame,
                 frame_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convert wide tracking CSV (one column per ID) to long format.

    Output columns: frame, orig_id, x, y
    """
    df = df_wide.copy()

    if frame_col is None:
        if "frame" in df.columns:
            frame_col = "frame"
        elif "Frame" in df.columns:
            frame_col = "Frame"

    if frame_col is None:
        df = df.reset_index().rename(columns={"index": "frame"})
        frame_col = "frame"

    id_cols = [c for c in df.columns if c != frame_col]

    records = []
    for _, row in df.iterrows():
        f = int(row[frame_col])
        for c in id_cols:
            xy = parse_xy_cell(row[c])
            if xy is None:
                continue
            x, y = xy
            records.append((f, str(c), float(x), float(y)))

    out = pd.DataFrame(records, columns=["frame", "orig_id", "x", "y"])
    out.sort_values(["orig_id", "frame"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out



# -----------------------------------------------------------------------------
# Tracklet summaries + step scale
#
# Before stitching, we collapse each orig_id into a "tracklet" (start/end + points),
# and estimate a typical step size. That step scale is what normalizes the link cost
# so we don't over-link things that jump too far
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tracklet:
    orig_id: str
    start_frame: int
    end_frame: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    n_points: int


def build_tracklets(long_df: pd.DataFrame) -> List[Tracklet]:
    """
    Collapse each orig_id into a single tracklet summary.
    """
    tracklets = []
    for oid, g in long_df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")
        tracklets.append(
            Tracklet(
                orig_id=str(oid),
                start_frame=int(g2.iloc[0]["frame"]),
                end_frame=int(g2.iloc[-1]["frame"]),
                start_xy=(float(g2.iloc[0]["x"]), float(g2.iloc[0]["y"])),
                end_xy=(float(g2.iloc[-1]["x"]), float(g2.iloc[-1]["y"])),
                n_points=int(len(g2)),
            )
        )
    tracklets.sort(key=lambda t: t.orig_id)
    return tracklets


def estimate_step_scale(long_df: pd.DataFrame) -> Dict[str, float]:
    """
    Robust estimate of step-size statistics across all tracklets.
    """
    steps = []
    for _, g in long_df.groupby("orig_id", sort=False):
        g2 = g.sort_values("frame")
        f = g2["frame"].to_numpy()
        x = g2["x"].to_numpy()
        y = g2["y"].to_numpy()

        consec = (f[1:] - f[:-1]) == 1
        if not np.any(consec):
            continue

        dx = x[1:][consec] - x[:-1][consec]
        dy = y[1:][consec] - y[:-1][consec]
        steps.extend(np.sqrt(dx*dx + dy*dy).tolist())

    if len(steps) == 0:
        return {"median_step": 10.0, "mad_step": 5.0, "sigma_step": 10.0}

    steps = np.asarray(steps)
    med = float(np.median(steps))
    mad = float(np.median(np.abs(steps - med)))

    # Convert MAD to a sigma-like scale and keep a minimum so denom never collapses.
    sigma = float(max(1.4826 * mad, 0.25 * med, 2.0))

    return {
        "median_step": med,
        "mad_step": mad,
        "sigma_step": sigma,
    }




# Link cost function for OC-SORT
# -----------------------------------------------------------------------------
# Stitching internals: link cost + assignment
#
# Given fragmented tracklets, we build candidate links across small gaps and pick
# the best non-conflicting matches (Hungarian if available, greedy fallback).
# The output is a mapping orig_id -> stitched_id.
# -----------------------------------------------------------------------------

def link_cost(end_xy, start_xy, gap, sigma_step, gap_penalty) -> float:
    dx = start_xy[0] - end_xy[0]
    dy = start_xy[1] - end_xy[1]
    dist = math.sqrt(dx*dx + dy*dy)

    # Normalize distance by an estimated per-frame step scale.
    # The sqrt(gap) term makes longer gaps a bit more forgiving (but still penalized).
    denom = sigma_step * math.sqrt(max(gap, 1))
    z = dist / max(denom, 1e-6)

    # Cost = normalized squared distance + linear gap penalty.
    # Lower is "more plausible link".
    return (z * z) + gap_penalty * gap

def build_cost_matrix(tracklets: List[Tracklet], max_gap: int, sigma_step: float,
                      gap_penalty: float, max_cost: float) -> np.ndarray:
    n = len(tracklets)
    BIG = 1e9

    # BIG entries mean "do not allow this link" (treated as impossible by assignment).
    C = np.full((n, n), BIG, dtype=float)

    for i, ti in enumerate(tracklets):
        for j, tj in enumerate(tracklets):
            if i == j:
                continue
            if ti.end_frame >= tj.start_frame:
                continue

            # Only consider forward-in-time links within the allowed gap window.
            gap = tj.start_frame - ti.end_frame
            if gap < 1 or gap > max_gap:
                continue

            c = link_cost(ti.end_xy, tj.start_xy, gap, sigma_step, gap_penalty)

            # Prune very expensive links so the solver only sees plausible candidates.
            if c <= max_cost:
                C[i, j] = c
    return C

# Assignment solver (Hungarian + greedy fallback)

def solve_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    BIG = 1e9
    C = cost_matrix

    # Prefer Hungarian (global minimum-cost 1-to-1 assignment) when SciPy is available.
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        matches = []
        for i, j in zip(r, c):
            if C[i, j] < BIG / 10:
                matches.append((int(i), int(j), float(C[i, j])))
        return matches
    except Exception:
        # Fallback: greedy matching by cheapest edges first (still enforces 1-to-1).
        matches = []
        used_r, used_c = set(), set()
        edges = np.argwhere(C < BIG / 10)
        edges = [(int(i), int(j), float(C[i, j])) for i, j in edges]
        edges.sort(key=lambda t: t[2])
        for i, j, cost in edges:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            matches.append((i, j, cost))
        return matches


# Build orig_id → stitched_id mapping from matches
def build_orig_to_stitched(tracklets: List[Tracklet], matches: List[Tuple[int, int, float]]) -> Dict[str, str]:
    succ = {}
    pred = {}

    # succ[i]=j means tracklet i is linked to successor j (one outgoing max)
    # pred[j]=i means tracklet j has predecessor i (one incoming max)
    for i, j, _ in matches:
        if i in succ or j in pred:
            continue
        succ[i] = j
        pred[j] = i

    stitched_root = {}

    # Find "root" tracklets (no predecessor), then walk successor chains to label stitched IDs.
    for idx in range(len(tracklets)):
        if idx in pred:
            continue
        root = tracklets[idx].orig_id
        cur = idx
        while True:
            stitched_root[cur] = root
            if cur not in succ:
                break
            cur = succ[cur]

    # Anything not reached in a chain becomes its own stitched root (no valid links).
    for idx in range(len(tracklets)):
        if idx not in stitched_root:
            stitched_root[idx] = tracklets[idx].orig_id

    return {tracklets[i].orig_id: stitched_root[i] for i in range(len(tracklets))}



# Main stitching function

# -----------------------------------------------------------------------------
# Stitch wide CSV into a stitched long CSV
#
# This is the main "offline linking" step. We take the wide tracker export,
# convert to long, link tracklets across gaps (max_gap ~ lost_track_buffer),
# and write a stitched long CSV with a stitched_id column.
#
# Notebook usage:
#   stitch_wide_csv_to_long(
#       input_csv=os.path.join(OUTPUT_PATH, "tracks_wide_format.csv"),
#       output_stitched_long=os.path.join(OUTPUT_PATH, "tracks_xy_stitched_long.csv"),
#       max_gap=lost_track_buffer,
#   )
# -----------------------------------------------------------------------------

def stitch_wide_csv_to_long(
    input_csv: str,
    output_stitched_long: str,
    max_gap: int,
    gap_penalty: float = 0.05,
    max_cost_quantile: float = 0.995,
    frame_col: Optional[str] = None,
) -> dict:
    """
    Stitch fragmented tracklets in a wide CSV using motion-consistent assignment.
    """
    df_wide = pd.read_csv(input_csv)
    long_df = wide_to_long(df_wide, frame_col=frame_col)

    tracklets = build_tracklets(long_df)
    stats = estimate_step_scale(long_df)
    sigma_step = stats["sigma_step"]

    # We scan candidate link costs first to set a reasonable cutoff (data-driven).
    tmp_costs = []
    for ti in tracklets:
        for tj in tracklets:
            if ti.end_frame >= tj.start_frame:
                continue
            gap = tj.start_frame - ti.end_frame
            if 1 <= gap <= max_gap:
                tmp_costs.append(
                    link_cost(ti.end_xy, tj.start_xy, gap, sigma_step, gap_penalty)
                )

    if len(tmp_costs) == 0:
        max_cost = 0.0
    else:
        # Keep only the cheaper X% of candidate links to avoid over-stitching.
        # (max_cost_quantile close to 1.0 is permissive, lower is stricter.)
        max_cost = float(np.quantile(tmp_costs, max_cost_quantile))

    C = build_cost_matrix(tracklets, max_gap, sigma_step, gap_penalty, max_cost)

    # Solve a 1-to-1 linking between tracklets under the pruned cost matrix.
    matches = solve_assignment(C)
    orig_to_stitched = build_orig_to_stitched(tracklets, matches)

    stitched_long = long_df.copy()

    # Rewrite orig_id into stitched_id (each chain collapses to its root id).
    stitched_long["stitched_id"] = stitched_long["orig_id"].map(
        orig_to_stitched
    ).fillna(stitched_long["orig_id"])

    stitched_long.sort_values(["stitched_id", "frame"], inplace=True)
    stitched_long.to_csv(output_stitched_long, index=False)

    return {
        "out_stitched_long": output_stitched_long,
        "n_points": int(len(long_df)),
        "n_orig_tracklets": int(len(tracklets)),
        "n_links": int(len(matches)),
        "sigma_step": float(sigma_step),
        "median_step": float(stats["median_step"]),
        "max_cost_threshold": float(max_cost),
    }


# Vial assignment and compact ID assignment

# -----------------------------------------------------------------------------
# Vial assignment + compact IDs (+ fps column)
#
# After stitching we:
# - assign each (x,y) to vial1..vial6 using the ROIs
# - drop points outside all vials
# - re-label IDs into compact_id in a left->right order (nice for plots/analysis)
# - optionally attach fps (constant column) so speed computations downstream are easy
#
# Notebook usage:
#   cap = cv2.VideoCapture(PATH_TO_VID)
#   fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0); cap.release()
#   fps = fps if fps > 0 else 30.0
#
#   assign_vials_and_compact_ids(
#       stitched_csv=os.path.join(OUTPUT_PATH, "tracks_xy_stitched_long.csv"),
#       roi_json=os.path.join(OUTPUT_PATH, "vial_rois.json"),
#       out_csv=os.path.join(OUTPUT_PATH, "compact_tracks.csv"),
#       fps=fps,
#   )
# -----------------------------------------------------------------------------

def assign_compact_ids_left_to_right(df: pd.DataFrame,
                                     id_col: str = "stitched_id") -> pd.DataFrame:
    """
    Assign compact IDs based on left→right median x ordering.
    """
    df = df.copy()
    x_rep = df.groupby(id_col)["x"].median().sort_values()
    mapping = {sid: i + 1 for i, sid in enumerate(x_rep.index)}
    df["compact_id"] = df[id_col].map(mapping).astype(int)
    return df


# Vial assignment and compact ID assignment
def assign_vials_and_compact_ids(
    stitched_csv: str,
    roi_json: str,
    out_csv: str,
    invert_y: bool = False,
    video_h: Optional[int] = None,
    fps: Optional[float] = None,   # adding fps to help with classification
):
    """
    Assign vial IDs using rectangular ROIs, then compact IDs within each vial.
    """
    with open(roi_json, "r") as f:
        vial_rois = {k: tuple(map(int, v)) for k, v in json.load(f).items()}

    def assign_vial(x, y):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return vid
        return None

    df = pd.read_csv(stitched_csv)
    df["frame"] = df["frame"].astype(int)

    y_use = (video_h - 1 - df["y"]) if invert_y else df["y"]
    df["vial_id"] = [assign_vial(x, y) for x, y in zip(df["x"], y_use)]
    df = df[df["vial_id"].notna()].copy()

    df["compact_id"] = -1
    offset = 0
    for vial, g in df.groupby("vial_id", sort=True):  # vial1, vial2, ...
        x_rep = g.groupby("stitched_id")["x"].median().sort_values()
        mapping = {sid: offset + i + 1 for i, sid in enumerate(x_rep.index)}
        df.loc[g.index, "compact_id"] = g["stitched_id"].map(mapping).astype(int)
        offset += len(x_rep)

    if fps is not None:
        df["fps"] = float(fps)

    df.to_csv(out_csv, index=False)
    return df





# Render vial overlay video

# -----------------------------------------------------------------------------
# Overlay video 
#
# This is allows us to get our final video rendering, useful to see how accurate the tracking is
# visually. 
#
# Notebook usage:
#   render_vial_overlay_video(
#       video_path=PATH_TO_VID,
#       csv_path=os.path.join(OUTPUT_PATH, "compact_tracks.csv"),
#       out_mp4=os.path.join(OUTPUT_PATH, "overlay_vials_shaded.mp4"),
#   )
# -----------------------------------------------------------------------------

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
    Render an overlay video where flies are colored by vial and shaded
    by compact_id within each vial.

    Expects CSV columns:
        frame, x, y, vial_id, compact_id

    Color scheme (HSV hue per vial):
        vial1: blue
        vial2: green
        vial3: yellow
        vial4: orange
        vial5: pink/magenta
        vial6: purple
    """


    # ---- Load detections
    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(int)
    df["compact_id"] = df["compact_id"].astype(int)
    df["vial_id"] = df["vial_id"].astype(str)

    by_frame = {
        int(f): g[["x", "y", "vial_id", "compact_id"]].to_numpy()
        for f, g in df.groupby("frame")
    }

    max_in_vial = df.groupby("vial_id")["compact_id"].max().to_dict()

    VIAL_HUE = {
        "vial1": 120,  # blue
        "vial2": 60,   # green
        "vial3": 30,   # yellow
        "vial4": 15,   # orange
        "vial5": 165,  # pink
        "vial6": 135,  # purple
    }

    def color_for(vial_id: str, cid: int) -> tuple:
        hue = int(VIAL_HUE.get(vial_id, 0))
        m = int(max_in_vial.get(vial_id, cid))
        if m <= 1:
            v = 235
        else:
            t = (cid - 1) / (m - 1)
            v = int(120 + t * (255 - 120))
        s = 240
        hsv = np.uint8([[[hue, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def put_text_outlined(img, text, org):
        cv2.putText(
            img, text, org,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), outline_thick, cv2.LINE_AA
        )
        cv2.putText(
            img, text, org,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), text_thick, cv2.LINE_AA
        )

    # ---- Video IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end == -1 or end >= n_frames:
        end = n_frames - 1

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    # ---- Render loop
    for frame_idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        f = int(frame_idx + frame_offset)
        dets = by_frame.get(f)
        if dets is not None:
            for x, y, vial_id, cid in dets:
                xi = int(round(float(x)))
                yi_raw = float(y)
                yi = int(round((h - 1 - yi_raw) if invert_y else yi_raw))

                if 0 <= xi < w and 0 <= yi < h:
                    cid = int(cid)
                    vial_id = str(vial_id)
                    cv2.circle(
                        frame_bgr,
                        (xi, yi),
                        radius,
                        color_for(vial_id, cid),
                        -1,
                    )
                    if show_ids:
                        put_text_outlined(
                            frame_bgr,
                            str(cid),
                            (xi + 8, yi - 8),
                        )

        writer.write(frame_bgr)

    cap.release()
    writer.release()


################################################################################################################################
################################################################################################################################
#################################### CLASSIFICATION UTILS ######################################################################


# ============================================================
# Genotype mapping
# ============================================================

def map_vial_to_genotype(df_path: str) -> pd.DataFrame:
    filename = os.path.basename(df_path)
    parts = filename.split("_")
    assert parts[2] == "hTDP43", "Unexpected filename format"

    genotypes = parts[3].split("-")
    vial_to_genotype = {f"vial{i+1}": genotypes[i] for i in range(len(genotypes))}

    df = pd.read_csv(df_path)
    df["genotype"] = df["vial_id"].map(vial_to_genotype)
    return df


# ============================================================
# Frame-level kinematics
# ============================================================

def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["compact_id", "frame"]).copy()
    fps = float(df.fps.iloc[0])

    df["dx"] = df.groupby("compact_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("compact_id")["y"].diff().fillna(0)
    df["dt"] = df.groupby("compact_id")["frame"].diff().fillna(1) / fps

    df["step_distance"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["velocity"] = df["step_distance"] / df["dt"]
    df["acceleration"] = (
        df.groupby("compact_id")["velocity"].diff().fillna(0) / df["dt"]
    )

    df["distance_traveled"] = df.groupby("compact_id")["step_distance"].cumsum()
    df["heading"] = np.arctan2(df["dy"], df["dx"])
    dtheta = df.groupby("compact_id")["heading"].diff().fillna(0)
    df["turning_angle"] = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    df["angular_velocity"] = df["turning_angle"] / df["dt"]

    return df


# ============================================================
# Trajectory geometry
# ============================================================

def add_area_covered(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for cid, g in df.groupby("compact_id"):
        pts = g[["x", "y"]].values
        if len(pts) < 3:
            area = 0.0
        else:
            try:
                area = ConvexHull(pts).volume
            except Exception:
                area = (g["x"].max() - g["x"].min()) * (g["y"].max() - g["y"].min())
        records.append((cid, area))

    area_df = pd.DataFrame(records, columns=["compact_id", "area_covered"])
    return df.merge(area_df, on="compact_id", how="left")


def add_path_tortuosity(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for cid, g in df.groupby("compact_id"):
        total = g["step_distance"].sum()
        net = np.sqrt(
            (g["x"].iloc[-1] - g["x"].iloc[0])**2 +
            (g["y"].iloc[-1] - g["y"].iloc[0])**2
        )
        value = total / net if net > 0 else np.nan
        records.append((cid, value))

    tort_df = pd.DataFrame(records, columns=["compact_id", "tortuosity"])
    return df.merge(tort_df, on="compact_id", how="left")


# ============================================================
# Feature extraction
# ============================================================

def extract_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_kinematics(df)
    df = add_area_covered(df)
    df = add_path_tortuosity(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.reset_index(drop=True)


def aggregate_per_fly_features(df: pd.DataFrame, pause_threshold: float = 1.0) -> pd.DataFrame:
    grouped = df.groupby("compact_id")

    return grouped.apply(
        lambda g: pd.Series({
            "mean_velocity": g["velocity"].mean(),
            "median_velocity": g["velocity"].median(),
            "std_velocity": g["velocity"].std(),
            "pause_fraction": (g["velocity"] < pause_threshold).mean(),
            "mean_abs_turning_angle": g["turning_angle"].abs().mean(),
            "mean_abs_angular_velocity": g["angular_velocity"].abs().mean(),
            "total_distance_traveled": g["distance_traveled"].iloc[-1],
            "tortuosity": g["tortuosity"].iloc[0],
            "area_covered": g["area_covered"].iloc[0],
        })
    ).reset_index()


# ============================================================
# Classification utilities
# ============================================================

def make_classifier(model_name: str):
    if model_name == "lda":
        return LinearDiscriminantAnalysis()
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000)
    if model_name == "svc":
        return SVC(kernel="linear")
    raise ValueError("model_name must be lda, logistic, or svc")


def prepare_xy(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).drop(columns=["compact_id"], errors="ignore")
    return X.values, X.columns.tolist()


def prepare_target(df: pd.DataFrame, mode: str = "multiclass"):
    if mode == "multiclass":
        return df["genotype"].values
    if mode == "binary":
        return np.where(df["genotype"] == "WT", "WT", "Mutant")
    raise ValueError("mode must be multiclass or binary")


# ============================================================
# Plotly-based evaluation
# ============================================================

def run_cross_validation(model, model_name, X, y, classification_mode, cv=5, outdir="report_figures"):
    scores = cross_val_score(model, X, y, cv=cv)

    fig = go.Figure()
    fig.add_bar(x=list(range(1, cv + 1)), y=scores)
    fig.add_hline(y=scores.mean(), line_dash="dash")

    fig.update_layout(
        title=f"{classification_mode.upper()} Cross-validation accuracy (mean={scores.mean():.3f})",
        xaxis_title="CV fold",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1]
    )

    save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}")
    
    return scores


def plot_feature_importance(model, X, y, feature_names, model_name, classification_mode):
    model.fit(X, y)

    if model_name == "logistic":
        values = np.mean(np.abs(model.named_steps["clf"].coef_), axis=0)
        xlabel = "Mean |coefficient|"
    elif model_name == "lda":
        values = np.mean(np.abs(model.named_steps["clf"].scalings_), axis=1)
        xlabel = "Mean |loading|"
    elif model_name == "svc":
        values = np.abs(model.named_steps["clf"].coef_).ravel()
        xlabel = "|weight|"
    else:
        return

    idx = np.argsort(values)

    fig = go.Figure()
    fig.add_bar(
        x=values[idx],
        y=[feature_names[i] for i in idx],
        orientation="h"
    )

    fig.update_layout(
        title=f"{classification_mode.upper()} Feature importance ({model_name.upper()})",
        xaxis_title=xlabel,
        yaxis_title="Feature"
    )

    return fig


def run_classifier(
    df: pd.DataFrame,
    outdir="report_figures", 
    model_name: str = "lda",
    classification_mode: str = "multiclass",
    cv: int = 5,
    plot_importance: bool = True
):
    X, feature_names = prepare_xy(df)
    y = prepare_target(df, classification_mode)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_classifier(model_name))
    ])

    run_cross_validation(pipeline, model_name, X, y, cv=cv, classification_mode = classification_mode)

    if plot_importance:
        fig = plot_feature_importance(
            pipeline,
            X,
            y,
            feature_names,
            model_name, 
            classification_mode
        )
        save_plotly_figure(fig, outdir, f"{model_name}_{classification_mode}")


# ============================================================
# Statistical plotting (already Plotly)
# ============================================================

def cliffs_delta(x, y):
    return (np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (len(x) * len(y))


def save_plotly_figure(fig, outdir, name, show=True):
    os.makedirs(outdir, exist_ok=True)
    fig.write_html(os.path.join(outdir, f"{name}.html"))
    fig.write_image(os.path.join(outdir, f"{name}.png"), width=1200, height=800, scale=2)
    if show:
        fig.show()


def plot_by_genotype(df, features, feature_titles, hover_data, outdir="report_figures"):
    for feat in features:
        groups = [g[feat].values for _, g in df.groupby("genotype")]
        _, p_kw = kruskal(*groups)

        fig = px.box(
            df, x="genotype", y=feat, color="genotype",
            points="all", hover_data=hover_data,
            title=f"{feature_titles[feat]} (Kruskal–Wallis p={p_kw:.3g})"
        )

        fig.update_traces(jitter=0.35, marker=dict(size=9, opacity=0.8))
        fig.update_layout(showlegend=False)

        save_plotly_figure(fig, outdir, f"{feat}_by_genotype")


def plot_wt_vs_mutant(df, features, feature_titles, hover_data, outdir="report_figures"):
    df = df.copy()
    df["WT_vs_mutant"] = np.where(df["genotype"] == "WT", "WT", "Mutant")

    for feat in features:
        wt = df[df["WT_vs_mutant"] == "WT"][feat]
        mut = df[df["WT_vs_mutant"] == "Mutant"][feat]

        _, p_u = mannwhitneyu(wt, mut, alternative="two-sided")
        delta = cliffs_delta(wt.values, mut.values)

        fig = px.box(
            df, x="WT_vs_mutant", y=feat, color="WT_vs_mutant",
            points="all", hover_data=hover_data,
            title=f"{feature_titles[feat]} — WT vs Mutant "
                  f"(MWU p={p_u:.3g}, Cliff’s δ={delta:.2f})"
        )

        fig.update_traces(jitter=0.35, marker=dict(size=9, opacity=0.8))
        fig.update_layout(showlegend=False)

        save_plotly_figure(fig, outdir, f"{feat}_WT_vs_mutant")






