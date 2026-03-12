"""
src/preprocessing.py

Background subtraction + interactive ROI/range GUI.

Optional first step: makes flies pop against the background before tracking.
GUI controls:
  - Drag mouse to draw ROI
  - Trackbars for start/end frame and current preview frame
  - ENTER to accept, ESC to cancel
"""

import os
import cv2
import numpy as np
from pathlib import Path


# -------------------------------------------------------------------------
# Default tuning constants (override via config.yaml or function kwargs)
# -------------------------------------------------------------------------
BG_GAIN = 1.2          # higher = flies darker / more contrast
BG_WHITE_LEVEL = 245   # higher = brighter background
BG_DEADZONE = 0.0      # >0 suppresses tiny static mismatches
BG_CODEC = "mp4v"


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
            end = max(start + 1, min(end, n_frames))
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
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        if roi is None:
            cv2.putText(
                img, "Draw ROI with mouse drag",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )
        else:
            x, y, w, h = roi
            cv2.putText(
                img, f"ROI x={x} y={y} w={w} h={h}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
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
    bg_sample_stride: int = 1,
) -> str:
    """
    GUI-driven ROI/range selection + average-background subtraction.

    Output path behaviour:
      - If out_mp4 is None:  "<same folder>/<stem>_pp.<ext>"
      - Otherwise writes to the given path (parent dir created if needed).

    Returns the output mp4 path as a string.
    """
    video_path = str(video_path)
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

    # 1) Compute average background
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
            raise ValueError("ROI out of bounds during background computation.")
        acc += _bgr_to_gray_float32(roi_bgr).astype(np.float64)
        count += 1

    if count == 0:
        cap.release()
        raise RuntimeError("No frames available to compute the average background.")

    bg_gray = (acc / float(count)).astype(np.float32)

    # 2) Write background-subtracted video
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

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
        motion = np.maximum(bg_gray - gray, 0.0)
        if deadzone and deadzone > 0:
            motion = np.maximum(motion - float(deadzone), 0.0)
        vis = float(white_level) - motion * float(gain)
        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(vis_u8, cv2.COLOR_GRAY2BGR))

    cap.release()
    writer.release()
    print("Saved bgsub video:", out_mp4)
    print(f"Background computed from {count} frames (stride={bg_sample_stride}).")
    return out_mp4
