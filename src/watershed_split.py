"""
src/watershed_split.py

Splits RF-DETR bboxes that contain multiple touching flies. Operates on the
detection dict produced by src/tracking.py before it reaches OC-SORT.

Bbox is "suspicious" when its area > median + k * MAD across all detections
in the run. Each suspicious bbox is cropped, Otsu-thresholded, distance-
transformed, seeded by local maxima, and split by skimage's watershed.

Rejects the split if watershed returns one region (no split happened) or if
any sub-region falls below min_region_area_fraction * median_area.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _area(box: np.ndarray) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _area_stats(det_by_frame: dict[int, np.ndarray]) -> tuple[float, float]:
    """Median area and MAD across every bbox in det_by_frame."""
    areas = []
    for arr in det_by_frame.values():
        if len(arr) == 0:
            continue
        w = arr[:, 2] - arr[:, 0]
        h = arr[:, 3] - arr[:, 1]
        areas.append(w * h)
    if not areas:
        return 0.0, 0.0
    flat = np.concatenate(areas).astype(np.float64)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return med, mad


def _split_one_bbox(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    median_area: float,
    max_flies: int,
    min_distance: int,
    min_region_area: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, tuple[int, int] | None]:
    """
    Try to split one bbox via watershed.

    Returns (sub_boxes, mask, peaks, crop_origin):
      sub_boxes  — (N, 4) array of new bboxes in original-frame coordinates,
                   or None if the split was rejected.
      mask       — binary mask of the crop (bbox-local), for debug rendering.
      peaks      — (M, 2) array of peak (y, x) in bbox-local coords, or None.
      crop_origin — (x1, y1) frame coords of the clipped bbox top-left,
                    needed to map mask/peaks back onto the full frame.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None, None, None

    origin = (x1, y1)
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Flies are dark on light, so invert: flies become white foreground.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    if mask.sum() == 0:
        return None, mask, None, origin

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = peak_local_max(
        dist,
        min_distance=max(1, int(round(min_distance))),
        num_peaks=max_flies,
        labels=mask.astype(bool),
    )
    if len(peaks) < 2:
        return None, mask, peaks if len(peaks) else None, origin

    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (py, px) in enumerate(peaks, start=1):
        markers[py, px] = i
    labels = watershed(-dist, markers, mask=mask.astype(bool))

    sub_boxes = []
    for lbl in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lbl)
        if len(xs) == 0:
            continue
        if len(xs) < min_region_area:
            return None, mask, peaks, origin  # degenerate split, reject
        sub_boxes.append([
            x1 + xs.min(),
            y1 + ys.min(),
            x1 + xs.max() + 1,
            y1 + ys.max() + 1,
        ])

    if len(sub_boxes) < 2:
        return None, mask, peaks, origin

    return np.asarray(sub_boxes, dtype=np.float32), mask, peaks, origin


def _draw_debug_png(
    out_path: str,
    frame_bgr: np.ndarray,
    parent_bbox: tuple[int, int, int, int],
    sub_boxes: list[tuple[int, int, int, int]] | None,
    mask: np.ndarray | None,
    mask_origin: tuple[int, int] | None,
    peaks: np.ndarray | None,
    title: str,
    all_detections: np.ndarray | None = None,
    context_pad: int = 30,
    upscale: int = 4,
) -> None:
    """Side-by-side debug: parent bbox | watershed result, both with context.

    parent_bbox / sub_boxes are in FRAME coordinates. mask is bbox-sized;
    mask_origin is the top-left of the (clipped) parent bbox in frame coords.
    peaks are (y, x) in mask-local coords.
    """
    H, W = frame_bgr.shape[:2]
    px1, py1, px2, py2 = parent_bbox
    cx1 = max(0, px1 - context_pad); cy1 = max(0, py1 - context_pad)
    cx2 = min(W, px2 + context_pad); cy2 = min(H, py2 + context_pad)
    crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    # Draw every detection in the frame as thin red boxes (skipping the parent,
    # which gets its own highlight). Same on both panels so the context is
    # identical and the only difference is the watershed result.
    def _draw_context_dets(img: np.ndarray) -> None:
        if all_detections is None or len(all_detections) == 0:
            return
        for det in all_detections:
            dx1, dy1, dx2, dy2 = (int(round(v)) for v in det[:4])
            if (dx1, dy1, dx2, dy2) == (px1, py1, px2, py2):
                continue  # parent gets highlighted separately
            if dx2 < cx1 or dx1 > cx2 or dy2 < cy1 or dy1 > cy2:
                continue  # outside context window
            cv2.rectangle(img, (dx1 - cx1, dy1 - cy1), (dx2 - cx1, dy2 - cy1),
                          (40, 40, 230), 1)

    # LEFT: context detections + parent bbox highlighted in orange (thicker).
    left = crop.copy()
    _draw_context_dets(left)
    cv2.rectangle(left, (px1 - cx1, py1 - cy1), (px2 - cx1, py2 - cy1),
                  (0, 165, 255), 2)

    # RIGHT: context detections + mask + split bboxes (cyan) + peaks (yellow).
    right = crop.copy()
    _draw_context_dets(right)
    if mask is not None and mask_origin is not None:
        mx, my = mask_origin
        mh, mw = mask.shape
        # paste mask into a frame-sized canvas, then crop to context.
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[my:my + mh, mx:mx + mw] = mask
        ctx_mask = full_mask[cy1:cy2, cx1:cx2]
        overlay = right.copy()
        overlay[ctx_mask > 0] = (60, 200, 60)
        right = cv2.addWeighted(overlay, 0.20, right, 0.80, 0)

    if sub_boxes:
        for sx1, sy1, sx2, sy2 in sub_boxes:
            cv2.rectangle(right, (sx1 - cx1, sy1 - cy1),
                          (sx2 - cx1, sy2 - cy1), (255, 255, 0), 2)

    if peaks is not None and mask_origin is not None:
        mx, my = mask_origin
        for py, px in peaks:
            cv2.circle(right, (int(mx + px - cx1), int(my + py - cy1)),
                       2, (0, 255, 255), -1, cv2.LINE_AA)

    # Upscale (nearest, keeps pixels crisp at small sizes).
    h, w = left.shape[:2]
    left  = cv2.resize(left,  (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(right, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)

    H2, W2 = left.shape[:2]
    title_h = 32
    gap = 6
    font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (text_w, _), _ = cv2.getTextSize(title, font, fscale, fthick)
    images_w = 2 * W2 + gap
    canvas_w = max(images_w, text_w + 16)

    canvas = np.full((H2 + title_h, canvas_w, 3), 30, dtype=np.uint8)
    x_off = (canvas_w - images_w) // 2
    canvas[title_h:, x_off:x_off + W2] = left
    canvas[title_h:, x_off + W2 + gap:x_off + 2 * W2 + gap] = right

    cv2.putText(canvas, title, (8, 22), font, fscale,
                (230, 230, 230), fthick, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)


def apply_watershed_splits(
    det_by_frame: dict[int, np.ndarray],
    video_path: str,
    cfg: dict,
    debug_dir: str | None = None,
) -> dict[int, np.ndarray]:
    """
    Split suspicious bboxes in det_by_frame using marker-controlled watershed.

    Parameters
    ----------
    det_by_frame : {frame_idx: (N, 5) array of [x1, y1, x2, y2, conf]}
    cfg : dict with keys
        enabled, area_outlier_k, max_flies_per_blob, min_distance_factor,
        min_region_area_fraction, debug, debug_max_images
    debug_dir : where to write PNGs when cfg['debug'] is True

    Returns
    -------
    Updated det_by_frame (new dict; input not mutated). Split sub-bboxes
    inherit the parent's confidence.
    """
    if not cfg.get("enabled", False):
        return det_by_frame

    median_area, mad = _area_stats(det_by_frame)
    if median_area <= 0 or mad <= 0:
        return det_by_frame

    k = float(cfg.get("area_outlier_k", 3.0))
    max_flies = int(cfg.get("max_flies_per_blob", 3))
    min_distance = float(cfg.get("min_distance_factor", 0.5)) * float(np.sqrt(median_area))
    min_region_area = float(cfg.get("min_region_area_fraction", 0.2)) * median_area
    area_threshold = median_area + k * mad

    debug = bool(cfg.get("debug", False)) and debug_dir is not None
    debug_cap = int(cfg.get("debug_max_images", 50))
    n_success = 0
    n_reject_written = 0

    # Find which frames have at least one suspicious bbox; only open those.
    flagged_per_frame: dict[int, list[int]] = {}
    for f, arr in det_by_frame.items():
        if len(arr) == 0:
            continue
        areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
        idxs = np.where(areas > area_threshold)[0].tolist()
        if idxs:
            flagged_per_frame[f] = idxs

    if not flagged_per_frame:
        return det_by_frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return det_by_frame

    out = {f: arr.copy() for f, arr in det_by_frame.items()}

    print(f"Watershed: median_area={median_area:.0f} px², MAD={mad:.0f}, "
          f"threshold={area_threshold:.0f} px², "
          f"flagged {sum(len(v) for v in flagged_per_frame.values())} bbox(es) "
          f"across {len(flagged_per_frame)} frame(s).")

    for f in sorted(flagged_per_frame.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        arr = out[f]
        # Snapshot detections for this frame BEFORE we mutate `arr`, so the
        # debug PNG always shows the pre-split context regardless of how many
        # bboxes we end up splitting.
        all_dets_for_frame = arr.copy()
        # walk indices in reverse so deletion doesn't shift later indices
        flagged_sorted = sorted(flagged_per_frame[f], reverse=True)
        for i in flagged_sorted:
            parent = arr[i]
            sub, mask, peaks, crop_origin = _split_one_bbox(
                frame_bgr=frame_bgr,
                bbox=parent,
                median_area=median_area,
                max_flies=max_flies,
                min_distance=min_distance,
                min_region_area=min_region_area,
            )

            accepted = sub is not None
            if accepted:
                # replace parent row with split rows
                new_rows = np.empty((len(sub), 5), dtype=arr.dtype)
                new_rows[:, :4] = sub
                new_rows[:, 4] = parent[4]  # inherit confidence
                arr = np.delete(arr, i, axis=0)
                arr = np.vstack([arr, new_rows]) if len(arr) else new_rows

            if debug:
                budget_ok = (
                    accepted
                    or (n_success + n_reject_written < debug_cap)
                )
                if accepted:
                    n_success += 1
                    title = (
                        f"frame={f} area={_area(parent):.0f}px² "
                        f"({_area(parent)/median_area:.1f}x median) "
                        f"-> ACCEPTED ({len(sub)} regions)"
                    )
                elif budget_ok:
                    n_reject_written += 1
                    title = (
                        f"frame={f} area={_area(parent):.0f}px² "
                        f"({_area(parent)/median_area:.1f}x median) -> REJECTED"
                    )
                else:
                    title = None

                if title is not None:
                    x1, y1, x2, y2 = (int(round(v)) for v in parent[:4])
                    parent_frame = (max(0, x1), max(0, y1),
                                    min(frame_bgr.shape[1], x2),
                                    min(frame_bgr.shape[0], y2))
                    sub_frame = [tuple(int(v) for v in s) for s in sub] if accepted else None
                    png_name = f"frame_{f:06d}_bbox_{i:02d}.png"
                    _draw_debug_png(
                        out_path=os.path.join(debug_dir, png_name),
                        frame_bgr=frame_bgr,
                        parent_bbox=parent_frame,
                        sub_boxes=sub_frame,
                        mask=mask,
                        mask_origin=crop_origin,
                        peaks=peaks,
                        title=title,
                        all_detections=all_dets_for_frame,
                    )

        out[f] = arr

    cap.release()

    if debug:
        print(f"Watershed debug: wrote {n_success} success + "
              f"{n_reject_written} rejection PNG(s) to {debug_dir}")

    return out
