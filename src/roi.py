"""
src/roi.py

Interactive vial ROI drawing + assignment of compact (sequential) fly IDs.

Workflow
--------
1. draw_and_save_vial_rois()  — one-time manual annotation per experiment setup
2. assign_vials_and_compact_ids()  — assign vial labels + compact IDs to each point
"""

import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def draw_and_save_vial_rois(
    video_path: str,
    roi_json_path: str,
    frame_idx: int = 0,
    n_vials: int = 6,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Interactive utility to manually draw rectangular ROIs for fly vials.

    Controls:
      - Mouse drag: draw ROI
      - u: undo last ROI
      - r: reset all ROIs
      - q: finish (only if exactly n_vials ROIs are drawn)
      - ESC: cancel

    Parameters
    ----------
    video_path : str
        Path to the experiment video (same file used for tracking).
    roi_json_path : str
        Where the ROIs will be saved as JSON.
    frame_idx : int
        Reference frame for drawing (default 0).
    n_vials : int
        Number of vials expected (default 6).

    Returns
    -------
    Dict mapping vial IDs to (x0, y0, x1, y1), sorted left -> right.
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
                img, f"{k}", (x0 + 5, y0 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
            )
        cv2.putText(
            img, f"ROIs: {len(rois)}/{n_vials}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
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
    roi_dict = {f"vial{i}": tuple(r) for i, r in enumerate(rois_sorted, start=1)}

    os.makedirs(os.path.dirname(roi_json_path) or ".", exist_ok=True)
    with open(roi_json_path, "w") as f:
        json.dump({k: list(v) for k, v in roi_dict.items()}, f, indent=2)

    print("Saved ROIs to:", roi_json_path)
    return roi_dict


def assign_compact_ids_left_to_right(
    df: pd.DataFrame,
    id_col: str = "stitched_id",
) -> pd.DataFrame:
    """Assign compact IDs based on left->right median x ordering."""
    df = df.copy()
    x_rep = df.groupby(id_col)["x"].median().sort_values()
    mapping = {sid: i + 1 for i, sid in enumerate(x_rep.index)}
    df["compact_id"] = df[id_col].map(mapping).astype(int)
    return df


def assign_vials_and_compact_ids(
    stitched_csv: str,
    roi_json: str,
    out_csv: str,
    invert_y: bool = False,
    video_h: Optional[int] = None,
    fps: Optional[float] = None,
):
    """
    Assign vial IDs using rectangular ROIs, then compact IDs within each vial.

    Parameters
    ----------
    stitched_csv : str
        Long-format stitched CSV from stitch_per_vial().
        Must have columns: frame, orig_id, x, y, stitched_id.
    roi_json : str
        JSON file produced by draw_and_save_vial_rois().
    out_csv : str
        Output path for the compact_tracks CSV.
    invert_y : bool
        Flip y coordinates (needed if tracker and video have different origins).
    video_h : int, optional
        Video height in pixels (required when invert_y=True).
    fps : float, optional
        Frames per second — appended as a constant column for downstream speed calc.

    Returns
    -------
    pd.DataFrame  — the compact_tracks DataFrame (also saved to out_csv).
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
    for vial, g in df.groupby("vial_id", sort=True):
        x_rep = g.groupby("stitched_id")["x"].median().sort_values()
        mapping = {sid: offset + i + 1 for i, sid in enumerate(x_rep.index)}
        df.loc[g.index, "compact_id"] = g["stitched_id"].map(mapping).astype(int)
        offset += len(x_rep)

    if fps is not None:
        df["fps"] = float(fps)

    df.to_csv(out_csv, index=False)
    return df
