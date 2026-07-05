"""
src/tracking.py

Roboflow RF-DETR detection + OC-SORT multi-object tracking.
Writes a wide CSV: rows = frames, columns = track IDs, cells = "(x, y)".

If a detection cache CSV (det_log_csv) already exists on disk, RF-DETR is
skipped entirely and detections are loaded from the cache — useful for
re-running the tracker with different association parameters without paying
the API cost again.

Optional watershed splitting (watershed_cfg) runs after detection collection,
before the tracker sees anything: oversized bboxes that likely contain >1
fly are split via marker-controlled watershed. Post-split detections are
what gets written to the cache.
"""

import logging
import os
import cv2
import numpy as np
import pandas as pd
import supervision as sv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_xy(cell):
    """Parse one wide-CSV track cell of the form ``"(x, y)"`` into numbers.

    Track positions are stored as strings so that frames where a track is absent
    can be left blank; this converts a populated cell back into floats and
    tolerates blank/malformed cells by returning None instead of raising.

    Inputs
    ------
    cell : str | float | None
        One cell from an ``id{N}`` column. Normally the string ``"(x, y)"``;
        may be NaN / empty / non-string for frames where the track was absent.

    Outputs
    -------
    tuple[float, float] | None
        ``(x, y)`` when the cell holds a valid ``"(x, y)"`` string, else None
        (blank, non-string, or unparseable).
    """
    if not isinstance(cell, str) or not cell.strip():
        return None
    try:
        s = cell.strip().lstrip("(").rstrip(")")
        x, y = s.split(",")
        return float(x), float(y)
    except Exception:
        return None


def _run_tracker_pass(det_by_frame, n_frames, img_h, img_w, vial_rois,
                      tracker_kwargs, max_frames=None):
    """Run one complete OC-SORT pass over pre-computed detections.

    This is the low-level single-pass engine shared by the normal (one-pass) and
    ghost (two-pass) flows. It constructs a fresh tracker, feeds detections frame
    by frame — optionally dropping detections whose centre falls outside every
    vial ROI — and records each surviving track's centroid into a wide table.

    Inputs
    ------
    det_by_frame : dict[int, np.ndarray]
        Frame index -> (N, 5) float32 array of [x1, y1, x2, y2, conf]. Frames
        with no entry are treated as empty.
    n_frames : int
        Number of frames to iterate (0 .. n_frames - 1).
    img_h : int
        Frame height in pixels (passed to OC-SORT as the image size).
    img_w : int
        Frame width in pixels.
    vial_rois : dict[str, tuple[int, int, int, int]] | None
        vial id -> (x0, y0, x1, y1). When given, a detection is dropped if its
        centre lies outside every ROI. None disables the filter.
    tracker_kwargs : dict
        Keyword arguments forwarded verbatim to the ``OCSort`` constructor
        (det_thresh, max_age, min_hits, ...).
    max_frames : int | None, default None
        Optional hard cap on frames processed (debugging). None processes all
        ``n_frames``.

    Outputs
    -------
    tuple[pandas.DataFrame, OCSort]
        df_wide : column ``"frame"`` plus one ``"id{N}"`` column per track id
            seen; cells are ``"(x, y)"`` strings or NaN.
        tracker : the ``OCSort`` instance after the pass (carries detection_log,
            suppressed_tracks, and per-track observation_log).
    """
    from .ocsort import OCSort
    tracker = OCSort(**tracker_kwargs)
    rows = []
    all_track_ids = set()

    for frame_idx in range(n_frames):
        if max_frames is not None and frame_idx >= max_frames:
            break

        det_array = det_by_frame.get(frame_idx, np.empty((0, 5), dtype=np.float32))

        if vial_rois is not None and len(det_array) > 0:
            cx = (det_array[:, 0] + det_array[:, 2]) / 2.0
            cy = (det_array[:, 1] + det_array[:, 3]) / 2.0
            in_vial = np.zeros(len(det_array), dtype=bool)
            for x0, y0, x1, y1 in vial_rois.values():
                in_vial |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
            det_array = det_array[in_vial]

        frame_row = {"frame": frame_idx}
        if len(det_array) > 0:
            tracks = tracker.update(det_array, [img_h, img_w], [img_h, img_w])
            if tracks is not None and len(tracks) > 0:
                if tracks.ndim == 1:
                    tracks = tracks[None, :]
                xyxy = tracks[:, :4]
                tids = tracks[:, 4].astype(int)
                cx_out = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                cy_out = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                for tid, x, y in zip(tids, cx_out, cy_out):
                    all_track_ids.add(int(tid))
                    frame_row[f"id{int(tid)}"] = f"({x:.2f}, {y:.2f})"
        else:
            tracker.update(np.empty((0, 5)), [img_h, img_w], [img_h, img_w])

        rows.append(frame_row)

    df = pd.DataFrame(rows)
    id_cols = [f"id{tid}" for tid in sorted(all_track_ids)]
    df = df.reindex(columns=["frame"] + id_cols)
    return df, tracker


def _find_occlusion_gaps(df_wide, vial_rois, vial_expected_counts, det_by_frame,
                         top_exit_px=2.0, occlusion_max_gap=90):
    """Classify each track's disappearance in Pass-1 as a top exit or an occlusion.

    After the first tracking pass some tracks stop early. This decides, per drop,
    whether the fly legitimately left the vial through the top edge or was hidden
    behind another fly (an occlusion). Only occlusions that leave a vial below its
    expected count are returned for ghost filling, so the second pass never
    invents a fly that actually exited.

    Inputs
    ------
    df_wide : pandas.DataFrame
        Pass-1 wide tracks (``"frame"`` plus ``"id{N}"`` columns of ``"(x, y)"``).
    vial_rois : dict[str, tuple[int, int, int, int]]
        vial id -> (x0, y0, x1, y1) bounding box.
    vial_expected_counts : dict[str, int]
        vial id -> expected number of flies; gates which count drops qualify.
    det_by_frame : dict[int, np.ndarray]
        Frame -> (N, 5) detections, used to estimate a median fly bbox size per
        vial (the ghost box size).
    top_exit_px : float, default 2.0
        A centroid within this many pixels of a vial's top edge counts as a top
        exit (no ghost fired for that track).
    occlusion_max_gap : int, default 90
        Maximum gap length in frames still treated as an occlusion; longer gaps
        are ignored.

    Outputs
    -------
    tuple[list[dict], list[dict], list[dict]]
        top_exit_events : tracks that reached the vial top edge; each
            ``{"track_id": int, "frame": int, "vial": str}``.
        top_reentry_events : tracks whose first appearance was at the top edge;
            same dict shape.
        occlusion_gaps : qualifying gaps to fill; each
            ``{"vial": str, "missing_track_id": int, "start_frame": int,
            "end_frame": int, "last_known_cx": float, "last_known_cy": float,
            "bbox_w": float, "bbox_h": float}``.
    """
    id_cols = [c for c in df_wide.columns if c.startswith("id")]
    frames_list = df_wide["frame"].tolist()

    def _vial_of(cx, cy):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return vid
        return None

    # per-track position sequence
    track_positions = {}
    for col in id_cols:
        tid = int(col[2:])
        for i, frame in enumerate(frames_list):
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is not None:
                track_positions.setdefault(tid, []).append((int(frame), pos[0], pos[1]))

    # top exit / reentry
    top_exit_events = []
    top_exit_tids = set()
    top_reentry_events = []

    for tid, positions in track_positions.items():
        for frame, cx, cy in positions:
            vid = _vial_of(cx, cy)
            if vid is None:
                continue
            _, y0, _, _ = vial_rois[vid]
            if cy <= y0 + top_exit_px:
                top_exit_events.append({"track_id": tid, "frame": int(frame), "vial": vid})
                top_exit_tids.add(tid)
                break

        if positions:
            f0, cx0, cy0 = positions[0]
            vid = _vial_of(cx0, cy0)
            if vid is not None:
                _, y0, _, _ = vial_rois[vid]
                if cy0 <= y0 + top_exit_px:
                    top_reentry_events.append(
                        {"track_id": tid, "frame": int(f0), "vial": vid}
                    )

    # median bbox per vial from detection cache
    vial_median_bbox = {}
    for vid, (x0, y0, x1, y1) in vial_rois.items():
        widths, heights = [], []
        for dets in det_by_frame.values():
            if len(dets) == 0:
                continue
            cx_arr = (dets[:, 0] + dets[:, 2]) / 2.0
            cy_arr = (dets[:, 1] + dets[:, 3]) / 2.0
            mask = (cx_arr >= x0) & (cx_arr <= x1) & (cy_arr >= y0) & (cy_arr <= y1)
            if mask.any():
                widths.extend((dets[mask, 2] - dets[mask, 0]).tolist())
                heights.extend((dets[mask, 3] - dets[mask, 1]).tolist())
        vial_median_bbox[vid] = (
            float(np.median(widths)) if widths else 20.0,
            float(np.median(heights)) if heights else 20.0,
        )

    # per-vial per-frame active track set for count verification
    vial_frame_active = {}
    for col in id_cols:
        tid = int(col[2:])
        for i, frame in enumerate(frames_list):
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is None:
                continue
            vid = _vial_of(pos[0], pos[1])
            if vid is None:
                continue
            vial_frame_active.setdefault(vid, {}).setdefault(int(frame), set()).add(tid)

    # find occlusion gaps
    occlusion_gaps = []

    for tid, positions in track_positions.items():
        if tid in top_exit_tids or len(positions) < 2:
            continue

        for i in range(len(positions) - 1):
            f_before, cx_before, cy_before = positions[i]
            f_after  = positions[i + 1][0]
            gap_len  = f_after - f_before - 1

            if gap_len <= 0 or gap_len > occlusion_max_gap:
                continue

            vid = _vial_of(cx_before, cy_before)
            if vid is None:
                continue

            _, y0, _, _ = vial_rois[vid]
            if cy_before <= y0 + top_exit_px:
                continue  # track was near top before gap — treat as exit

            expected_n = vial_expected_counts.get(vid, 0)
            if expected_n <= 0:
                continue

            # Vial must have been full just before the gap
            count_before = len(vial_frame_active.get(vid, {}).get(f_before, set()))
            if count_before < expected_n:
                continue

            # Count must have dropped during the gap
            count_during = len(
                vial_frame_active.get(vid, {}).get(f_before + 1, set())
            )
            if count_during >= expected_n:
                continue

            bbox_w, bbox_h = vial_median_bbox.get(vid, (20.0, 20.0))
            occlusion_gaps.append({
                "vial":             vid,
                "missing_track_id": tid,
                "start_frame":      f_before,
                "end_frame":        f_after,
                "last_known_cx":    cx_before,
                "last_known_cy":    cy_before,
                "bbox_w":           bbox_w,
                "bbox_h":           bbox_h,
            })

    return top_exit_events, top_reentry_events, occlusion_gaps


def _inject_ghost_detections(det_by_frame, occlusion_gaps, df_wide, vial_rois,
                             offset_fraction=0.5, ghost_confidence=0.45):
    """Fill each occlusion gap with one synthetic ("ghost") detection per frame.

    Feeding a ghost detection through the occlusion lets OC-SORT carry the hidden
    fly's track across the gap instead of dropping it and later spawning a new id.
    Each ghost is placed just off the occluding fly's centroid, offset toward the
    missing fly's last known position (or at that last position when no occluder
    is visible), and clamped inside the vial.

    Inputs
    ------
    det_by_frame : dict[int, np.ndarray]
        Original detections, frame -> (N, 5). Not mutated; a shallow copy is
        edited (arrays copied on write).
    occlusion_gaps : list[dict]
        Gaps from ``_find_occlusion_gaps`` (keys: vial, missing_track_id,
        start_frame, end_frame, last_known_cx/cy, bbox_w/bbox_h).
    df_wide : pandas.DataFrame
        Pass-1 wide tracks, used to know where every other track sat each frame
        so the nearest visible track can be chosen as the occluder.
    vial_rois : dict[str, tuple[int, int, int, int]]
        vial id -> (x0, y0, x1, y1); used to clamp ghosts inside the vial.
    offset_fraction : float, default 0.5
        Ghost offset from the occluder centroid, in units of the vial's median
        bbox width.
    ghost_confidence : float, default 0.45
        Confidence assigned to each ghost detection (kept below the tracker's
        spawn threshold so ghosts update tracks but never spawn new ones).

    Outputs
    -------
    tuple[dict[int, np.ndarray], list[dict]]
        det_out : detections with ghosts appended (a new dict; the input is
            untouched).
        ghost_log : one entry per injected ghost:
            ``{"frame": int, "missing_track_id": int,
            "occluder_track_id": int | None, "ghost_cx": float,
            "ghost_cy": float, "vial": str}``.
    """
    det_out = dict(det_by_frame)  # shallow copy; arrays copied on write

    id_cols = [c for c in df_wide.columns if c.startswith("id")]
    frames_list = df_wide["frame"].tolist()

    def _vial_of(cx, cy):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return vid
        return None

    # frame → {tid: (cx, cy)} from Pass-1
    frame_track_pos = {}
    for i, frame in enumerate(frames_list):
        frame = int(frame)
        frame_track_pos[frame] = {}
        for col in id_cols:
            tid = int(col[2:])
            pos = _parse_xy(df_wide.iloc[i][col])
            if pos is not None:
                frame_track_pos[frame][tid] = pos

    ghost_log = []

    for gap in occlusion_gaps:
        vid         = gap["vial"]
        missing_tid = gap["missing_track_id"]
        start_frame = gap["start_frame"]
        end_frame   = gap["end_frame"]
        last_cx     = gap["last_known_cx"]
        last_cy     = gap["last_known_cy"]
        bbox_w      = gap["bbox_w"]
        bbox_h      = gap["bbox_h"]
        x0, y0, x1, y1 = vial_rois[vid]

        for frame in range(start_frame + 1, end_frame):
            candidates = {
                tid: pos
                for tid, pos in frame_track_pos.get(frame, {}).items()
                if tid != missing_tid and _vial_of(pos[0], pos[1]) == vid
            }

            if candidates:
                occluder_tid = min(
                    candidates,
                    key=lambda t: (candidates[t][0] - last_cx) ** 2
                                + (candidates[t][1] - last_cy) ** 2,
                )
                occ_cx, occ_cy = candidates[occluder_tid]
                dx   = last_cx - occ_cx
                dy   = last_cy - occ_cy
                dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
                ghost_cx = occ_cx + (dx / dist) * bbox_w * offset_fraction
                ghost_cy = occ_cy + (dy / dist) * bbox_w * offset_fraction
            else:
                occluder_tid = None
                ghost_cx, ghost_cy = last_cx, last_cy

            ghost_cx = float(np.clip(ghost_cx, x0 + bbox_w / 2, x1 - bbox_w / 2))
            ghost_cy = float(np.clip(ghost_cy, y0 + bbox_h / 2, y1 - bbox_h / 2))

            ghost_det = np.array([[
                ghost_cx - bbox_w / 2, ghost_cy - bbox_h / 2,
                ghost_cx + bbox_w / 2, ghost_cy + bbox_h / 2,
                ghost_confidence,
            ]], dtype=np.float32)

            if frame in det_out:
                det_out[frame] = np.vstack([det_out[frame], ghost_det])
            else:
                det_out[frame] = ghost_det

            ghost_log.append({
                "frame":             frame,
                "missing_track_id":  missing_tid,
                "occluder_track_id": occluder_tid,
                "ghost_cx":          ghost_cx,
                "ghost_cy":          ghost_cy,
                "vial":              vid,
            })

    return det_out, ghost_log


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def collect_detections(
    video_path: str,
    *,
    det_log_csv: str | None,
    api_key: str,
    model_id: str,
    inference_api_url: str,
    inference_mode: str,
    local_weights_path: str,
    local_resolution: int,
    local_num_classes: int,
    local_optimize_for_gpu: bool,
    repo_root: str,
    detection_confidence_rfdetr: float,
    max_frames: int | None,
) -> tuple[dict, int, int, int, float, bool]:
    """Return one detection array per frame, plus the video's geometry.

    Detections are RF-DETR boxes for each frame, kept as (x1, y1, x2, y2, conf).
    They come from either Roboflow hosted API or local weights (see inference.mode
    in config.yaml). If a detection cache exists at ``det_log_csv`` it is read
    back instead of calling the model, so association parameters can be re-tuned
    without paying for inference again. The video is opened either way because
    the tracker needs the frame height/width, and fps is reported so the caller
    can fall back to it when no explicit fps is given.

    Inputs
    ------
    video_path : str
        Path to the video to read. Opened even on a cache hit, to probe fps and
        frame size.
    det_log_csv : str | None (keyword-only)
        Path to a cached detections CSV. If the file exists, inference is skipped
        and detections are read from it; None or a missing file => run RF-DETR.
    api_key : str (keyword-only)
        Roboflow API key. Used only on a cache miss.
    model_id : str (keyword-only)
        Roboflow model id, ``"<workspace>/<version>"``.
    inference_api_url : str (keyword-only)
        Roboflow inference host URL (hosted mode only).
    inference_mode : str (keyword-only)
        ``hosted`` or ``local`` — where RF-DETR runs (config.inference.mode).
    local_weights_path : str (keyword-only)
        Path to weights.pt when inference_mode is ``local``.
    local_resolution : int (keyword-only)
        Model input resolution; must match the checkpoint.
    local_num_classes : int (keyword-only)
        Number of detection classes (1 for fly-only).
    local_optimize_for_gpu : bool (keyword-only)
        When True and CUDA is available, enable FP16 inference optimization.
    repo_root : str (keyword-only)
        Repo root for resolving relative local_weights_path.
    detection_confidence_rfdetr : float (keyword-only)
        RF-DETR confidence threshold applied at inference time.
    max_frames : int | None (keyword-only)
        Optional cap on the number of frames to infer; None = whole video.

    Outputs
    -------
    tuple[dict[int, np.ndarray], int, int, int, float, bool]
        det_by_frame : frame index -> (N, 5) float32 [x1, y1, x2, y2, conf].
        n_frames : int, number of frames represented.
        img_h : int, frame height in pixels.
        img_w : int, frame width in pixels.
        fps : float, frame rate reported by OpenCV (may be 0 if unknown).
        loaded_from_cache : bool, True if detections came from ``det_log_csv``,
            False if produced by RF-DETR.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    use_cache = det_log_csv is not None and os.path.exists(det_log_csv)
    det_by_frame: dict[int, np.ndarray] = {}

    if use_cache:
        print(f"Detection cache found — skipping RF-DETR: {det_log_csv}")
        det_cache = pd.read_csv(det_log_csv)
        det_by_frame = {
            int(frame_idx): grp[["x1", "y1", "x2", "y2", "conf"]].values.astype(np.float32)
            for frame_idx, grp in det_cache.groupby("frame")
        }
        n_frames = int(det_cache["frame"].max()) + 1 if len(det_cache) else 0
        cap.release()
    else:
        from pathlib import Path

        from src.inference_backends import create_detection_backend

        backend = create_detection_backend(
            inference_mode,
            repo_root=Path(repo_root),
            api_url=inference_api_url,
            api_key=api_key,
            model_id=model_id,
            threshold=detection_confidence_rfdetr,
            local_weights_path=local_weights_path,
            local_resolution=local_resolution,
            local_num_classes=local_num_classes,
            local_optimize_for_gpu=local_optimize_for_gpu,
        )
        print(f"RF-DETR inference: {inference_mode}")

        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            dets = backend.predict(frame)
            if len(dets) > 0:
                det_by_frame[frame_idx] = np.hstack(
                    [dets.xyxy, dets.confidence[:, None]]
                ).astype(np.float32)
            frame_idx += 1
        n_frames = frame_idx
        cap.release()

    return det_by_frame, n_frames, img_h, img_w, fps, use_cache


def split_merged_detections(det_by_frame: dict, video_path: str, watershed_cfg: dict,
                            det_log_csv: str | None) -> dict:
    """Split detection boxes that enclose more than one touching fly.

    RF-DETR occasionally returns a single box over two flies in contact. Boxes
    flagged as oversized (per ``watershed_cfg``) are re-segmented with
    marker-controlled watershed and replaced by one box per fly, so downstream
    counts and tracks are not silently merged. With ``watershed_cfg["debug"]``
    on, before/after crops are written next to the detection cache for review.

    Inputs
    ------
    det_by_frame : dict[int, np.ndarray]
        Detections per frame, frame -> (N, 5) [x1, y1, x2, y2, conf].
    video_path : str
        Path to the video; watershed reads the frame pixels to segment blobs.
    watershed_cfg : dict
        Watershed parameters (e.g. area_outlier_k, max_flies_per_blob,
        min_distance_factor, min_region_area_fraction, debug, debug_max_images).
    det_log_csv : str | None
        Used only to locate the debug-image folder (``watershed_debug`` written
        beside it) when ``watershed_cfg["debug"]`` is set; may be None.

    Outputs
    -------
    dict[int, np.ndarray]
        A new per-frame detection dict where oversized boxes have been replaced
        by one box per fly.
    """
    from .watershed_split import apply_watershed_splits
    debug_dir = None
    if watershed_cfg.get("debug", False) and det_log_csv is not None:
        debug_dir = os.path.join(os.path.dirname(det_log_csv) or ".", "watershed_debug")
    return apply_watershed_splits(
        det_by_frame=det_by_frame,
        video_path=video_path,
        cfg=watershed_cfg,
        debug_dir=debug_dir,
    )


def cache_detections(det_by_frame: dict, det_log_csv: str | None, *,
                     loaded_from_cache: bool, watershed_on: bool) -> None:
    """Persist detections to ``det_log_csv`` so later runs can skip inference.

    Written on a cache miss, or after watershed has changed the boxes. Skipped
    when the detections were themselves loaded from cache and watershed is off,
    since re-running with identical detections would only rewrite the same file.

    Inputs
    ------
    det_by_frame : dict[int, np.ndarray]
        Detections to write, frame -> (N, 5) [x1, y1, x2, y2, conf].
    det_log_csv : str | None
        Destination CSV path. None disables writing.
    loaded_from_cache : bool (keyword-only)
        Whether these detections were read from that cache earlier this run.
    watershed_on : bool (keyword-only)
        Whether watershed changed the detections this run.

    Outputs
    -------
    None
        Writes ``det_log_csv`` (columns frame, x1, y1, x2, y2, conf) as a side
        effect, or does nothing when ``det_log_csv`` is None or when
        (loaded_from_cache and not watershed_on).
    """
    if det_log_csv is None or (loaded_from_cache and not watershed_on):
        return
    rows = []
    for f in sorted(det_by_frame.keys()):
        for x1, y1, x2, y2, conf in det_by_frame[f]:
            rows.append({"frame": int(f),
                         "x1": float(x1), "y1": float(y1),
                         "x2": float(x2), "y2": float(y2),
                         "conf": float(conf)})
    pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf"]).to_csv(
        det_log_csv, index=False
    )
    print(f"Saved detection cache: {det_log_csv}  ({len(rows)} detections)")


def run_tracker(
    det_by_frame: dict, n_frames: int, img_h: int, img_w: int, vial_rois: dict | None,
    tracker_kwargs: dict, *, max_frames: int | None,
    ghost_detection_enabled: bool, vial_expected_counts: dict | None,
    ghost_offset_fraction: float, ghost_confidence: float,
    ghost_occlusion_max_gap: int, ghost_top_exit_px: float,
) -> tuple:
    """Associate the per-frame detections into tracks with OC-SORT.

    One pass produces the wide track table. When ghost detection is enabled and
    a vial ends the first pass with fewer tracks than its expected fly count, a
    second pass is run: occlusion gaps from pass one are located and a synthetic
    ("ghost") detection is injected through each gap, so a fly that disappeared
    behind another is carried across the occlusion instead of being dropped and
    re-spawned as a new id.

    Inputs
    ------
    det_by_frame : dict[int, np.ndarray]
        Detections per frame, frame -> (N, 5) [x1, y1, x2, y2, conf].
    n_frames : int
        Number of frames to process.
    img_h : int
        Frame height in pixels.
    img_w : int
        Frame width in pixels.
    vial_rois : dict[str, tuple[int, int, int, int]] | None
        vial id -> (x0, y0, x1, y1). Filters detections and is required for the
        ghost pass; None disables both the ROI filter and ghost detection.
    tracker_kwargs : dict
        Keyword arguments for the ``OCSort`` constructor.
    max_frames : int | None (keyword-only)
        Optional cap on frames processed (debugging); None = all frames.
    ghost_detection_enabled : bool (keyword-only)
        Master switch for the ghost second pass.
    vial_expected_counts : dict[str, int] | None (keyword-only)
        Per-vial expected fly counts; the ghost pass runs only if at least one
        is > 0.
    ghost_offset_fraction : float (keyword-only)
        Ghost placement offset from the occluder, in bbox-width units.
    ghost_confidence : float (keyword-only)
        Confidence assigned to injected ghost detections.
    ghost_occlusion_max_gap : int (keyword-only)
        Maximum gap length in frames still treated as an occlusion.
    ghost_top_exit_px : float (keyword-only)
        Top-edge tolerance (px) for classifying a drop as a top exit.

    Outputs
    -------
    tuple[pandas.DataFrame, OCSort, list[dict], list[dict], list[dict]]
        df : wide track table from the final pass.
        tracker : the ``OCSort`` instance from the final pass.
        ghost_log : injected ghosts (empty when the ghost pass did not run).
        top_exit_events : top-edge exits (empty when no ghost pass).
        top_reentry_events : top-edge reentries (empty when no ghost pass).
        The last three are bookkeeping the caller writes to ``tracker_log.json``.
    """
    do_ghost = (
        ghost_detection_enabled
        and vial_rois is not None
        and vial_expected_counts is not None
        and any(v > 0 for v in vial_expected_counts.values())
    )

    if do_ghost:
        print("Ghost detection enabled — running Pass 1 (baseline)...")
        df_pass1, _ = _run_tracker_pass(
            det_by_frame, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )
        n_pass1_tracks = len([c for c in df_pass1.columns if c.startswith("id")])
        print(f"  Pass 1: {n_pass1_tracks} tracks")

        top_exit_events, top_reentry_events, occlusion_gaps = _find_occlusion_gaps(
            df_pass1, vial_rois, vial_expected_counts, det_by_frame,
            top_exit_px=ghost_top_exit_px,
            occlusion_max_gap=ghost_occlusion_max_gap,
        )
        print(f"  Gap analysis: {len(occlusion_gaps)} occlusion gaps, "
              f"{len(top_exit_events)} top exits, {len(top_reentry_events)} top reentries")

        det_with_ghosts, ghost_log = _inject_ghost_detections(
            det_by_frame, occlusion_gaps, df_pass1, vial_rois,
            offset_fraction=ghost_offset_fraction,
            ghost_confidence=ghost_confidence,
        )
        print(f"  Injected {len(ghost_log)} ghost detections — running Pass 2...")

        df, tracker = _run_tracker_pass(
            det_with_ghosts, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )
        n_pass2_tracks = len([c for c in df.columns if c.startswith("id")])
        print(f"  Pass 2: {n_pass2_tracks} tracks")
    else:
        ghost_log = []
        top_exit_events = []
        top_reentry_events = []
        df, tracker = _run_tracker_pass(
            det_by_frame, n_frames, img_h, img_w, vial_rois, tracker_kwargs, max_frames
        )

    return df, tracker, ghost_log, top_exit_events, top_reentry_events


def export_tracks_xy_tuple_csv_one_config(
    video_path: str,
    output_csv: str,
    api_key: str,
    model_id: str,
    *,
    # Every value below is set in config.yaml and has no default here on
    # purpose: config.yaml is the single source of truth, so a caller that omits
    # one fails loudly rather than tracking with a stale hardcoded fallback.
    # (tests/test_tracking_contract.py guards this.)
    inference_api_url: str,
    inference_mode: str,
    local_weights_path: str,
    local_resolution: int,
    local_num_classes: int,
    local_optimize_for_gpu: bool,
    repo_root: str,
    detection_confidence_rfdetr: float,
    confidence: float,
    lost_track_buffer: int,
    minimum_matching_threshold: float,
    minimum_consecutive_frames: int,
    asso_func: str,
    brownian_pos_noise: float,
    aspect_weight: float,
    behavioral_weights: dict,
    overlap_weight_scale: float,
    jump_factor: float,
    jump_iou_threshold: float,
    jump_inertia: float,
    inertia: float,
    delta_t: int,
    overlap_iou_scale: float,
    edge_fraction: float,
    expected_count: int | None,
    w_under: float,
    w_over: float,
    ghost_detection_enabled: bool,
    ghost_offset_fraction: float,
    ghost_confidence: float,
    ghost_occlusion_max_gap: int,
    ghost_top_exit_px: float,
    # Runtime inputs, not config knobs. None means "not supplied / feature off":
    # a detections cache path, the vial ROIs, watershed settings, per-vial
    # expected counts, an optional frame cap, and an fps override (otherwise the
    # video's own fps is used).
    det_log_csv: str | None = None,
    vial_rois: dict | None = None,
    watershed_cfg: dict | None = None,
    vial_expected_counts: dict | None = None,
    max_frames: int | None = None,
    fps_assumed: float | None = None,
) -> tuple[pd.DataFrame, object]:
    """Track one video end-to-end for a single configuration and write the tracks.

    This is the top-level entry point: it runs RF-DETR detection (or reuses a
    cache), optionally splits multi-fly boxes with watershed, associates the
    detections into tracks with OC-SORT (optionally with the ghost second pass),
    and writes the result as a wide CSV where rows are frames, columns are track
    ids (``id{N}``), and cells are ``"(x, y)"`` centres (empty when absent). It
    exists so every caller (notebook, run_tracking, run_all, grid search) tracks
    a video the exact same way from one config.

    Side effects: writes ``output_csv``, the per-track observation log
    ``<output>_obs_logs.json``, and — on a fresh run or after watershed —
    ``det_log_csv`` (plus watershed debug crops when enabled).

    Inputs
    ------
    video_path : str
        Path to the video to track.
    output_csv : str
        Path to write the wide tracks CSV.
    api_key : str
        Roboflow API key (used only on a detection cache miss).
    model_id : str
        Roboflow model id, ``"<workspace>/<version>"``.
    inference_api_url : str (keyword-only, required)
        Roboflow inference host URL (hosted mode only).
    inference_mode : str (keyword-only, required)
        ``hosted`` or ``local``.
    local_weights_path : str (keyword-only, required)
        Path to local weights when mode is ``local``.
    local_resolution : int (keyword-only, required)
        Local model resolution.
    local_num_classes : int (keyword-only, required)
        Local model class count.
    local_optimize_for_gpu : bool (keyword-only, required)
        FP16 optimization on CUDA when True.
    repo_root : str (keyword-only, required)
        Repo root for relative weight paths.
    detection_confidence_rfdetr : float (keyword-only, required)
        RF-DETR confidence threshold at inference time.
    confidence : float (keyword-only, required)
        OC-SORT detection/spawn threshold (``det_thresh``).
    lost_track_buffer : int (keyword-only, required)
        Frames a track survives unmatched before removal (``max_age``).
    minimum_matching_threshold : float (keyword-only, required)
        Minimum association score to keep a match (``iou_threshold``).
    minimum_consecutive_frames : int (keyword-only, required)
        Frames needed to confirm a new track (``min_hits``).
    asso_func : str (keyword-only, required)
        Association metric name ("iou", "giou", "diou", "ciou", "hmiou").
    brownian_pos_noise : float (keyword-only, required)
        Kalman position process-noise scale.
    aspect_weight : float (keyword-only, required)
        Bonus weight for matching similar box aspect ratios.
    behavioral_weights : dict (keyword-only, required)
        Per-behaviour consistency weights (e.g. speed, scale, turning_angle,
        pause, acceleration).
    overlap_weight_scale : float (keyword-only, required)
        Multiplier applied to the behavioural weights when detections overlap.
    jump_factor : float (keyword-only, required)
        Velocity / search-region inflation for the second ("jump") match pass.
    jump_iou_threshold : float (keyword-only, required)
        Looser IoU threshold used in the jump pass.
    jump_inertia : float (keyword-only, required)
        Reduced direction-consistency weight used in the jump pass.
    inertia : float (keyword-only, required)
        Round-1 direction-consistency (OCM) weight.
    delta_t : int (keyword-only, required)
        Frames of look-back for the velocity direction estimate (-1 = full
        history).
    overlap_iou_scale : float (keyword-only, required)
        IoU downscale applied when detections overlap.
    edge_fraction : float (keyword-only, required)
        Vial-wall exclusion band (fraction) used in overlap handling.
    expected_count : int | None (keyword-only, required)
        Total expected flies enabling count-aware spawning; None disables it.
    w_under : float (keyword-only, required)
        Spawn penalty per tracker below ``expected_count``.
    w_over : float (keyword-only, required)
        Spawn penalty per tracker above ``expected_count``.
    ghost_detection_enabled : bool (keyword-only, required)
        Enable the two-pass ghost injection.
    ghost_offset_fraction : float (keyword-only, required)
        Ghost placement offset from the occluder, in bbox-width units.
    ghost_confidence : float (keyword-only, required)
        Confidence assigned to ghost detections.
    ghost_occlusion_max_gap : int (keyword-only, required)
        Maximum gap length in frames treated as an occlusion.
    ghost_top_exit_px : float (keyword-only, required)
        Top-edge tolerance (px) for classifying a drop as a top exit.
    det_log_csv : str | None, default None
        Detection cache path: read to skip inference, written after a fresh run.
        None = no cache.
    vial_rois : dict[str, tuple[int, int, int, int]] | None, default None
        vial id -> (x0, y0, x1, y1). Filters detections and enables the per-vial
        ghost logic.
    watershed_cfg : dict | None, default None
        Watershed settings; None or ``{"enabled": False}`` skips box splitting.
    vial_expected_counts : dict[str, int] | None, default None
        Per-vial expected fly counts that gate ghost detection.
    max_frames : int | None, default None
        Optional cap on frames processed (debugging); None = all frames.
    fps_assumed : float | None, default None
        fps override; None uses the video's own fps (falling back to 30.0 when
        the video reports none).

    Outputs
    -------
    tuple[pandas.DataFrame, OCSort]
        df : the wide-format tracks table that was written to ``output_csv``.
        tracker : the ``OCSort`` instance, carrying detection_log,
            suppressed_tracks, ghost_log, top_exit_events, and top_reentry_events.
    """
    det_by_frame, n_frames, img_h, img_w, fps, loaded_from_cache = collect_detections(
        video_path,
        det_log_csv=det_log_csv,
        api_key=api_key,
        model_id=model_id,
        inference_api_url=inference_api_url,
        inference_mode=inference_mode,
        local_weights_path=local_weights_path,
        local_resolution=local_resolution,
        local_num_classes=local_num_classes,
        local_optimize_for_gpu=local_optimize_for_gpu,
        repo_root=repo_root,
        detection_confidence_rfdetr=detection_confidence_rfdetr,
        max_frames=max_frames,
    )
    if fps_assumed is None:
        fps_assumed = float(fps) if fps and fps > 0 else 30.0

    # Split any multi-fly boxes before the tracker sees them, then persist the
    # (possibly edited) detections for cheap re-runs.
    watershed_on = bool(watershed_cfg and watershed_cfg.get("enabled", False))
    if watershed_on:
        det_by_frame = split_merged_detections(det_by_frame, video_path, watershed_cfg, det_log_csv)
    cache_detections(det_by_frame, det_log_csv,
                     loaded_from_cache=loaded_from_cache, watershed_on=watershed_on)

    # OC-SORT's constructor uses its own short parameter names (det_thresh,
    # max_age, min_hits, ...); this dict is the one place our config names are
    # translated to them.
    tracker_kwargs = dict(
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
        behavioral_weights=behavioral_weights,
        overlap_weight_scale=overlap_weight_scale,
        jump_factor=jump_factor,
        jump_iou_threshold=jump_iou_threshold,
        jump_inertia=jump_inertia,
        expected_count=expected_count,
        w_under=w_under,
        w_over=w_over,
        overlap_iou_scale=overlap_iou_scale,
        edge_fraction=edge_fraction,
        fps=fps_assumed,
    )

    df, tracker, ghost_log, top_exit_events, top_reentry_events = run_tracker(
        det_by_frame, n_frames, img_h, img_w, vial_rois, tracker_kwargs,
        max_frames=max_frames,
        ghost_detection_enabled=ghost_detection_enabled,
        vial_expected_counts=vial_expected_counts,
        ghost_offset_fraction=ghost_offset_fraction,
        ghost_confidence=ghost_confidence,
        ghost_occlusion_max_gap=ghost_occlusion_max_gap,
        ghost_top_exit_px=ghost_top_exit_px,
    )

    # The tracker carries the ghost/top-exit bookkeeping out so the caller can
    # log it alongside the tracks.
    tracker.ghost_log          = ghost_log
    tracker.top_exit_events    = top_exit_events
    tracker.top_reentry_events = top_reentry_events

    id_cols = [c for c in df.columns if c.startswith("id")]
    df.to_csv(output_csv, index=False, na_rep="")
    print(f"Saved: {output_csv}  (frames={len(df)}, tracks={len(id_cols)})")

    obs_log_path = output_csv.replace(".csv", "_obs_logs.json")
    save_observation_logs(tracker, obs_log_path)

    return df, tracker


def save_observation_logs(tracker, path: str) -> None:
    """Dump each track's per-frame observation history to a JSON file.

    This is what lets the re-linking pass (``parameter_tuning/relink_grid_search.py``)
    stitch broken tracks back together from a finished run without re-running
    detection or the tracker. Only tracks still alive at the end of the video
    carry an ``observation_log``; suppressed tracks are plain dicts and are
    skipped. Ids are written 1-based to match the ``id{N}`` columns in the CSV.

    Inputs
    ------
    tracker : OCSort
        The finished tracker. Its ``.trackers`` are ``KalmanBoxTracker`` objects,
        each with ``.observation_log`` (list of (frame, [x1,y1,x2,y2], score))
        and ``.id``.
    path : str
        Destination JSON path (conventionally ``<output>_obs_logs.json``).

    Outputs
    -------
    None
        Writes ``path`` as a side effect: a JSON list of
        ``{"id": int, "log": [[frame, [x1, y1, x2, y2], score], ...]}``.
    """
    import json
    records = []
    for trk in tracker.trackers:
        log = [
            [int(f), [float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(s) if s is not None else None]
            for f, b, s in trk.observation_log
        ]
        if log:
            records.append({"id": int(trk.id) + 1, "log": log})  # 1-based to match CSV
    with open(path, "w") as fh:
        json.dump(records, fh)
    print(f"Saved observation logs: {path}  ({len(records)} trackers)")
