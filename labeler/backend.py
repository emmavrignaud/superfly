"""LabelerBackend — Python QObject exposed to QML as `backend`.

Owns the data layer (raw detections + annotation store) and the current
frame index. QML reads properties and calls slots; Python emits signals
when state changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import Property, QObject, Signal, Slot

from .color_engine import track_color_hex
from .data_model import (
    AnnotationStore,
    Detection,
    SOURCE_HUMAN,
    SOURCE_OCSORT,
    load_ocsort_wide,
    load_raw_detections,
    match_ocsort_to_raw,
)


# Subtle grey for unannotated detections (Catppuccin Mocha overlay0).
UNANNOTATED_COLOR = "#6c7086"


class LabelerBackend(QObject):
    # ── signals ────────────────────────────────────────────────────────────
    frameChanged = Signal(int)
    frameTickChanged = Signal()
    frameCountChanged = Signal()
    videoSizeChanged = Signal()
    videoPathChanged = Signal()
    selectionChanged = Signal()       # selected_det_idx changed
    annotationsChanged = Signal()     # any mutation to the store
    tracksChanged = Signal()          # set of track_ids changed (subset of annotationsChanged)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str = ""
        self._frame_count: int = 0
        self._video_w: int = 0
        self._video_h: int = 0
        self._current_frame: int = 0
        self._frame_tick: int = 0  # bumped on every seek to invalidate QML image cache
        self._selected_det_idx: int = -1   # -1 means nothing selected

        self._raw_by_frame: dict[int, list[Detection]] = {}
        self._store: Optional[AnnotationStore] = None

    # ── loading (called from main.py before the QML engine starts) ────────

    def load(self, video_path: str, raw_csv: str, ocsort_csv: Optional[str] = None) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._video_path = str(Path(video_path).as_posix())

        self._raw_by_frame = load_raw_detections(raw_csv)

        seed = {}
        if ocsort_csv:
            ocs = load_ocsort_wide(ocsort_csv)
            seed = match_ocsort_to_raw(self._raw_by_frame, ocs)
        self._store = AnnotationStore(self._raw_by_frame, seed=seed)

        # Notify QML once everything is loaded
        self.frameCountChanged.emit()
        self.videoSizeChanged.emit()
        self.videoPathChanged.emit()
        self.frameChanged.emit(self._current_frame)
        self.frameTickChanged.emit()

    # ── properties (QML reads these) ───────────────────────────────────────

    @Property(int, notify=frameCountChanged)
    def frameCount(self) -> int:
        return self._frame_count

    @Property(int, notify=videoSizeChanged)
    def videoWidth(self) -> int:
        return self._video_w

    @Property(int, notify=videoSizeChanged)
    def videoHeight(self) -> int:
        return self._video_h

    @Property(int, notify=frameChanged)
    def currentFrame(self) -> int:
        return self._current_frame

    @Property(int, notify=frameTickChanged)
    def frameTick(self) -> int:
        return self._frame_tick

    @Property(str, notify=videoPathChanged)
    def videoPath(self) -> str:
        return self._video_path

    @Property(int, notify=selectionChanged)
    def selectedDetIdx(self) -> int:
        return self._selected_det_idx

    # ── slots (QML calls these) ────────────────────────────────────────────

    @Slot(int)
    def seek_frame(self, frame: int) -> None:
        if self._frame_count == 0:
            return
        target = max(0, min(self._frame_count - 1, int(frame)))
        if target == self._current_frame:
            return
        self._current_frame = target
        self._frame_tick += 1
        # selection is per-frame; clear on navigate
        if self._selected_det_idx != -1:
            self._selected_det_idx = -1
            self.selectionChanged.emit()
        self.frameTickChanged.emit()
        self.frameChanged.emit(target)

    @Slot(int, result=list)
    def detections_for_frame(self, frame: int) -> list:
        """Return a list of dicts describing every raw detection in `frame`.

        Shape per entry:
            {
              det_idx: int,
              x, y: float,             # centroid
              x1, y1, x2, y2: float,   # bbox
              track_id: int,           # -1 if unannotated
              source: str,             # "ocsort" | "human" | "" if unannotated
              color: str,              # hex; grey if unannotated
              filled: bool,            # true iff source == "human"
            }
        """
        dets = self._raw_by_frame.get(int(frame), [])
        out: list[dict] = []
        for d in dets:
            ann = self._store.get(int(frame), d.det_idx) if self._store else None
            if ann is None:
                tid, src, color, filled = -1, "", UNANNOTATED_COLOR, False
            else:
                tid = ann.track_id
                src = ann.source
                color = track_color_hex(tid)
                filled = (src == SOURCE_HUMAN)
            out.append({
                "det_idx": d.det_idx,
                "x": d.x, "y": d.y,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "track_id": tid,
                "source": src,
                "color": color,
                "filled": filled,
            })
        return out

    # ── selection ──────────────────────────────────────────────────────────

    @Slot(float, float, result=int)
    def hit_test_bbox(self, x_video: float, y_video: float) -> int:
        """Return det_idx of the detection whose bbox contains (x, y) in
        video-pixel space, or -1 if no bbox contains the point.

        On overlap, the bbox with the smallest area wins (more specific
        match). Ties broken by closeness of click to bbox centroid.
        """
        dets = self._raw_by_frame.get(self._current_frame, [])
        best: Optional[tuple[float, float, int]] = None  # (area, centroid_dist, det_idx)
        for d in dets:
            if not (d.x1 <= x_video <= d.x2 and d.y1 <= y_video <= d.y2):
                continue
            area = max(0.0, (d.x2 - d.x1)) * max(0.0, (d.y2 - d.y1))
            cdist = (d.x - x_video) ** 2 + (d.y - y_video) ** 2
            cand = (area, cdist, d.det_idx)
            if best is None or cand < best:
                best = cand
        return -1 if best is None else best[2]

    @Slot(int)
    def select(self, det_idx: int) -> None:
        if det_idx == self._selected_det_idx:
            return
        # validate -1 or a real det_idx in current frame
        if det_idx != -1:
            dets = self._raw_by_frame.get(self._current_frame, [])
            if not any(d.det_idx == det_idx for d in dets):
                return
        self._selected_det_idx = int(det_idx)
        self.selectionChanged.emit()

    @Slot()
    def clear_selection(self) -> None:
        if self._selected_det_idx != -1:
            self._selected_det_idx = -1
            self.selectionChanged.emit()

    @Slot()
    def select_next(self) -> None:
        self._cycle_selection(+1)

    @Slot()
    def select_prev(self) -> None:
        self._cycle_selection(-1)

    def _cycle_selection(self, step: int) -> None:
        dets = self._raw_by_frame.get(self._current_frame, [])
        if not dets:
            return
        idxs = [d.det_idx for d in dets]
        if self._selected_det_idx == -1:
            new = idxs[0] if step > 0 else idxs[-1]
        else:
            try:
                pos = idxs.index(self._selected_det_idx)
            except ValueError:
                new = idxs[0]
            else:
                new = idxs[(pos + step) % len(idxs)]
        self._selected_det_idx = new
        self.selectionChanged.emit()

    # ── annotation mutations ───────────────────────────────────────────────

    @Slot(int)
    def assign_to_selection(self, track_id: int) -> None:
        """Assign `track_id` to the currently selected detection (human source)."""
        if self._store is None or self._selected_det_idx == -1:
            return
        if track_id <= 0:
            return
        self._store.assign(self._current_frame, self._selected_det_idx, int(track_id))
        self.annotationsChanged.emit()
        self.tracksChanged.emit()

    @Slot()
    def clear_selection_annotation(self) -> None:
        """Remove the annotation on the selected detection (if any)."""
        if self._store is None or self._selected_det_idx == -1:
            return
        self._store.clear(self._current_frame, self._selected_det_idx)
        self.annotationsChanged.emit()
        self.tracksChanged.emit()

    @Slot()
    def undo(self) -> bool:
        if self._store is None:
            return False
        ok = self._store.undo()
        if ok:
            self.annotationsChanged.emit()
            self.tracksChanged.emit()
        return bool(ok)

    # ── track-panel data ───────────────────────────────────────────────────

    @Slot(result=list)
    def track_summary(self) -> list:
        """Return [{track_id, color, count, human_count}] for the right-side
        panel. `count` is total annotations on that track; `human_count` is
        the subset that have been human-confirmed (vs. still-OC-SORT)."""
        if self._store is None:
            return []
        counts: dict[int, list[int]] = {}  # tid -> [total, human]
        for ann in self._store.all().values():
            t = counts.setdefault(ann.track_id, [0, 0])
            t[0] += 1
            if ann.source == SOURCE_HUMAN:
                t[1] += 1
        out = []
        for tid in sorted(counts.keys()):
            total, human = counts[tid]
            out.append({
                "track_id": tid,
                "color": track_color_hex(tid),
                "count": total,
                "human_count": human,
            })
        return out
