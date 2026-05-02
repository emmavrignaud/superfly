"""LabelerBackend — Python QObject exposed to QML as `backend`.

Owns the data layer (raw detections + annotation store) and the current
frame index. QML reads properties and calls slots; Python emits signals
when state changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from .color_engine import track_color_hex
from .data_model import (
    AnnotationStore,
    Annotation,
    Detection,
    SOURCE_HUMAN,
    SOURCE_HUMAN_SYNTH,
    SOURCE_OCSORT,
    load_raw_detections,
    load_tracks_any,
    match_ocsort_to_raw,
)
from .session import (
    annotations_from_payload,
    load_session,
    save_session,
    synthetics_from_payload,
)
from .assets import (
    update_metadata_counts,
    write_export_summary,
)


# Auto-save runs this often, but only writes if there's been a mutation since
# the last save. 60 s matches the roadmap.
AUTOSAVE_INTERVAL_MS = 60_000


# True red for unannotated detections (off-palette — Catppuccin Mocha's
# "red" #f38ba8 is actually salmon-pink, too soft for a "needs attention" cue).
UNANNOTATED_COLOR = "#ef4444"

# Pixels added to each bbox edge for hit-testing only (rendered bbox unchanged).
# Forgives sloppy clicks on isolated flies; tight clusters still get
# disambiguated by the smallest-area-wins tie-break.
HIT_TEST_PAD = 5.0

# Minimum synthetic bbox side (video px). Shift+arrows shrink until this floor.
SYNTHETIC_MIN_SIDE_PX = 2.0


# ── Unified undo system ────────────────────────────────────────────────────
# One stack of ops, in chronological order, covering everything reversible:
# annotations (assign/clear) AND synthetic detections (create/resize/delete).
# Ctrl+Z pops the latest op and runs its reverse.

@dataclass
class _Op:
    """A reversible action. `kind` selects the reversal logic; `payload`
    carries kind-specific state needed to undo it."""
    kind: str           # "ann_assign" | "ann_clear" | "synth_create"
                        # | "synth_resize" | "synth_delete"
    payload: dict = field(default_factory=dict)


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
    statusChanged = Signal()          # last save / export / autosave status text changed
    autosavePulse = Signal()          # one-shot: fire on every successful autosave write
    displayFrameChanged = Signal()    # what frame the canvas should be showing changed
    isPlayingChanged = Signal()       # playback toggle
    modeChanged = Signal()            # frame <-> track
    focusedTrackChanged = Signal()    # which track is the Track Mode focus
    prefillRequested = Signal(int)    # auto-follow: prefill the bubble with this track id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str = ""
        self._raw_csv: str = ""
        self._ocsort_csv: Optional[str] = None
        self._session_path: str = ""
        self._autosave_path: str = ""
        self._export_path: str = ""
        self._out_dir: str = ""
        self._summary_path: str = ""

        self._frame_count: int = 0
        self._video_w: int = 0
        self._video_h: int = 0
        self._fps: float = 0.0
        self._current_frame: int = 0
        self._frame_tick: int = 0  # bumped on every seek to invalidate QML image cache
        self._selected_det_idx: int = -1   # -1 means nothing selected

        self._raw_by_frame: dict[int, list[Detection]] = {}
        self._store: Optional[AnnotationStore] = None

        self._dirty: bool = False        # has the store been mutated since last save?
        self._status_text: str = ""      # shown in HUD; updated by save/export/autosave
        self._autosave_timer: Optional[QTimer] = None

        # Playback state — one canvas, one-shot forward play from currentFrame
        # to (currentFrame + 1s), then snap back. Static otherwise.
        self._playback_frame: int = 0
        self._is_playing: bool = False
        self._playback_timer: Optional[QTimer] = None

        # Track Mode state. mode ∈ {"frame", "track"}. In track mode, the
        # canvas dims non-focused detections and overlays the focused track's
        # trajectory; otherwise behaves like Frame Mode.
        self._mode: str = "frame"
        self._focused_track_id: int = -1

        # Auto-follow trail: after each commit, ←/→ to a new frame auto-selects
        # the nearest detection and pre-fills the bubble with the same ID.
        # `_trail = (track_id, x, y)`. Cleared on Esc, mode change, big jumps.
        self._trail: Optional[tuple[int, float, float]] = None

        # Default bbox size for synthetic detections (median of real detections).
        self._median_w: float = 12.0
        self._median_h: float = 12.0
        # Counter for synthetic det_idx (negative, decrements per synthetic added).
        # Starts at -2 because -1 is reserved for "no selection".
        self._next_synth_idx: int = -2

        # Unified undo stack — every reversible action pushes one _Op.
        # See `undo()` for the reversal dispatch.
        self._ops: list[_Op] = []
        self._OPS_CAP: int = 500

        # After a click on overlapping bboxes, Tab cycles within this ordered
        # list (smallest-area-first); cleared on frame change / global cycle.
        self._overlap_stack: list[int] = []

        # When a mutation happens, mark dirty.
        self.annotationsChanged.connect(self._mark_dirty)

    # ── loading (called from main.py before the QML engine starts) ────────

    def load(
        self,
        video_path: str,
        raw_csv: str,
        ocsort_csv: Optional[str] = None,
        *,
        session_path: str = "",
        autosave_path: str = "",
        export_path: str = "",
        resume_from: str = "",
        out_dir: str = "",
        summary_path: str = "",
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps_raw if fps_raw > 1.0 else 30.0  # fallback for codecs that lie
        cap.release()

        self._video_path = str(Path(video_path).as_posix())
        self._raw_csv = str(Path(raw_csv).as_posix())
        self._ocsort_csv = str(Path(ocsort_csv).as_posix()) if ocsort_csv else None
        self._session_path = session_path
        self._autosave_path = autosave_path
        self._export_path = export_path
        self._out_dir = out_dir
        self._summary_path = summary_path

        self._raw_by_frame = load_raw_detections(raw_csv)
        self._compute_median_bbox()

        seed = {}
        if ocsort_csv:
            ocs = load_tracks_any(ocsort_csv)  # auto-detects wide vs long format
            seed = match_ocsort_to_raw(self._raw_by_frame, ocs)

        # If resuming from a session file, that takes precedence over OC-SORT seed.
        if resume_from:
            payload = load_session(resume_from)
            seed = annotations_from_payload(payload)
            self._current_frame = int(payload.get("current_frame", 0))
            # Re-add saved synthetic detections to raw_by_frame so annotations
            # against them validate. Update synth-idx counter to avoid collisions.
            for d in synthetics_from_payload(payload):
                self._raw_by_frame.setdefault(d.frame, []).append(d)
                if d.det_idx <= self._next_synth_idx:
                    self._next_synth_idx = d.det_idx - 1

        self._store = AnnotationStore(self._raw_by_frame, seed=seed)
        self._dirty = False

        if self._autosave_path:
            self._autosave_timer = QTimer(self)
            self._autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
            self._autosave_timer.timeout.connect(self._autosave_tick)
            self._autosave_timer.start()

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

    @Property(str, notify=statusChanged)
    def statusText(self) -> str:
        return self._status_text

    @Property(float, notify=videoSizeChanged)
    def fps(self) -> float:
        return self._fps

    @Property(int, notify=displayFrameChanged)
    def displayFrame(self) -> int:
        """Frame the canvas should currently render — playback frame if
        looping, otherwise the static currentFrame."""
        return self._playback_frame if self._is_playing else self._current_frame

    @Property(bool, notify=isPlayingChanged)
    def isPlaying(self) -> bool:
        return self._is_playing

    @Property(str, notify=modeChanged)
    def mode(self) -> str:
        return self._mode

    @Property(int, notify=focusedTrackChanged)
    def focusedTrackId(self) -> int:
        return self._focused_track_id

    @Property(int, notify=frameChanged)
    def timelineStartFrame(self) -> int:
        """First frame in the ±1 s timeline window around currentFrame."""
        if self._frame_count == 0:
            return 0
        span = max(1, int(round(self._fps)))
        return max(0, self._current_frame - span)

    @Property(int, notify=frameChanged)
    def timelineEndFrame(self) -> int:
        """Last frame (inclusive) in the ±1 s timeline window."""
        if self._frame_count == 0:
            return 0
        span = max(1, int(round(self._fps)))
        return min(self._frame_count - 1, self._current_frame + span)

    @Property(int, notify=tracksChanged)
    def nextFreeTrackId(self) -> int:
        """Smallest positive integer not currently assigned to any annotation."""
        if self._store is None:
            return 1
        used = {a.track_id for a in self._store.all().values()}
        n = 1
        while n in used:
            n += 1
        return n

    # ── slots (QML calls these) ────────────────────────────────────────────

    @Slot(int)
    def seek_frame(self, frame: int) -> None:
        self._do_seek(frame, follow_trail=True)

    @Slot(int)
    def jump_frame(self, frame: int) -> None:
        """Same as seek_frame but does NOT auto-follow the trail. Used for
        Home/End/PgUp/PgDn — explicit jumps clear the follow-fly state."""
        self._trail = None
        self._do_seek(frame, follow_trail=False)

    def _do_seek(self, frame: int, *, follow_trail: bool) -> None:
        if self._frame_count == 0:
            return
        target = max(0, min(self._frame_count - 1, int(frame)))
        if target == self._current_frame:
            return
        self._overlap_stack = []
        self._current_frame = target
        self._playback_frame = target
        self._frame_tick += 1
        # selection is per-frame; clear on navigate
        if self._selected_det_idx != -1:
            self._selected_det_idx = -1
            self.selectionChanged.emit()
        self.frameTickChanged.emit()
        self.frameChanged.emit(target)
        self.displayFrameChanged.emit()

        # Auto-follow: pick the nearest det in the new frame, pre-fill the
        # bubble. Bypassed for explicit jumps and when no trail is active.
        if follow_trail and self._trail is not None:
            self._auto_follow()

    def _auto_follow(self) -> None:
        if self._trail is None:
            return
        track_id, last_x, last_y = self._trail
        dets = self._raw_by_frame.get(self._current_frame, [])
        if not dets:
            return
        best = None  # (dist2, det_idx)
        for d in dets:
            d2 = (d.x - last_x) ** 2 + (d.y - last_y) ** 2
            if best is None or d2 < best[0]:
                best = (d2, d.det_idx)
        if best is None:
            return
        self._overlap_stack = []
        self._selected_det_idx = best[1]
        self.selectionChanged.emit()
        self.prefillRequested.emit(int(track_id))

    # ── playback (one canvas, ±1s loop) ────────────────────────────────────

    @Slot()
    def toggle_playback(self) -> None:
        if self._is_playing:
            self.pause_playback()
        else:
            self.play_playback()

    @Slot()
    def play_playback(self) -> None:
        if self._is_playing or self._frame_count == 0:
            return
        # Play the full ±1s window: start 1s before currentFrame, walk through
        # to 1s after, then snap back. Single pass — no looping.
        self._playback_frame = self.timelineStartFrame
        self._is_playing = True
        if self._playback_timer is None:
            self._playback_timer = QTimer(self)
            self._playback_timer.timeout.connect(self._playback_tick)
        interval_ms = max(16, int(round(1000.0 / max(1.0, self._fps))))
        self._playback_timer.setInterval(interval_ms)
        self._playback_timer.start()
        self.isPlayingChanged.emit()
        self.displayFrameChanged.emit()

    @Slot()
    def pause_playback(self) -> None:
        if not self._is_playing:
            return
        if self._playback_timer is not None:
            self._playback_timer.stop()
        self._is_playing = False
        # displayFrame falls back to currentFrame automatically via the
        # property; just notify QML.
        self.isPlayingChanged.emit()
        self.displayFrameChanged.emit()

    def _playback_tick(self) -> None:
        if not self._is_playing:
            return
        end = self.timelineEndFrame
        nxt = self._playback_frame + 1
        if nxt > end:
            # Reached end of window: snap back to anchor and stop. No loop.
            self.pause_playback()
            return
        self._playback_frame = nxt
        self.displayFrameChanged.emit()

    # ── Track Mode ────────────────────────────────────────────────────────

    @Slot()
    def toggle_track_mode(self) -> None:
        """Toggle Frame <-> Track Mode. Entering Track Mode focuses the first
        available track if none is currently focused."""
        if self._mode == "track":
            self._set_mode("frame")
            return
        if self._focused_track_id <= 0:
            tids = self._sorted_track_ids()
            if not tids:
                # Nothing to focus on — stay in frame mode.
                self._set_status("no tracks yet — assign one first")
                return
            self._focused_track_id = tids[0]
            self.focusedTrackChanged.emit()
        self._set_mode("track")

    @Slot(int)
    def set_focused_track(self, track_id: int) -> None:
        """Enter Track Mode focused on `track_id` (or stay in track mode and
        switch focus). No-op if track_id <= 0 or doesn't exist."""
        tid = int(track_id)
        if tid <= 0:
            return
        if tid not in self._sorted_track_ids():
            return
        if tid != self._focused_track_id:
            self._focused_track_id = tid
            self.focusedTrackChanged.emit()
        if self._mode != "track":
            self._set_mode("track")

    @Slot(int)
    def cycle_focused_track(self, step: int) -> None:
        """+1 = next track id; -1 = previous. Wraps. Track Mode only."""
        if self._mode != "track":
            return
        tids = self._sorted_track_ids()
        if not tids:
            return
        if self._focused_track_id in tids:
            i = tids.index(self._focused_track_id)
            new = tids[(i + int(step)) % len(tids)]
        else:
            new = tids[0]
        if new != self._focused_track_id:
            self._focused_track_id = new
            self.focusedTrackChanged.emit()

    @Slot(int, result=list)
    def track_positions(self, track_id: int) -> list:
        """Return [{frame, x, y}] for every annotated detection on this track,
        sorted by frame. Used to draw the trajectory polyline."""
        if self._store is None or int(track_id) <= 0:
            return []
        tid = int(track_id)
        out = []
        for (frame, det_idx), ann in self._store.all().items():
            if ann.track_id != tid:
                continue
            try:
                d = self._raw_by_frame[frame][det_idx]
            except (KeyError, IndexError):
                continue
            out.append({"frame": frame, "x": d.x, "y": d.y})
        out.sort(key=lambda r: r["frame"])
        return out

    def _set_mode(self, new_mode: str) -> None:
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self._trail = None  # mode change ends any in-progress follow trail
        self.modeChanged.emit()

    def _sorted_track_ids(self) -> list[int]:
        if self._store is None:
            return []
        return sorted({a.track_id for a in self._store.all().values()})

    def _lookup_det(self, frame: int, det_idx: int) -> Optional[Detection]:
        """Find a Detection by its `det_idx` field (not list position)."""
        for d in self._raw_by_frame.get(frame, []):
            if d.det_idx == det_idx:
                return d
        return None

    def _compute_median_bbox(self) -> None:
        """Median width/height of real detections — default size for synthetics."""
        ws, hs = [], []
        for dets in self._raw_by_frame.values():
            for d in dets:
                if d.is_synthetic:
                    continue
                ws.append(d.x2 - d.x1)
                hs.append(d.y2 - d.y1)
        if ws and hs:
            ws.sort(); hs.sort()
            self._median_w = max(SYNTHETIC_MIN_SIDE_PX, float(ws[len(ws) // 2]))
            self._median_h = max(SYNTHETIC_MIN_SIDE_PX, float(hs[len(hs) // 2]))

    # ── synthetic detections (Shift+Click) ─────────────────────────────────

    @Slot(float, float, result=int)
    def create_synthetic_at(self, x_video: float, y_video: float) -> int:
        """Add a synthetic detection at (x, y) in the current frame and
        auto-select it. Bbox sized to the median of real detections.
        Returns the new det_idx (negative)."""
        if self._frame_count == 0:
            return -1
        frame = self._current_frame
        w = self._median_w
        h = self._median_h
        det_idx = self._next_synth_idx
        self._next_synth_idx -= 1
        d = Detection(
            frame=frame, det_idx=det_idx,
            x=float(x_video), y=float(y_video),
            x1=float(x_video) - w / 2, y1=float(y_video) - h / 2,
            x2=float(x_video) + w / 2, y2=float(y_video) + h / 2,
            conf=float("nan"),
            is_synthetic=True,
        )
        self._raw_by_frame.setdefault(frame, []).append(d)
        self._push_op(_Op(kind="synth_create",
                          payload={"frame": frame, "det": d}))
        self._overlap_stack = []
        # Trigger redraw + auto-select
        self.annotationsChanged.emit()
        self._selected_det_idx = det_idx
        self.selectionChanged.emit()
        return det_idx

    @Slot(int, int)
    def resize_selected_synthetic(self, dw: int, dh: int) -> None:
        """Grow/shrink the selected synthetic's bbox by (dw, dh) pixels total
        (split evenly across each side). No-op for real detections."""
        if self._selected_det_idx == -1:
            return
        d = self._lookup_det(self._current_frame, self._selected_det_idx)
        if d is None or not d.is_synthetic:
            return
        prev_bbox = (d.x1, d.y1, d.x2, d.y2)
        new_w = max(SYNTHETIC_MIN_SIDE_PX, (d.x2 - d.x1) + float(dw))
        new_h = max(SYNTHETIC_MIN_SIDE_PX, (d.y2 - d.y1) + float(dh))
        d.x1 = d.x - new_w / 2
        d.x2 = d.x + new_w / 2
        d.y1 = d.y - new_h / 2
        d.y2 = d.y + new_h / 2
        self._push_op(_Op(kind="synth_resize",
                          payload={"frame": self._current_frame,
                                   "det_idx": d.det_idx,
                                   "prev_bbox": prev_bbox}))
        self.annotationsChanged.emit()

    @Slot()
    def delete_selected_synthetic(self) -> None:
        """Remove the selected synthetic detection (and any annotation on it).
        No-op for real detections — they come from the input CSV and aren't
        ours to delete. Reversible via Ctrl+Z."""
        if self._selected_det_idx == -1 or self._store is None:
            return
        d = self._lookup_det(self._current_frame, self._selected_det_idx)
        if d is None or not d.is_synthetic:
            return
        frame = self._current_frame
        prev_ann = self._store.get(frame, d.det_idx)
        # Remove any annotation on the synth, then remove the synth itself.
        if prev_ann is not None:
            self._store.remove_annotation_silent(frame, d.det_idx)
        dets = self._raw_by_frame.get(frame, [])
        for i, x in enumerate(dets):
            if x.det_idx == d.det_idx:
                dets.pop(i)
                break
        # Deselect (the det no longer exists).
        self._selected_det_idx = -1
        self.selectionChanged.emit()
        self._push_op(_Op(kind="synth_delete",
                          payload={"frame": frame, "det": d, "prev_ann": prev_ann}))
        self.annotationsChanged.emit()
        if prev_ann is not None:
            self.tracksChanged.emit()

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
                "is_synthetic": d.is_synthetic,
            })
        return out

    # ── selection ──────────────────────────────────────────────────────────

    def _hits_at_point(self, frame: int, x_video: float, y_video: float) -> list[int]:
        """All detections whose padded bbox contains (x,y), sorted like
        hit_test_bbox: ascending area, then centroid distance, then det_idx."""
        dets = self._raw_by_frame.get(frame, [])
        pad = HIT_TEST_PAD
        scored: list[tuple[float, float, int]] = []
        for d in dets:
            if not (d.x1 - pad <= x_video <= d.x2 + pad and d.y1 - pad <= y_video <= d.y2 + pad):
                continue
            area = max(0.0, (d.x2 - d.x1)) * max(0.0, (d.y2 - d.y1))
            cdist = (d.x - x_video) ** 2 + (d.y - y_video) ** 2
            scored.append((area, cdist, d.det_idx))
        scored.sort()
        return [t[2] for t in scored]

    @Slot(float, float, result=int)
    def hit_test_bbox(self, x_video: float, y_video: float) -> int:
        """Return det_idx of the detection whose bbox contains (x, y) in
        video-pixel space (in the *currently displayed* frame), or -1 if no
        bbox contains the point.

        On overlap, the bbox with the smallest area wins (more specific
        match). Ties broken by closeness of click to bbox centroid.
        """
        frame = self._playback_frame if self._is_playing else self._current_frame
        hits = self._hits_at_point(frame, x_video, y_video)
        return -1 if not hits else hits[0]

    @Slot(float, float)
    def select_at_video_point(self, x_video: float, y_video: float) -> None:
        """Select best detection under (x,y); if multiple overlap, Tab cycles
        that pile only. Empty patch clears selection."""
        frame = self._playback_frame if self._is_playing else self._current_frame
        hits = self._hits_at_point(frame, x_video, y_video)

        if hits:
            if self._is_playing:
                anchor = self._playback_frame
                self.pause_playback()
                if anchor != self._current_frame:
                    self._current_frame = anchor
                    self._frame_tick += 1
                    self.frameTickChanged.emit()
                    self.frameChanged.emit(anchor)
                    self.displayFrameChanged.emit()

            best = hits[0]
            self._overlap_stack = hits if len(hits) >= 2 else []
            if best == self._selected_det_idx:
                return
            self._selected_det_idx = best
            self.selectionChanged.emit()
            return

        self._overlap_stack = []
        self.clear_selection()

    @Slot(int)
    def select(self, det_idx: int) -> None:
        # If a click lands on a fly during playback, pause and anchor the
        # annotation context to the frame the user was actually looking at.
        if self._is_playing and det_idx != -1:
            anchor = self._playback_frame
            self.pause_playback()
            if anchor != self._current_frame:
                self._current_frame = anchor
                self._frame_tick += 1
                self.frameTickChanged.emit()
                self.frameChanged.emit(anchor)
                self.displayFrameChanged.emit()

        self._overlap_stack = []
        if det_idx == self._selected_det_idx:
            return
        # validate -1 or a real det_idx in the active (current/display) frame
        if det_idx != -1:
            dets = self._raw_by_frame.get(self._current_frame, [])
            if not any(d.det_idx == det_idx for d in dets):
                return
        self._selected_det_idx = int(det_idx)
        self.selectionChanged.emit()

    @Slot()
    def clear_selection(self) -> None:
        # Esc clears the auto-follow trail too — explicit "I'm done with that fly".
        self._trail = None
        self._overlap_stack = []
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
        stack = self._overlap_stack
        if (
            len(stack) >= 2
            and self._selected_det_idx in stack
        ):
            pos = stack.index(self._selected_det_idx)
            new = stack[(pos + step) % len(stack)]
            if new != self._selected_det_idx:
                self._selected_det_idx = new
                self.selectionChanged.emit()
            return

        self._overlap_stack = []
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
        d = self._lookup_det(self._current_frame, self._selected_det_idx)
        if d is not None:
            # Remember where we just committed so the next ←/→ can auto-follow.
            self._trail = (int(track_id), d.x, d.y)

        frame, det_idx = self._current_frame, self._selected_det_idx
        prev_ann = self._store.get(frame, det_idx)

        source = SOURCE_HUMAN_SYNTH if (d is not None and d.is_synthetic) else SOURCE_HUMAN
        # Use silent setter; we record the op ourselves (no double-bookkeeping).
        self._store.restore_annotation(frame, det_idx,
                                       Annotation(track_id=int(track_id), source=source))
        self._push_op(_Op(kind="ann_assign",
                          payload={"frame": frame, "det_idx": det_idx, "prev_ann": prev_ann}))
        self.annotationsChanged.emit()
        self.tracksChanged.emit()

    @Slot()
    def clear_selection_annotation(self) -> None:
        """Remove the annotation on the selected detection (if any)."""
        if self._store is None or self._selected_det_idx == -1:
            return
        frame, det_idx = self._current_frame, self._selected_det_idx
        prev_ann = self._store.get(frame, det_idx)
        if prev_ann is None:
            return  # nothing to clear; don't dirty the undo stack
        self._store.remove_annotation_silent(frame, det_idx)
        self._push_op(_Op(kind="ann_clear",
                          payload={"frame": frame, "det_idx": det_idx, "prev_ann": prev_ann}))
        self.annotationsChanged.emit()
        self.tracksChanged.emit()

    # ── unified undo ──────────────────────────────────────────────────────

    def _push_op(self, op: _Op) -> None:
        self._ops.append(op)
        if len(self._ops) > self._OPS_CAP:
            # Drop oldest; a 500-action history is plenty for human pace.
            self._ops.pop(0)

    @Slot()
    def undo(self) -> bool:
        """Reverse the most recent reversible action — annotations AND
        synthetic-detection ops, all in one chronological stack."""
        if not self._ops:
            return False
        op = self._ops.pop()
        ok = self._apply_reverse(op)
        return bool(ok)

    def _apply_reverse(self, op: _Op) -> bool:
        kind = op.kind
        p = op.payload
        if self._store is None:
            return False

        if kind == "ann_assign":
            frame, det_idx = p["frame"], p["det_idx"]
            prev = p["prev_ann"]
            if prev is None:
                self._store.remove_annotation_silent(frame, det_idx)
            else:
                self._store.restore_annotation(frame, det_idx, prev)
            self.annotationsChanged.emit()
            self.tracksChanged.emit()
            return True

        if kind == "ann_clear":
            frame, det_idx = p["frame"], p["det_idx"]
            self._store.restore_annotation(frame, det_idx, p["prev_ann"])
            self.annotationsChanged.emit()
            self.tracksChanged.emit()
            return True

        if kind == "synth_create":
            frame, d = p["frame"], p["det"]
            dets = self._raw_by_frame.get(frame, [])
            for i, x in enumerate(dets):
                if x.det_idx == d.det_idx:
                    dets.pop(i)
                    break
            if self._selected_det_idx == d.det_idx:
                self._selected_det_idx = -1
                self.selectionChanged.emit()
            self.annotationsChanged.emit()
            return True

        if kind == "synth_resize":
            d = self._lookup_det(p["frame"], p["det_idx"])
            if d is None:
                return False
            d.x1, d.y1, d.x2, d.y2 = p["prev_bbox"]
            self.annotationsChanged.emit()
            return True

        if kind == "synth_delete":
            frame, d, prev_ann = p["frame"], p["det"], p["prev_ann"]
            self._raw_by_frame.setdefault(frame, []).append(d)
            if prev_ann is not None:
                self._store.restore_annotation(frame, d.det_idx, prev_ann)
                self.tracksChanged.emit()
            self.annotationsChanged.emit()
            return True

        return False

    # ── track-panel data ───────────────────────────────────────────────────

    @Slot(int, result=bool)
    def would_duplicate_in_current_frame(self, track_id: int) -> bool:
        """True if assigning `track_id` to the selected detection would put
        the same ID on two different detections in the current frame.

        Used by the typing bubble to warn the user, not to block the assign.
        """
        if self._store is None or self._selected_det_idx == -1 or int(track_id) <= 0:
            return False
        tid = int(track_id)
        for (frame, det_idx), ann in self._store.all().items():
            if frame == self._current_frame and det_idx != self._selected_det_idx and ann.track_id == tid:
                return True
        return False

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

    # ── save / export / autosave ───────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.statusChanged.emit()

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _do_save(self, path: str) -> None:
        if self._store is None or not path:
            return
        synths = [
            d for dets in self._raw_by_frame.values()
            for d in dets if d.is_synthetic
        ]
        save_session(
            path,
            video_path=self._video_path,
            raw_csv=self._raw_csv,
            ocsort_csv=self._ocsort_csv,
            current_frame=self._current_frame,
            current_mode=self._mode,
            store=self._store,
            synthetic_detections=synths,
        )

    @Slot()
    def save(self) -> None:
        if not self._session_path:
            self._set_status("save: no session path configured")
            return
        try:
            self._do_save(self._session_path)
        except Exception as e:
            self._set_status(f"save failed: {e}")
            return
        self._dirty = False
        self._update_metadata(saved=True)
        self._set_status(f"saved -> {Path(self._session_path).name}")

    @Slot()
    def export_csv(self) -> None:
        if self._store is None or not self._export_path:
            self._set_status("export: no export path configured")
            return
        try:
            n = self._store.export_long_csv(self._export_path)
        except Exception as e:
            self._set_status(f"export failed: {e}")
            return

        # Companion QC summary next to the CSV.
        if self._summary_path:
            try:
                write_export_summary(
                    Path(self._summary_path),
                    annotations=self._store.all(),
                    raw_by_frame=self._raw_by_frame,
                    video_props={
                        "frame_count": self._frame_count,
                        "width": self._video_w,
                        "height": self._video_h,
                    },
                    export_csv_name=Path(self._export_path).name,
                )
            except Exception as e:
                # Don't fail the whole export over a summary glitch.
                self._set_status(f"exported {n} rows (summary failed: {e})")
                return

        self._update_metadata(exported=True)
        self._set_status(f"exported {n} rows -> {Path(self._export_path).name}")

    def _autosave_tick(self) -> None:
        if not self._dirty or not self._autosave_path:
            return
        try:
            self._do_save(self._autosave_path)
        except Exception as e:
            self._set_status(f"autosave failed: {e}")
            return
        self._dirty = False
        self._update_metadata(saved=True)
        self._set_status(f"autosaved {_short_time()}")
        self.autosavePulse.emit()

    def _update_metadata(self, *, saved: bool = False, exported: bool = False) -> None:
        if not self._out_dir or self._store is None:
            return
        anns = self._store.all()
        total = len(anns)
        human = sum(1 for a in anns.values() if a.source == SOURCE_HUMAN)
        ocsort = total - human
        tracks = len({a.track_id for a in anns.values()})
        update_metadata_counts(
            Path(self._out_dir),
            total=total, human=human, ocsort=ocsort, tracks=tracks,
            saved=saved, exported=exported,
        )


def _short_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
