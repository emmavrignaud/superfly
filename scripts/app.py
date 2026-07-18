#!/usr/bin/env python
"""
scripts/app.py

Single persistent setup window for the tracking pipeline.

Instead of the old flow — where the crop dialog and the vial-ROI dialog each
opened as a separate modal window that vanished on confirm — this launches ONE
window that stays on screen and walks the user through two pages:

    Page 1  Crop / trim   (draw a crop box + pick the kept frame range)
    Page 2  Vial ROIs     (draw each vial, set its fly count + colour)

A persistent footer holds the selected video (with a Browse button), two
checkboxes (Overlay, Use saved ROI), and Back / Next / Finish navigation.

On Finish the window writes the two toggles to config.yaml (comments preserved,
via ruamel), writes the crop + vial ROIs (+ per-vial fly count + colour) into
roi_library.json, closes, and hands off to the existing single-video pipeline:

    python scripts/run_tracking.py --video <selected>

which reuses the just-captured crop + ROIs (no GUI reopen) and prints its normal
progress in the console.

The heavy GUI logic is reused unchanged from the existing dialogs' canvases
(``_VideoCanvas`` in src/preprocessing.py, ``_MultiROICanvas`` in src/roi.py);
this file only rebuilds the small surrounding chrome and the window shell, so the
standalone dialogs and their callers are untouched.

Usage
-----
    python scripts/app.py
    python scripts/app.py --video data/raw/my_experiment.mp4   # preselect a video
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal

# Reused canvases + the shared per-vial control row from the existing dialogs
# (unmodified apart from the backward-compatible set_vial_colors hook on the
# canvas). _VialControlRow is the single fly-count + colour-swatch widget used by
# both this window and the standalone vial dialog.
from src.preprocessing import (
    _VideoCanvas, _SliderRow, _STYLE_ROI_NONE, _STYLE_ROI_SET,
    _QSS as _PP_QSS,
)
from src.roi import _MultiROICanvas, _VIAL_COLOURS, _VialControlRow, _QSS as _ROI_QSS
from src.plot_colors import load_vial_palette
from utils import load_config

CONFIG_PATH         = REPO_ROOT / "config.yaml"
ROI_LIBRARY_PATH    = REPO_ROOT / "roi_library.json"


# ═══════════════════════════════════════════════════════════════════════════
# Small persistence helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_library() -> dict:
    """Load roi_library.json (video-stem -> stored crop + vial geometry)."""
    if ROI_LIBRARY_PATH.exists():
        with open(ROI_LIBRARY_PATH) as f:
            return json.load(f)
    return {}


def _save_library(library: dict) -> None:
    """Persist the ROI library (pretty-printed), mirroring run_all._save_library."""
    with open(ROI_LIBRARY_PATH, "w") as f:
        json.dump(library, f, indent=2)


def _write_config_toggles(overlay: bool, use_saved_roi: bool) -> None:
    """Update the two boolean toggles in config.yaml, preserving all comments.

    Uses ruamel.yaml round-trip mode so the plain-language comments and worked
    examples in config.yaml survive the write untouched — only the two scalar
    values change.

    Inputs
    ------
    overlay : bool
        New value for ``visualization.enabled``.
    use_saved_roi : bool
        New value for ``roi.use_saved_roi``.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(CONFIG_PATH) as f:
        data = yaml.load(f)
    data["visualization"]["enabled"] = bool(overlay)
    data["roi"]["use_saved_roi"] = bool(use_saved_roi)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f)


def _read_first_frame(video_path: str, frame_idx: int = 0) -> Optional[np.ndarray]:
    """Return one BGR frame from a video (``frame_idx``), or None if unreadable."""
    cap = cv2.VideoCapture(video_path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 — Crop / trim
# ═══════════════════════════════════════════════════════════════════════════

class CropPage(QWidget):
    """Draw a crop box + choose the kept frame range for one video.

    Rebuilds the chrome of ``_ROIPickerDialog`` (src/preprocessing.py) around the
    reused ``_VideoCanvas``. ``get_crop()`` returns the same dict shape the rest
    of the pipeline stores: ``{x, y, w, h, start, end}`` in raw-video pixels.
    """

    crop_ready = pyqtSignal(bool)   # emitted when a crop box exists / is cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self.n_frames = 0
        self.start = 0
        self.end = 0
        self.cur = 0
        self._build()

    # ── layout ──────────────────────────────────────────────────────────────
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        hdr = QLabel("STEP 1  —  CROP & TRIM")
        hdr.setObjectName("header")
        root.addWidget(hdr)

        self.canvas = _VideoCanvas()
        self.canvas.roi_changed.connect(self._on_roi)
        root.addWidget(self.canvas, 1)

        self.roi_lbl = QLabel("Draw a crop box by dragging on the video")
        self.roi_lbl.setStyleSheet(_STYLE_ROI_NONE)
        root.addWidget(self.roi_lbl)

        self.sl_start = _SliderRow("Trim - keep from frame", 0, 1, 0)
        self.sl_end   = _SliderRow("Trim - keep until frame", 1, 1, 1)
        self.sl_cur   = _SliderRow("Preview frame", 0, 1, 0)
        self.sl_start.value_changed.connect(self._on_start)
        self.sl_end.value_changed.connect(self._on_end)
        self.sl_cur.value_changed.connect(self._on_cur)
        for sl in (self.sl_start, self.sl_end, self.sl_cur):
            root.addWidget(sl)

        self.stats_lbl = QLabel()
        self.stats_lbl.setObjectName("stats")
        root.addWidget(self.stats_lbl)

        btn_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_reset.clicked.connect(self.canvas.reset)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── video wiring ─────────────────────────────────────────────────────────
    def load_video(self, video_path: str) -> bool:
        """Open ``video_path``, show its first frame, and size the trim sliders.

        Returns True on success. Any previously open capture is released first.
        """
        self._release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        self._cap = cap
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start, self.cur = 0, 0
        self.end = self.n_frames if self.n_frames > 0 else 1

        hi_n = max(self.n_frames - 1, 1)
        hi_e = max(self.n_frames, 1)
        self.sl_start.slider.setRange(0, hi_n); self.sl_start.set_value(0)
        self.sl_end.slider.setRange(1, hi_e);   self.sl_end.set_value(self.end)
        self.sl_cur.slider.setRange(0, hi_n);   self.sl_cur.set_value(0)

        self.canvas.clear_roi()
        self._refresh_stats()
        self._load_frame()
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── slots (mirror _ROIPickerDialog) ──────────────────────────────────────
    def _on_roi(self, roi) -> None:
        has = roi is not None
        if has:
            x, y, w, h = roi
            self.roi_lbl.setText(f"Crop   x={x}   y={y}   w={w}   h={h}")
            self.roi_lbl.setStyleSheet(_STYLE_ROI_SET)
        else:
            self.roi_lbl.setText("Draw a crop box by dragging on the video")
            self.roi_lbl.setStyleSheet(_STYLE_ROI_NONE)
        self.btn_undo.setEnabled(has)
        self.btn_reset.setEnabled(has)
        self.crop_ready.emit(has)

    def _on_start(self, v: int) -> None:
        self.start = v
        if self.end <= self.start:
            self.end = self.start + 1
            self.sl_end.set_value(self.end)
        if self.cur < self.start:
            self.cur = self.start
            self.sl_cur.set_value(self.cur)
        self._refresh_stats()
        self._load_frame()

    def _on_end(self, v: int) -> None:
        self.end = v
        if self.end <= self.start:
            self.start = max(0, self.end - 1)
            self.sl_start.set_value(self.start)
        if self.cur >= self.end:
            self.cur = self.end - 1
            self.sl_cur.set_value(self.cur)
        self._refresh_stats()
        self._load_frame()

    def _on_cur(self, v: int) -> None:
        clamped = max(self.start, min(v, self.end - 1))
        if clamped != v:
            self.sl_cur.set_value(clamped)
        self.cur = clamped
        self._refresh_stats()
        self._load_frame()

    def _refresh_stats(self) -> None:
        kept = max(0, self.end - self.start)
        discarded = max(0, self.n_frames - kept)
        self.stats_lbl.setText(
            f"preview frame={self.cur}   |   trim: keep frames [{self.start}, {self.end})"
            f"   ->   {kept} kept, {discarded} discarded   (total: {self.n_frames})"
        )

    def _load_frame(self) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur)
        ok, frame = self._cap.read()
        if ok:
            self.canvas.set_frame(frame)

    # ── result ───────────────────────────────────────────────────────────────
    def has_crop(self) -> bool:
        return self.canvas.get_roi() is not None

    def get_crop(self) -> Optional[dict]:
        roi = self.canvas.get_roi()
        if roi is None:
            return None
        x, y, w, h = roi
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "start": int(self.start), "end": int(self.end)}

    def cropped_reference_frame(self) -> Optional[np.ndarray]:
        """Crop the trim-start frame with the drawn box (for the vial page).

        Reads ``crop['start']`` (the trim-start), not frame 0, so the vial ROIs
        are drawn on the same first frame the tracker will see, in the same
        cropped space. Matches ``draw_and_save_vial_rois``.
        """
        crop = self.get_crop()
        if crop is None or self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, crop["start"])
        ok, frame = self._cap.read()
        if not ok:
            return None
        x, y, w, h = crop["x"], crop["y"], crop["w"], crop["h"]
        sub = frame[y:y + h, x:x + w]
        return sub if sub.size else None


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Vial ROIs (+ per-vial fly count and colour)
# ═══════════════════════════════════════════════════════════════════════════

class VialPage(QWidget):
    """Draw vial ROIs on the cropped frame; set each vial's fly count + colour.

    Rebuilds the chrome of ``_VialROIDialog`` (src/roi.py) around the reused
    ``_MultiROICanvas``. Produces, in draw order, per-vial (bbox, count, colour).
    """

    rois_ready = pyqtSignal(bool)   # True when >= 1 ROI exists

    def __init__(self, snap_threshold_pct: float, snap_enabled: bool,
                 default_count: int, default_n_vials: int = 6,
                 default_gap_ratio: float = 0.5, parent=None):
        super().__init__(parent)
        self._default_count   = default_count
        self._default_n_vials = max(1, int(default_n_vials))
        self._default_gap     = min(1.0, max(0.0, float(default_gap_ratio)))
        self._rows: List[_VialControlRow] = []
        self._palette = load_vial_palette()   # fixed vial->hex from roi_library.json
        self._build(snap_threshold_pct, snap_enabled)

    def _build(self, snap_threshold_pct: float, snap_enabled: bool) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        hdr = QLabel("STEP 2  —  VIAL ROIs")
        hdr.setObjectName("header")
        root.addWidget(hdr)

        root.addLayout(self._build_suggest_row())

        self.canvas = _MultiROICanvas(snap_threshold_pct, snap_enabled)
        self.canvas.rois_changed.connect(self._on_rois_changed)
        self.canvas.snap_toggled.connect(lambda _e: self._refresh_status())
        root.addWidget(self.canvas, 1)

        self.status_lbl = QLabel()
        self.status_lbl.setObjectName("status")
        root.addWidget(self.status_lbl)

        # Per-vial controls, hidden until the first ROI is drawn.
        self._rows_container = QWidget()
        self._rows_layout = QHBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 2, 0, 2)
        self._rows_layout.setSpacing(12)
        _lbl = QLabel("Per vial (flies + colour):")
        _lbl.setObjectName("status")
        self._rows_layout.addWidget(_lbl)
        self._rows_layout.addStretch()
        self._rows_container.setVisible(False)
        root.addWidget(self._rows_container)

        btn_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_reset.clicked.connect(self.canvas.reset)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        root.addLayout(btn_row)
        self._refresh_status()

    def _build_suggest_row(self) -> QHBoxLayout:
        """Vial-count spinbox + gap slider that (re)generate the suggested boxes."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        _nv = QLabel("Vials:")
        _nv.setObjectName("status")
        row.addWidget(_nv)
        self.spin_nvials = QSpinBox()
        self.spin_nvials.setRange(1, 60)
        self.spin_nvials.setValue(self._default_n_vials)
        self.spin_nvials.valueChanged.connect(self._resuggest)
        row.addWidget(self.spin_nvials)

        row.addSpacing(16)
        _gp = QLabel("Gap:")
        _gp.setObjectName("status")
        row.addWidget(_gp)
        self.slider_gap = QSlider(Qt.Horizontal)
        self.slider_gap.setRange(0, 100)   # percent of a vial's width
        self.slider_gap.setValue(int(round(self._default_gap * 100)))
        self.slider_gap.setFixedWidth(160)
        self.slider_gap.valueChanged.connect(lambda _v: self._resuggest())
        row.addWidget(self.slider_gap)
        self.gap_val_lbl = QLabel(f"{self._default_gap:.2f}")
        self.gap_val_lbl.setObjectName("status")
        row.addWidget(self.gap_val_lbl)

        self.btn_suggest = QPushButton("Suggest")
        self.btn_suggest.setToolTip(
            "Replace the boxes with N evenly spaced vials across the frame"
        )
        self.btn_suggest.clicked.connect(self._resuggest)
        row.addWidget(self.btn_suggest)
        row.addStretch()
        return row

    def _resuggest(self) -> None:
        """Regenerate the uniform suggestion from the current count + gap."""
        r = self.slider_gap.value() / 100.0
        self.gap_val_lbl.setText(f"{r:.2f}")
        self.canvas.prefill_uniform_vials(self.spin_nvials.value(), r)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self.canvas.set_frame(frame_bgr)
        self._resuggest()   # show the auto-suggested boxes as soon as the crop lands

    # ── per-vial rows ────────────────────────────────────────────────────────
    def _sync_rows(self, n: int) -> None:
        while len(self._rows) > n:
            row = self._rows.pop()
            row.deleteLater()
        while len(self._rows) < n:
            idx = len(self._rows)
            color = self._palette.get(f"vial{idx + 1}") or _VIAL_COLOURS[idx % len(_VIAL_COLOURS)]
            row = _VialControlRow(idx, self._default_count, color)
            row.changed.connect(self._push_to_canvas)
            pos = self._rows_layout.count() - 1   # before the trailing stretch
            self._rows_layout.insertWidget(pos, row)
            self._rows.append(row)
        self._rows_container.setVisible(n > 0)
        self._push_to_canvas()

    def _push_to_canvas(self) -> None:
        self.canvas.set_fly_counts([r.count() for r in self._rows])
        self.canvas.set_vial_colors([r.color for r in self._rows])

    def _on_rois_changed(self, rois: list) -> None:
        n = len(rois)
        self._sync_rows(n)
        self.btn_undo.setEnabled(n > 0)
        self._refresh_status()
        self.rois_ready.emit(n > 0)

    def _refresh_status(self) -> None:
        n = len(self.canvas.get_rois())
        snap = "[snap ON]" if self.canvas.snap_enabled else "[snap OFF]"
        action = "ready" if n > 0 else "drag to add a vial"
        self.status_lbl.setText(
            f"{n} vial{'s' if n != 1 else ''} drawn  -  {action}    {snap}  (S to toggle)"
        )

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_U:
            self.canvas.undo()
        elif e.key() == Qt.Key_R:
            self.canvas.reset()
        elif e.key() == Qt.Key_S:
            self.canvas.toggle_snap()
        else:
            super().keyPressEvent(e)

    # ── result ───────────────────────────────────────────────────────────────
    def get_vials(self) -> List[Tuple[Tuple[int, int, int, int], int, str]]:
        """(bbox, fly_count, colour_hex) per vial, in draw order."""
        rois = self.canvas.get_rois()
        out = []
        for i, bbox in enumerate(rois):
            count = self._rows[i].count() if i < len(self._rows) else self._default_count
            color = (self._rows[i].color if i < len(self._rows)
                     else self._palette.get(f"vial{i + 1}") or _VIAL_COLOURS[i % len(_VIAL_COLOURS)])
            out.append((tuple(bbox), count, color))
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════

class AppWindow(QMainWindow):
    """Persistent shell: video picker + toggles + Crop/Vial pages + nav footer."""

    def __init__(self, cfg, initial_video: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.video_path: Optional[str] = None
        self.finished_ok = False           # set True only on a successful Finish
        self.reuse_only = False            # True -> handoff reuses saved ROIs, no draw

        snap_pct = float(cfg.roi.snap_threshold_pct)
        snap_on  = bool(cfg.roi.snap_enabled)
        default_count = int(cfg.pipeline.expected_per_vial)
        default_n_vials = int(getattr(cfg.roi, "n_vials", 6))
        default_gap = float(getattr(cfg.roi, "gap_ratio", 0.5))

        self.setWindowTitle("SuperFly - Setup")
        self.setMinimumSize(1120, 860)
        self.resize(1380, 960)
        self.setStyleSheet(_PP_QSS + "\n" + _ROI_QSS)

        self.crop_page = CropPage()
        self.vial_page = VialPage(snap_pct, snap_on, default_count,
                                  default_n_vials, default_gap)
        self.crop_page.crop_ready.connect(self._update_nav)
        self.vial_page.rois_ready.connect(self._update_nav)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.crop_page)   # index 0
        self.stack.addWidget(self.vial_page)   # index 1

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.stack, 1)            # canvas fills the remaining width
        outer.addWidget(self._build_sidebar())    # fixed-ish ~22% control column
        self.setCentralWidget(central)

        if initial_video:
            self._set_video(initial_video)
        self._update_nav()

    # ── sidebar (right-hand control column) ───────────────────────────────────
    def _build_sidebar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("sidebar")
        bar.setMinimumWidth(220)
        bar.setMaximumWidth(300)   # ~22% of the default 1380px window
        v = QVBoxLayout(bar)
        v.setContentsMargins(16, 18, 16, 18)
        v.setSpacing(12)

        title = QLabel("SETUP")
        title.setObjectName("header")
        v.addWidget(title)

        # video
        _vl = QLabel("Video")
        _vl.setObjectName("status")
        v.addWidget(_vl)
        self.video_label = QLabel("No video selected")
        self.video_label.setObjectName("status")
        self.video_label.setWordWrap(True)
        v.addWidget(self.video_label)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        v.addWidget(browse)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setObjectName("divider")
        v.addWidget(div)

        # toggles (stacked in the narrow column)
        self.chk_overlay = QCheckBox("Overlay videos")
        self.chk_overlay.setChecked(bool(self.cfg.visualization.enabled))
        self.chk_overlay.setToolTip("Write detection + track overlay MP4s after tracking")
        self.chk_saved = QCheckBox("Use saved ROI when available")
        self.chk_saved.setChecked(bool(self.cfg.roi.use_saved_roi))
        self.chk_saved.setToolTip(
            "If this video already has a saved crop + ROIs, skip drawing and reuse them"
        )
        self.chk_saved.stateChanged.connect(lambda _s: self._reevaluate_reuse())
        v.addWidget(self.chk_overlay)
        v.addWidget(self.chk_saved)

        v.addStretch(1)

        # navigation (stacked, Finish emphasised at the bottom)
        self.btn_back = QPushButton("< Back")
        self.btn_back.clicked.connect(self._back)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self._next)
        self.btn_finish = QPushButton("Finish + Track")
        self.btn_finish.setObjectName("done")
        self.btn_finish.clicked.connect(self._finish)
        for b in (self.btn_back, self.btn_next, self.btn_finish):
            v.addWidget(b)
        return bar

    # ── video selection ──────────────────────────────────────────────────────
    def _browse(self) -> None:
        start_dir = str(REPO_ROOT / "data" / "raw")
        if not Path(start_dir).exists():
            start_dir = str(REPO_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a video", start_dir,
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self._set_video(path)

    def _set_video(self, path: str) -> None:
        self.video_path = path
        self.video_label.setText(Path(path).name)
        self.video_label.setToolTip(path)
        ok = self.crop_page.load_video(path)
        if not ok:
            self.video_label.setText(f"[unreadable]\n{Path(path).name}")
            self.video_path = None
        self.stack.setCurrentIndex(0)
        self._reevaluate_reuse()
        self._update_nav()

    def _reevaluate_reuse(self) -> None:
        """Decide whether this video can skip drawing (saved ROI + checkbox on)."""
        self.reuse_only = False
        if self.video_path and self.chk_saved.isChecked():
            entry = _load_library().get(Path(self.video_path).stem, {})
            if entry.get("preprocessing") and entry.get("vial_rois"):
                self.reuse_only = True
        self._update_nav()

    # ── navigation ───────────────────────────────────────────────────────────
    def _back(self) -> None:
        self.stack.setCurrentIndex(0)
        self._update_nav()

    def _next(self) -> None:
        if self.stack.currentIndex() != 0:
            return
        if not self.crop_page.has_crop():
            return
        cropped = self.crop_page.cropped_reference_frame()
        if cropped is None:
            return
        self.vial_page.set_frame(cropped)
        self.stack.setCurrentIndex(1)
        self._update_nav()

    def _update_nav(self, *args) -> None:
        page = self.stack.currentIndex()
        have_video = self.video_path is not None
        # Reuse shortcut: saved ROIs available -> Finish straight away.
        if self.reuse_only:
            self.btn_back.setVisible(False)
            self.btn_next.setVisible(False)
            self.btn_finish.setVisible(True)
            self.btn_finish.setEnabled(have_video)
            return
        self.btn_back.setVisible(page == 1)
        self.btn_next.setVisible(page == 0)
        self.btn_finish.setVisible(page == 1)
        self.btn_next.setEnabled(have_video and self.crop_page.has_crop())
        self.btn_finish.setEnabled(
            have_video and self.stack.currentIndex() == 1
            and len(self.vial_page.canvas.get_rois()) > 0
        )

    # ── finish ───────────────────────────────────────────────────────────────
    def _finish(self) -> None:
        if not self.video_path:
            return
        stem = Path(self.video_path).stem
        library = _load_library()

        if not self.reuse_only:
            crop = self.crop_page.get_crop()
            vials = self.vial_page.get_vials()   # (bbox, count, colour) draw order
            if crop is None or not vials:
                return
            # Sort left -> right by box centre-x, keeping counts + colours aligned.
            vials_sorted = sorted(vials, key=lambda t: (t[0][0] + t[0][2]) / 2.0)
            vial_rois = {}
            palette_edits = {}
            for i, (bbox, count, color) in enumerate(vials_sorted, start=1):
                vial_rois[f"vial{i}"] = {
                    "bbox": [int(c) for c in bbox],
                    "n_flies": int(count),
                }
                palette_edits[f"vial{i}"] = color
            entry = library.setdefault(stem, {})
            entry["preprocessing"] = crop
            entry["video_path"] = str(self.video_path)
            entry["vial_rois"] = vial_rois
            # Colours live in one fixed top-level block (vial i -> colour), the
            # single source for the overlay and analysis. Editing a colour here
            # updates that block for every video and both views.
            library.setdefault("vial_colors", {}).update(palette_edits)
            _save_library(library)

        # Persist the user's actual toggle choices (honest config edit). The
        # handoff reuses the just-captured ROIs via a transient --reuse-roi flag,
        # so we do NOT force use_saved_roi here.
        _write_config_toggles(
            overlay=self.chk_overlay.isChecked(),
            use_saved_roi=self.chk_saved.isChecked(),
        )

        self.finished_ok = True
        self.close()

    def closeEvent(self, e) -> None:
        self.crop_page._release()
        super().closeEvent(e)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def _handoff(video_path: str) -> int:
    """Run the existing single-video pipeline on ``video_path`` in the console.

    Invokes the very same ``run_tracking.py`` a CLI user would, plus ``--reuse-roi``
    so it runs headless off the ROIs the window just captured — the GUI and the
    CLI therefore share one execution path.
    """
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_tracking.py"),
           "--video", video_path, "--reuse-roi"]
    print(f"\n>>> Handoff: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperFly setup window + tracking handoff.")
    parser.add_argument("--video", help="Preselect a video (skips the first Browse).")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)

    app = QApplication.instance() or QApplication(sys.argv)
    win = AppWindow(cfg, initial_video=args.video)
    win.show()
    win.raise_()
    win.activateWindow()
    app.exec_()

    if win.finished_ok and win.video_path:
        sys.exit(_handoff(win.video_path))
    print("Setup cancelled - nothing tracked.")


if __name__ == "__main__":
    main()
