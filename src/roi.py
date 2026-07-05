"""
src/roi.py

Interactive vial ROI drawing + assignment of ordered (sequential) fly IDs.

Workflow
--------
1. draw_and_save_vial_rois()       — one-time manual annotation per experiment setup
2. assign_vials_and_ordered_ids()  — assign vial labels + ordered IDs to each point
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from src.ui_context import VideoContext, build_context_chips, build_window_title

from PyQt5.QtWidgets import (
    QApplication, QColorDialog, QDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap


# ─── Config loader ────────────────────────────────────────────────────────────

def _roi_cfg() -> dict:
    p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f).get("roi", {})


def _default_fly_count() -> int:
    """Read pipeline.expected_per_vial from config as default spinbox value."""
    p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        return 7
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return int(cfg.get("pipeline", {}).get("expected_per_vial", 7))


# ─── Stylesheet (Catppuccin Mocha) ────────────────────────────────────────────

_QSS = """
QDialog, QWidget {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#header {
    font-size: 18px;
    font-weight: bold;
    color: #89b4fa;
    padding: 6px 0;
}
QLabel#status {
    color: #a6adc8;
    font-size: 12px;
    padding: 4px 0;
}
QPushButton {
    background: #313244;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
}
QPushButton:hover    { background: #45475a; }
QPushButton:pressed  { background: #585b70; }
QPushButton:disabled { color: #6c7086; }
QPushButton#done {
    background: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
}
QPushButton#done:hover     { background: #94e2d5; }
QPushButton#done:disabled  { background: #313244; color: #6c7086; }
"""

# One distinct colour per vial (cycles if n_vials > 6)
_VIAL_COLOURS = ["#a6e3a1", "#89b4fa", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5"]


# ─── Multi-ROI canvas ─────────────────────────────────────────────────────────

_COLOUR_GUIDE = "#89dceb"   # sky — both guide types; solid=horizontal, dashed=vertical


class _MultiROICanvas(QLabel):
    """
    QLabel that renders a video frame and lets the user drag multiple ROIs.
    Each confirmed ROI is drawn in a distinct colour and labelled with its index.

    Snap guides
    -----------
    Horizontal: derived from the last confirmed vial's top-y / bottom-y.
      When the in-progress rect's top or bottom edge comes within
      snap_threshold px of a guide, that edge locks to the guide y.
    Vertical gap: once ≥2 vials exist, the inferred inter-vial gap is used
      to project a left-edge snap column for the next vial.
    Press S (handled by the parent dialog) to toggle snap on/off; this only
    affects the vial currently being drawn.
    """

    rois_changed = pyqtSignal(list)   # emits current list of (x0, y0, x1, y1)
    snap_toggled = pyqtSignal(bool)   # emits new snap_enabled state

    def __init__(self, snap_threshold_pct: float, snap_enabled: bool,
                 parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #11111b; border-radius: 6px;")

        self._raw      = None
        self._vw       = 1
        self._vh       = 1
        self._rois: List[Tuple[int,int,int,int]] = []
        self._drawing  = False
        self._p0       = None        # drag anchor in label coords — NEVER modified after press
        self._p0_vid   = None        # drag anchor in video coords — NEVER modified after press
        self._p1       = None        # current drag corner in label coords (snapped)

        # snap state
        self._snap_threshold_pct = snap_threshold_pct
        self.snap_enabled        = snap_enabled
        self._active_h_guides: List[int] = []   # h-guide y values active this frame
        self._active_v_guide: Optional[int] = None  # v-guide x snapped at press time
        self._cursor_vid: Optional[QPoint] = None   # current mouse in video coords
        self._fly_counts: List[int] = []
        self._vial_colors: List[str] = []   # per-vial colour override (hex);
        #                                     empty -> fall back to _VIAL_COLOURS

    # ── public ────────────────────────────────────────────────────────────────

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._vh, self._vw = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, self._vw, self._vh, 3 * self._vw,
                      QImage.Format_RGB888).copy()
        self._raw = QPixmap.fromImage(qimg)
        self._repaint()

    def get_rois(self):
        return list(self._rois)

    def set_fly_counts(self, counts: List[int]) -> None:
        self._fly_counts = list(counts)
        self._repaint()

    def set_vial_colors(self, colors: List[str]) -> None:
        """Override the per-vial box/label colour (hex strings, in draw order).

        Optional: when unset (empty list) the canvas keeps the default
        ``_VIAL_COLOURS`` cycle, so existing callers are unaffected. app.py's
        vial page calls this so a user's colour picks show live on the boxes.
        """
        self._vial_colors = list(colors)
        self._repaint()

    def _colour_for(self, idx: int) -> str:
        """Colour for vial ``idx``: the override when present, else the cycle."""
        if idx < len(self._vial_colors) and self._vial_colors[idx]:
            return self._vial_colors[idx]
        return _VIAL_COLOURS[idx % len(_VIAL_COLOURS)]

    def toggle_snap(self) -> bool:
        self.snap_enabled = not self.snap_enabled
        self._active_h_guides = []
        self._active_v_guide  = None
        self.snap_toggled.emit(self.snap_enabled)
        self._repaint()
        return self.snap_enabled

    def undo(self) -> None:
        if self._rois:
            self._rois.pop()
            self.rois_changed.emit(self._rois)
            self._repaint()

    def reset(self) -> None:
        self._rois.clear()
        self.rois_changed.emit(self._rois)
        self._repaint()

    # ── snap helpers ──────────────────────────────────────────────────────────

    def _snap_threshold(self) -> int:
        """Threshold in video pixels, proportional to last drawn box height."""
        if not self._rois:
            return 0
        x0, y0, x1, y1 = self._rois[-1]
        return max(1, round(abs(y1 - y0) * self._snap_threshold_pct))

    def _avg_width(self) -> Optional[int]:
        """Mean width of all confirmed vials. None if none drawn yet."""
        if not self._rois:
            return None
        return round(sum(abs(r[2] - r[0]) for r in self._rois) / len(self._rois))

    def _h_guide_ys(self) -> Tuple[Optional[int], Optional[int]]:
        """Top-y and bottom-y of the last confirmed vial (video coords)."""
        if not self._rois:
            return None, None
        x0, y0, x1, y1 = self._rois[-1]
        return min(y0, y1), max(y0, y1)

    def _v_guide_x(self) -> Optional[int]:
        """
        Projected left-edge x for the next vial based on inferred gap.
        Requires ≥2 confirmed vials.
        """
        if len(self._rois) < 2:
            return None
        prev = self._rois[-2]
        last = self._rois[-1]
        # left/right edges of each box
        prev_x1 = max(prev[0], prev[2])
        last_x0 = min(last[0], last[2])
        last_x1 = max(last[0], last[2])
        gap = last_x0 - prev_x1
        return last_x1 + gap

    def _snap_on_press(self, v0: QPoint):
        """
        Snap the drag anchor's x to the vertical gap guide at press time.
        Returns (snapped_v0, active_v_guide).
        The anchor is fixed for the entire drag — this is the only place x is snapped.
        """
        if not self.snap_enabled:
            return v0, None
        v_guide = self._v_guide_x()
        if v_guide is None:
            return v0, None
        thresh = self._snap_threshold()
        if abs(v0.x() - v_guide) <= thresh:
            return QPoint(v_guide, v0.y()), v_guide
        return v0, None

    def _snap_on_move(self, v1: QPoint):
        """
        Snap the moving corner's y to the nearest horizontal height guide,
        and its x to the width suggestion line (p0_vid.x ± avg_width).
        Returns (snapped_v1, active_h_guides).
        Only v1 is touched — the anchor (_p0_vid) is never modified.
        """
        if not self.snap_enabled:
            return v1, []
        thresh = self._snap_threshold()

        # ── snap y to h-guides ────────────────────────────────────────────
        top_guide, bot_guide = self._h_guide_ys()
        y = v1.y()
        active_h: List[int] = []
        for guide in (g for g in (top_guide, bot_guide) if g is not None):
            if abs(y - guide) <= thresh:
                y = guide
                active_h.append(guide)
                break   # snap to at most one guide per frame

        # ── snap x to width suggestion ────────────────────────────────────
        x = v1.x()
        avg_w = self._avg_width()
        if avg_w is not None and self._p0_vid is not None:
            for candidate in (self._p0_vid.x() + avg_w, self._p0_vid.x() - avg_w):
                if abs(x - candidate) <= thresh:
                    x = candidate
                    break

        return QPoint(x, y), active_h

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _tfm(self):
        """(scale, ox, oy): label_pos = video_pos * scale + offset."""
        if self._raw is None:
            return 1.0, 0.0, 0.0
        s  = min(self.width() / self._vw, self.height() / self._vh)
        ox = (self.width()  - self._vw * s) / 2
        oy = (self.height() - self._vh * s) / 2
        return s, ox, oy

    def _to_vid(self, pt: QPoint) -> QPoint:
        s, ox, oy = self._tfm()
        return QPoint(
            max(0, min(int((pt.x() - ox) / s), self._vw - 1)),
            max(0, min(int((pt.y() - oy) / s), self._vh - 1)),
        )

    def _to_lbl(self, vx: int, vy: int) -> QPoint:
        s, ox, oy = self._tfm()
        return QPoint(int(vx * s + ox), int(vy * s + oy))

    # ── painting ──────────────────────────────────────────────────────────────

    def _repaint(self) -> None:
        if self._raw is None:
            return
        s, ox, oy = self._tfm()
        scaled = self._raw.scaled(
            int(self._vw * s), int(self._vh * s),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor("#11111b"))

        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawPixmap(int(ox), int(oy), scaled)

        # ── snap guide lines ──────────────────────────────────────────────
        # Always drawn when snap is enabled (passive) so the user can see
        # where to position the cursor before pressing.
        # Active guides (cursor near, or snapping during drag) render at 2px;
        # passive guides render at 1px.
        if self.snap_enabled:
            thresh = self._snap_threshold()
            cur    = self._cursor_vid  # may be None

            # horizontal height guides — solid
            top_guide, bot_guide = self._h_guide_ys()
            for gy in (g for g in (top_guide, bot_guide) if g is not None):
                lp = self._to_lbl(0, gy)
                active = (gy in self._active_h_guides) or (
                    not self._drawing and cur is not None
                    and abs(cur.y() - gy) <= thresh
                )
                p.setPen(QPen(QColor(_COLOUR_GUIDE), 2 if active else 1, Qt.SolidLine))
                p.drawLine(0, lp.y(), self.width(), lp.y())

            # vertical gap guide — dashed
            v_guide = self._v_guide_x()
            if v_guide is not None:
                lp = self._to_lbl(v_guide, 0)
                active = (self._active_v_guide == v_guide) or (
                    not self._drawing and cur is not None
                    and abs(cur.x() - v_guide) <= thresh
                )
                p.setPen(QPen(QColor(_COLOUR_GUIDE), 2 if active else 1, Qt.DashLine))
                p.drawLine(lp.x(), 0, lp.x(), self.height())

            # width suggestion guides — dashed, only during drag
            if self._drawing and self._p0_vid is not None:
                avg_w = self._avg_width()
                if avg_w is not None:
                    cur_x = self._to_vid(self._p1).x() if self._p1 else None
                    for wx in (self._p0_vid.x() + avg_w, self._p0_vid.x() - avg_w):
                        if not (0 <= wx <= self._vw):
                            continue
                        lp = self._to_lbl(wx, 0)
                        active = cur_x is not None and abs(cur_x - wx) <= thresh
                        p.setPen(QPen(QColor(_COLOUR_GUIDE), 2 if active else 1, Qt.DashLine))
                        p.drawLine(lp.x(), 0, lp.x(), self.height())

        # ── confirmed ROIs ────────────────────────────────────────────────
        for idx, (x0, y0, x1, y1) in enumerate(self._rois):
            colour = self._colour_for(idx)
            a = self._to_lbl(x0, y0)
            b = self._to_lbl(x1, y1)
            p.setPen(QPen(QColor(colour), 2))
            p.drawRect(QRect(a, b).normalized())
            font = QFont("Segoe UI", 11)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor(colour))
            label = str(idx + 1)
            if idx < len(self._fly_counts):
                label = f"{idx + 1}  [{self._fly_counts[idx]} flies]"
            p.drawText(a.x() + 6, a.y() + 20, label)

        # ── in-progress drag — dashed peach ──────────────────────────────
        if self._drawing and self._p0 and self._p1:
            p.setPen(QPen(QColor("#fab387"), 2, Qt.DashLine))
            p.drawRect(QRect(self._p0, self._p1).normalized())

        p.end()
        self.setPixmap(canvas)

    # ── events ────────────────────────────────────────────────────────────────

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drawing = True
            v0 = self._to_vid(e.pos())
            # snap anchor x to gap guide once at press time — never again
            v0, self._active_v_guide = self._snap_on_press(v0)
            self._p0_vid = v0
            self._p0     = self._to_lbl(v0.x(), v0.y())
            self._p1     = self._p0
            self._active_h_guides = []

    def mouseMoveEvent(self, e):
        self._cursor_vid = self._to_vid(e.pos())
        if self._drawing:
            v1 = self._cursor_vid
            v1, self._active_h_guides = self._snap_on_move(v1)
            self._p1 = self._to_lbl(v1.x(), v1.y())
            # _p0 / _p0_vid are never touched here
        self._repaint()

    def leaveEvent(self, e):
        self._cursor_vid = None
        self._repaint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            v1 = self._to_vid(e.pos())
            v1, _ = self._snap_on_move(v1)
            x0, x1 = sorted([self._p0_vid.x(), v1.x()])
            y0, y1 = sorted([self._p0_vid.y(), v1.y()])
            if x1 - x0 > 5 and y1 - y0 > 5:
                self._rois.append((x0, y0, x1, y1))
                self.rois_changed.emit(self._rois)
            self._active_h_guides = []
            self._active_v_guide  = None
            self._p0 = self._p1 = self._p0_vid = None
            self._repaint()


# ─── Per-vial control row (fly count + colour swatch) ─────────────────────────

class _VialControlRow(QWidget):
    """One vial's controls: a fly-count spinbox and a colour swatch button.

    Shared by the standalone vial dialog (below) and the setup window
    (scripts/app.py) so both offer identical per-vial fly-count + colour picking.
    """

    changed = pyqtSignal()

    def __init__(self, index: int, default_count: int, default_color: str, parent=None):
        super().__init__(parent)
        self.color = default_color
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        lbl = QLabel(f"Vial {index + 1}:")
        lbl.setObjectName("status")
        row.addWidget(lbl)

        self.spin = QSpinBox()
        self.spin.setRange(1, 200)
        self.spin.setValue(default_count)
        self.spin.valueChanged.connect(lambda _v: self.changed.emit())
        row.addWidget(self.spin)

        self.swatch = QPushButton()
        self.swatch.setFixedWidth(34)
        self.swatch.setToolTip("Pick this vial's colour")
        self.swatch.clicked.connect(self._pick_color)
        row.addWidget(self.swatch)
        self._restyle()

    def _restyle(self) -> None:
        self.swatch.setStyleSheet(
            f"background:{self.color}; border:1px solid #45475a; border-radius:4px;"
        )

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(self.color), self, "Pick vial colour")
        if chosen.isValid():
            self.color = chosen.name()
            self._restyle()
            self.changed.emit()

    def count(self) -> int:
        return self.spin.value()


# ─── Dialog ───────────────────────────────────────────────────────────────────

class _VialROIDialog(QDialog):
    def __init__(self, frame_bgr: np.ndarray,
                 snap_threshold_pct: float, snap_enabled: bool,
                 default_fly_count: int = 7,
                 video_context: VideoContext | None = None,
                 parent=None):
        super().__init__(parent)
        self._snap_threshold_pct = snap_threshold_pct
        self._snap_enabled       = snap_enabled
        self._default_fly_count  = default_fly_count
        self.video_context       = video_context
        self._fc_rows: List[_VialControlRow] = []   # one control row per vial
        self._build()
        self.canvas.set_frame(frame_bgr)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.show()
        self.raise_()
        self.activateWindow()

    def _build(self) -> None:
        self.setWindowTitle("Draw Vial ROIs")
        self.setMinimumSize(1100, 800)
        self.resize(1380, 940)
        self.setStyleSheet(_QSS)
        self.setWindowTitle(build_window_title("Vial ROI", self.video_context))

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        hdr = QLabel("DRAW VIAL ROIs")
        hdr.setObjectName("header")
        root.addWidget(hdr)

        if self.video_context is not None:
            root.addWidget(build_context_chips(self.video_context))

        self.canvas = _MultiROICanvas(self._snap_threshold_pct,
                                      self._snap_enabled)
        self.canvas.rois_changed.connect(self._on_rois_changed)
        self.canvas.snap_toggled.connect(self._on_snap_toggled)
        root.addWidget(self.canvas, 1)

        self.status_lbl = QLabel()
        self.status_lbl.setObjectName("status")
        self._refresh_status(0)
        root.addWidget(self.status_lbl)

        # ── Fly count row (hidden until first ROI is drawn) ───────────────
        self._fc_container = QWidget()
        self._fc_layout = QHBoxLayout(self._fc_container)
        self._fc_layout.setContentsMargins(0, 2, 0, 2)
        self._fc_layout.setSpacing(10)
        _lbl = QLabel("Per vial (flies + colour):")
        _lbl.setObjectName("status")
        self._fc_layout.addWidget(_lbl)
        self._fc_layout.addStretch()
        self._fc_container.setVisible(False)
        root.addWidget(self._fc_container)

        btn_row = QHBoxLayout()
        self.btn_undo  = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_reset.clicked.connect(self.canvas.reset)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        self.btn_done = QPushButton("Done ->")
        self.btn_done.setObjectName("done")
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_done)
        root.addLayout(btn_row)

    def _snap_badge(self) -> str:
        return "[snap ON]" if self.canvas.snap_enabled else "[snap OFF]"

    def _refresh_status(self, n: int) -> None:
        action = "ready — press Done when finished" if n > 0 else "drag to add"
        self.status_lbl.setText(
            f"{n} ROI{'s' if n != 1 else ''} drawn  —  {action}"
            f"    {self._snap_badge()}  (S to toggle)"
        )

    def _sync_fly_spinboxes(self, n: int) -> None:
        """Keep one control row (fly count + colour swatch) per drawn ROI."""
        # Remove extras (from the end, inserted before the stretch)
        while len(self._fc_rows) > n:
            row = self._fc_rows.pop()
            row.deleteLater()

        # Add new ones, seeding each with the default palette colour
        while len(self._fc_rows) < n:
            idx = len(self._fc_rows)
            color = _VIAL_COLOURS[idx % len(_VIAL_COLOURS)]
            row = _VialControlRow(idx, self._default_fly_count, color)
            row.changed.connect(self._update_canvas_counts)
            # Insert before the trailing stretch (last item)
            pos = self._fc_layout.count() - 1
            self._fc_layout.insertWidget(pos, row)
            self._fc_rows.append(row)

        self._fc_container.setVisible(n > 0)
        self._update_canvas_counts()

    def _update_canvas_counts(self) -> None:
        self.canvas.set_fly_counts([r.count() for r in self._fc_rows])
        self.canvas.set_vial_colors([r.color for r in self._fc_rows])

    def _on_rois_changed(self, rois: list) -> None:
        n = len(rois)
        self._refresh_status(n)
        self._sync_fly_spinboxes(n)
        self.btn_done.setEnabled(n > 0)
        self.btn_undo.setEnabled(n > 0)

    def _on_snap_toggled(self, enabled: bool) -> None:
        self._refresh_status(len(self.canvas.get_rois()))

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.btn_done.isEnabled():
                self.accept()
        elif e.key() == Qt.Key_Escape:
            self.reject()
        elif e.key() == Qt.Key_U:
            self.canvas.undo()
        elif e.key() == Qt.Key_R:
            self.canvas.reset()
        elif e.key() == Qt.Key_S:
            self.canvas.toggle_snap()
        else:
            super().keyPressEvent(e)

    def get_rois(self):
        return self.canvas.get_rois()

    def get_fly_counts(self) -> List[int]:
        """Per-vial fly count in draw order (same order as get_rois())."""
        return [r.count() for r in self._fc_rows]

    def get_fly_colors(self) -> List[str]:
        """Per-vial colour hex in draw order (same order as get_rois())."""
        return [r.color for r in self._fc_rows]


# ─── Public API ───────────────────────────────────────────────────────────────

def draw_and_save_vial_rois(
    video_path: str,
    roi_json_path: str,
    frame_idx: int = 0,
    video_context: VideoContext | None = None,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Interactive PyQt5 GUI to manually draw rectangular ROIs for fly vials.

    The user draws as many ROIs as the experiment has vials; the count is
    determined entirely by what was drawn. Downstream steps read the saved
    JSON to learn how many vials exist.

    Controls
    --------
    Drag mouse  : draw ROI
    U           : undo last ROI
    R           : reset all ROIs
    Enter       : finish (requires at least one ROI)
    Esc         : cancel

    Parameters
    ----------
    video_path    : path to the experiment video
    roi_json_path : where the ROIs will be saved as JSON
    frame_idx     : reference frame for drawing (default 0)

    Returns
    -------
    Dict mapping vial IDs to (x0, y0, x1, y1), sorted left -> right.
    """
    library_path = Path(__file__).parent.parent / "roi_library.json"
    entry = {}
    if library_path.exists():
        with open(library_path) as f:
            entry = json.load(f).get(Path(video_path).stem, {}) or {}

    crop = entry.get("preprocessing")
    raw_path = entry.get("video_path") if crop else None
    source_path = raw_path if raw_path and os.path.exists(raw_path) else video_path

    cap = cv2.VideoCapture(source_path)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Could not read reference frame from video")

    if crop and source_path == raw_path:
        cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
        frame = frame[cy:cy + ch, cx:cx + cw]
        if frame.shape[0] != ch or frame.shape[1] != cw:
            raise ValueError("Crop region from roi_library.json is out of bounds for the raw video.")

    app = QApplication.instance() or QApplication(sys.argv)

    cfg                = _roi_cfg()
    snap_threshold_pct = cfg.get("snap_threshold_pct", 0.02)
    snap_enabled       = cfg.get("snap_enabled", True)
    fly_count_default  = _default_fly_count()

    dlg      = _VialROIDialog(
        frame,
        snap_threshold_pct,
        snap_enabled,
        default_fly_count=fly_count_default,
        video_context=video_context,
    )
    accepted = dlg.exec_()

    if not accepted:
        raise RuntimeError("ROI selection cancelled")

    rois       = dlg.get_rois()
    fly_counts = dlg.get_fly_counts()   # aligned with rois (draw order)
    fly_colors = dlg.get_fly_colors()   # aligned with rois (draw order)

    # Sort left-to-right by centre x, keeping fly counts + colours aligned
    indexed_sorted = sorted(enumerate(rois), key=lambda t: (t[1][0] + t[1][2]) / 2.0)

    roi_dict  = {}
    save_data = {}
    for i, (orig_idx, r) in enumerate(indexed_sorted, start=1):
        key = f"vial{i}"
        roi_dict[key]  = tuple(r)
        save_data[key] = {
            "bbox": list(r),
            "n_flies": fly_counts[orig_idx],
            "color": fly_colors[orig_idx],
        }

    os.makedirs(os.path.dirname(roi_json_path) or ".", exist_ok=True)
    with open(roi_json_path, "w") as f:
        json.dump(save_data, f, indent=2)

    # Canonical genotype colours (best-effort) so classification / embedding
    # follow the colours picked here, exactly like the setup window does.
    try:
        from src.plot_colors import parse_vial_genotypes, write_genotype_color_overrides
        genos = parse_vial_genotypes(video_path)
        if genos:
            ordered_colors = [save_data[f"vial{i + 1}"]["color"] for i in range(len(save_data))]
            mapping = {g: c for g, c in zip(genos, ordered_colors)}
            if mapping:
                write_genotype_color_overrides(mapping)
    except Exception as _e:
        print(f"[warn] genotype colour update skipped: {_e}")

    print("Saved ROIs to:", roi_json_path)
    return roi_dict


def load_vial_rois(
    path: str,
) -> Tuple[Dict[str, Tuple[int, int, int, int]], Dict[str, int]]:
    """
    Load vial ROIs from a JSON file, handling both old and new formats.

    Old format: ``{"vial1": [x0, y0, x1, y1], ...}``
    New format: ``{"vial1": {"bbox": [x0, y0, x1, y1], "n_flies": 7}, ...}``

    Returns
    -------
    bbox_dict   : {vial_id: (x0, y0, x1, y1)}
    n_flies_dict: {vial_id: int}  — 0 for vials loaded from old-format files
    """
    with open(path) as f:
        raw = json.load(f)
    bbox_dict: Dict[str, Tuple[int, int, int, int]] = {}
    n_flies_dict: Dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            bbox_dict[k]    = tuple(map(int, v["bbox"]))
            n_flies_dict[k] = int(v.get("n_flies", 0))
        else:
            bbox_dict[k]    = tuple(map(int, v))
            n_flies_dict[k] = 0
    return bbox_dict, n_flies_dict


def load_vial_colors(path: str) -> Dict[str, str]:
    """Return {vial_id: hex_colour} for vials that carry a stored colour.

    Colours are written per vial by the setup window (scripts/app.py) into the
    ``color`` field of the new-format ROI JSON. Old-format files (bare bbox
    lists, or dicts without ``color``) yield an empty / partial map; callers
    fall back to the default palette for any vial missing here.
    """
    with open(path) as f:
        raw = json.load(f)
    colors: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(v.get("color"), str):
            colors[k] = v["color"]
    return colors


def resolve_vial_expected_counts(
    n_flies_dict: Dict[str, int],
    vial_ids,
    fallback_per_vial: int,
) -> Dict[str, int]:
    """Expected fly count per vial: the real ROI count where present, else the
    config fallback.

    ``n_flies`` is a per-run label (drawn in the ROI GUI); the ROI library only
    stores box geometry, so a reused ROI has no count. Per vial: use the real
    count when it is > 0, otherwise ``fallback_per_vial``. This is the single
    definition every entrypoint uses, both to gate ghost detection
    (vial_expected_counts) and to compute the diagnostics expected total, so the
    notebook and the two scripts can never disagree.
    """
    return {
        vid: (int(n_flies_dict.get(vid, 0)) or int(fallback_per_vial))
        for vid in vial_ids
    }


def assign_ordered_ids_left_to_right(
    df: pd.DataFrame,
    id_col: str = "orig_id",
) -> pd.DataFrame:
    """Assign ordered IDs based on left->right median x ordering."""
    df = df.copy()
    x_rep   = df.groupby(id_col)["x"].median().sort_values()
    mapping = {sid: i + 1 for i, sid in enumerate(x_rep.index)}
    df["ordered_id"] = df[id_col].map(mapping).astype(int)
    return df


def assign_vials_and_ordered_ids(
    ocsort_csv: str,
    roi_json: str,
    out_csv: str,
    invert_y: bool = False,
    video_h: Optional[int] = None,
    fps: Optional[float] = None,
):
    """
    Assign vial IDs using rectangular ROIs, then ordered IDs within each vial.

    Parameters
    ----------
    ocsort_csv   : long-format OC-SORT tracks CSV. Must have columns: frame, x, y, orig_id.
    roi_json     : JSON file produced by draw_and_save_vial_rois().
    out_csv      : output path for the ordered_tracks CSV.
    invert_y     : flip y coordinates (needed if tracker and video have different origins).
    video_h      : video height in pixels (required when invert_y=True).
    fps          : frames per second — appended as a constant column for downstream use.

    Returns
    -------
    pd.DataFrame — the ordered_tracks DataFrame (also saved to out_csv).
    """
    vial_rois, _ = load_vial_rois(roi_json)

    def assign_vial(x, y):
        for vid, (x0, y0, x1, y1) in vial_rois.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return vid
        return None

    df = pd.read_csv(ocsort_csv)
    df["frame"] = df["frame"].astype(int)

    # Support both direct OC-SORT output (orig_id) and legacy stitched format (stitched_id)
    id_col = "stitched_id" if "stitched_id" in df.columns else "orig_id"

    y_use        = (video_h - 1 - df["y"]) if invert_y else df["y"]
    df["vial_id"] = [assign_vial(x, y) for x, y in zip(df["x"], y_use)]
    df            = df[df["vial_id"].notna()].copy()

    df["ordered_id"] = -1
    offset = 0
    for vial, g in df.groupby("vial_id", sort=True):
        x_rep   = g.groupby(id_col)["x"].median().sort_values()
        mapping = {sid: offset + i + 1 for i, sid in enumerate(x_rep.index)}
        df.loc[g.index, "ordered_id"] = g[id_col].map(mapping).astype(int)
        offset += len(x_rep)

    if fps is not None:
        df["fps"] = float(fps)

    df.to_csv(out_csv, index=False)
    return df
