"""
src/roi.py

Interactive vial ROI drawing + assignment of compact (sequential) fly IDs.

Workflow
--------
1. draw_and_save_vial_rois()       — one-time manual annotation per experiment setup
2. assign_vials_and_compact_ids()  — assign vial labels + compact IDs to each point
"""

import json
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout,
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap


# ─── Stylesheet (Catppuccin Mocha) ────────────────────────────────────────────

_QSS = """
QDialog, QWidget {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#header {
    font-size: 15px;
    font-weight: bold;
    color: #89b4fa;
    padding: 4px 0;
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

class _MultiROICanvas(QLabel):
    """
    QLabel that renders a video frame and lets the user drag multiple ROIs.
    Each confirmed ROI is drawn in a distinct colour and labelled with its index.
    """

    rois_changed = pyqtSignal(list)   # emits current list of (x0, y0, x1, y1)

    def __init__(self, parent=None):
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
        self._rois     = []    # confirmed (x0, y0, x1, y1) in video coords
        self._drawing  = False
        self._p0       = None  # drag start in label coords
        self._p1       = None  # drag current in label coords

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

    def undo(self) -> None:
        if self._rois:
            self._rois.pop()
            self.rois_changed.emit(self._rois)
            self._repaint()

    def reset(self) -> None:
        self._rois.clear()
        self.rois_changed.emit(self._rois)
        self._repaint()

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

        # Confirmed ROIs — each in its own colour
        for idx, (x0, y0, x1, y1) in enumerate(self._rois):
            colour = _VIAL_COLOURS[idx % len(_VIAL_COLOURS)]
            a = self._to_lbl(x0, y0)
            b = self._to_lbl(x1, y1)
            p.setPen(QPen(QColor(colour), 2))
            p.drawRect(QRect(a, b).normalized())
            font = QFont("Segoe UI", 11)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor(colour))
            p.drawText(a.x() + 6, a.y() + 20, str(idx + 1))

        # In-progress drag — dashed peach
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
            self._p0 = self._p1 = e.pos()

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._p1 = e.pos()
            self._repaint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            v0 = self._to_vid(self._p0)
            v1 = self._to_vid(self._p1)
            x0, x1 = sorted([v0.x(), v1.x()])
            y0, y1 = sorted([v0.y(), v1.y()])
            if x1 - x0 > 5 and y1 - y0 > 5:   # ignore accidental single clicks
                self._rois.append((x0, y0, x1, y1))
                self.rois_changed.emit(self._rois)
            self._p0 = self._p1 = None
            self._repaint()


# ─── Dialog ───────────────────────────────────────────────────────────────────

class _VialROIDialog(QDialog):
    def __init__(self, frame_bgr: np.ndarray, n_vials: int, parent=None):
        super().__init__(parent)
        self.n_vials = n_vials
        self._build()
        self.canvas.set_frame(frame_bgr)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.raise_()
        self.activateWindow()

    def _build(self) -> None:
        self.setWindowTitle("Draw Vial ROIs")
        self.setMinimumSize(1100, 800)
        self.resize(1380, 940)
        self.setStyleSheet(_QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        hdr = QLabel("DRAW VIAL ROIs")
        hdr.setObjectName("header")
        root.addWidget(hdr)

        self.canvas = _MultiROICanvas()
        self.canvas.rois_changed.connect(self._on_rois_changed)
        root.addWidget(self.canvas, 1)

        self.status_lbl = QLabel(
            f"0 / {self.n_vials} ROIs drawn  —  drag to add"
        )
        self.status_lbl.setObjectName("status")
        root.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        self.btn_undo  = QPushButton("Undo")
        self.btn_reset = QPushButton("Reset")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_reset.clicked.connect(self.canvas.reset)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        self.btn_done = QPushButton("Done  →")
        self.btn_done.setObjectName("done")
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_done)
        root.addLayout(btn_row)

    def _on_rois_changed(self, rois: list) -> None:
        n = len(rois)
        suffix = "  —  ready" if n == self.n_vials else "  —  drag to add"
        self.status_lbl.setText(f"{n} / {self.n_vials} ROIs drawn{suffix}")
        self.btn_done.setEnabled(n == self.n_vials)
        self.btn_undo.setEnabled(n > 0)

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
        else:
            super().keyPressEvent(e)

    def get_rois(self):
        return self.canvas.get_rois()


# ─── Public API ───────────────────────────────────────────────────────────────

def draw_and_save_vial_rois(
    video_path: str,
    roi_json_path: str,
    frame_idx: int = 0,
    n_vials: int = 6,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Interactive PyQt5 GUI to manually draw rectangular ROIs for fly vials.

    Controls
    --------
    Drag mouse  : draw ROI
    U           : undo last ROI
    R           : reset all ROIs
    Enter       : finish (only when exactly n_vials ROIs are drawn)
    Esc         : cancel

    Parameters
    ----------
    video_path    : path to the experiment video
    roi_json_path : where the ROIs will be saved as JSON
    frame_idx     : reference frame for drawing (default 0)
    n_vials       : number of vials expected (default 6)

    Returns
    -------
    Dict mapping vial IDs to (x0, y0, x1, y1), sorted left -> right.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Could not read reference frame from video")

    app = QApplication.instance() or QApplication(sys.argv)

    dlg      = _VialROIDialog(frame, n_vials)
    accepted = dlg.exec_()

    if not accepted:
        raise RuntimeError("ROI selection cancelled")

    rois = dlg.get_rois()
    rois_sorted = sorted(rois, key=lambda r: (r[0] + r[2]) / 2.0)
    roi_dict    = {f"vial{i}": tuple(r) for i, r in enumerate(rois_sorted, start=1)}

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
    x_rep   = df.groupby(id_col)["x"].median().sort_values()
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
    stitched_csv : long-format stitched CSV from stitch().
                   Must have columns: frame, orig_id, x, y, stitched_id.
    roi_json     : JSON file produced by draw_and_save_vial_rois().
    out_csv      : output path for the compact_tracks CSV.
    invert_y     : flip y coordinates (needed if tracker and video have different origins).
    video_h      : video height in pixels (required when invert_y=True).
    fps          : frames per second — appended as a constant column for downstream use.

    Returns
    -------
    pd.DataFrame — the compact_tracks DataFrame (also saved to out_csv).
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

    y_use        = (video_h - 1 - df["y"]) if invert_y else df["y"]
    df["vial_id"] = [assign_vial(x, y) for x, y in zip(df["x"], y_use)]
    df            = df[df["vial_id"].notna()].copy()

    df["compact_id"] = -1
    offset = 0
    for vial, g in df.groupby("vial_id", sort=True):
        x_rep   = g.groupby("stitched_id")["x"].median().sort_values()
        mapping = {sid: offset + i + 1 for i, sid in enumerate(x_rep.index)}
        df.loc[g.index, "compact_id"] = g["stitched_id"].map(mapping).astype(int)
        offset += len(x_rep)

    if fps is not None:
        df["fps"] = float(fps)

    df.to_csv(out_csv, index=False)
    return df
