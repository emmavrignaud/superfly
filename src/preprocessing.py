"""
src/preprocessing.py

Background subtraction + interactive ROI/range GUI.

Optional first step: makes flies pop against the background before tracking.
GUI controls:
  - Drag mouse on the video to draw a ROI
  - Sliders for start / end frame and preview frame
  - Accept button (or Enter) to confirm, Cancel (or Esc) to abort

Background method: temporal median (FreeClimber-style).
  - Collects sampled ROI frames into an array, then np.median(axis=0).
  - Resistant to transient objects (fly must occupy a pixel >50% of sampled
    frames to influence the background estimate).
  - Uses signed subtraction (no one-sided clamp) so both polarities survive
    into the output.
"""

import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path

from src.ui_context import VideoContext, build_context_chips, build_window_title

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap


# ─── Config loader ────────────────────────────────────────────────────────

def _preprocessing_cfg() -> dict:
    p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f).get("preprocessing", {})


# ─── Stylesheet ───────────────────────────────────────────────────────────
_QSS = (Path(__file__).parent / "preprocessing_style.qss").read_text()

_STYLE_ROI_NONE = (
    "color:#f38ba8; background:#313244; border-radius:4px; padding:5px 12px;"
)
_STYLE_ROI_SET = (
    "color:#a6e3a1; background:#313244; border-radius:4px; padding:5px 12px;"
)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _bgr_to_gray_float32(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> grayscale float32 (OpenCV ordering)."""
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return 0.1140 * b + 0.5870 * g + 0.2989 * r


def _frame_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # .copy() makes QImage own the buffer so numpy array can be GC'd safely
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ─── Video canvas with drag-to-draw ROI ───────────────────────────────────

class _VideoCanvas(QLabel):
    """QLabel that renders a video frame and lets the user drag a ROI."""

    roi_changed = pyqtSignal(object)  # emits (x, y, w, h) in video coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #11111b; border-radius: 6px;")

        self._raw     = None   # QPixmap of current frame
        self._vw      = 1
        self._vh      = 1
        self._roi     = None   # confirmed (x, y, w, h) in video coords
        self._drawing = False
        self._p0      = None   # drag start in label coords (QPoint)
        self._p1      = None   # drag current in label coords (QPoint)

    # ── public ────────────────────────────────────────────────────────────

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._vh, self._vw = frame_bgr.shape[:2]
        self._raw = _frame_to_pixmap(frame_bgr)
        self._repaint()

    def get_roi(self):
        return self._roi

    def undo(self) -> None:
        self.clear_roi()

    def reset(self) -> None:
        self.clear_roi()

    def clear_roi(self) -> None:
        self._drawing = False
        self._roi = None
        self._p0 = None
        self._p1 = None
        self.roi_changed.emit(None)
        self._repaint()

    # ── coordinate helpers ────────────────────────────────────────────────

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

    # ── painting ──────────────────────────────────────────────────────────

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

        # confirmed ROI — solid green
        if self._roi is not None:
            x, y, w, h = self._roi
            a = self._to_lbl(x, y)
            b = self._to_lbl(x + w, y + h)
            p.setPen(QPen(QColor("#a6e3a1"), 2))
            p.drawRect(QRect(a, b).normalized())

        # in-progress ROI — dashed blue
        if self._drawing and self._p0 and self._p1:
            p.setPen(QPen(QColor("#89b4fa"), 2, Qt.DashLine))
            p.drawRect(QRect(self._p0, self._p1).normalized())

        p.end()
        self.setPixmap(canvas)

    # ── events ────────────────────────────────────────────────────────────

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
            x0 = max(0, min(x0, self._vw - 1))
            y0 = max(0, min(y0, self._vh - 1))
            w  = max(1, min(x1 - x0, self._vw - x0))
            h  = max(1, min(y1 - y0, self._vh - y0))
            self._roi = (x0, y0, w, h)
            self._p0 = self._p1 = None
            self.roi_changed.emit(self._roi)
            self._repaint()


# ─── Labeled slider row ───────────────────────────────────────────────────

class _SliderRow(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(self, label: str, lo: int, hi: int, init: int, parent=None):
        super().__init__(parent)
        lbl = QLabel(label)
        lbl.setObjectName("sliderLabel")

        self._val = QLabel(str(init))
        self._val.setObjectName("sliderValue")
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val.setFixedWidth(55)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(init)
        self.slider.valueChanged.connect(self._emit)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.addWidget(lbl)
        row.addWidget(self.slider, 1)
        row.addWidget(self._val)

    def _emit(self, v: int) -> None:
        self._val.setText(str(v))
        self.value_changed.emit(v)

    def set_value(self, v: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self._val.setText(str(v))
        self.slider.blockSignals(False)

    def value(self) -> int:
        return self.slider.value()


# ─── Main dialog ──────────────────────────────────────────────────────────

class _ROIPickerDialog(QDialog):
    def __init__(self, cap, n_frames: int, w_vid: int, h_vid: int,
                 initial_end: int, video_context: VideoContext | None = None,
                 parent=None):
        super().__init__(parent)
        self.cap      = cap
        self.n_frames = n_frames
        self.w_vid    = w_vid
        self.h_vid    = h_vid
        self.start    = 0
        self.end      = min(initial_end, n_frames) if n_frames > 0 else initial_end
        self.cur      = 0
        self.video_context = video_context

        self._build()
        self._load_frame()

    # ── layout ────────────────────────────────────────────────────────────

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setObjectName("divider")
        return f

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowStaysOnTopHint) & ~Qt.WindowContextHelpButtonHint
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def _build(self) -> None:
        self.setWindowTitle("Preprocessing  —  Pick ROI & Frame Range")
        self.setMinimumSize(1100, 820)
        self.resize(1380, 940)
        self.setStyleSheet(_QSS)
        self.setWindowTitle(build_window_title("Preprocessing", self.video_context))

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        hdr = QLabel("PICK ROI & TRIM VIDEO TO FRAME RANGE")
        hdr.setObjectName("header")
        root.addWidget(hdr)

        if self.video_context is not None:
            root.addWidget(build_context_chips(self.video_context))

        self.canvas = _VideoCanvas()
        self.canvas.roi_changed.connect(self._on_roi)
        root.addWidget(self.canvas, 1)

        self.roi_lbl = QLabel("Draw a ROI by dragging on the video")
        self.roi_lbl.setStyleSheet(_STYLE_ROI_NONE)
        root.addWidget(self.roi_lbl)

        root.addWidget(self._divider())

        hi_n = max(self.n_frames - 1, 1)
        hi_e = max(self.n_frames, 1)
        self.sl_start = _SliderRow("Trim — keep from frame",  0,    hi_n, self.start)
        self.sl_end   = _SliderRow("Trim — keep until frame",  1,    hi_e, self.end)
        self.sl_cur   = _SliderRow("Preview frame",            0,    hi_n, self.cur)
        self.sl_start.value_changed.connect(self._on_start)
        self.sl_end.value_changed.connect(self._on_end)
        self.sl_cur.value_changed.connect(self._on_cur)
        for sl in (self.sl_start, self.sl_end, self.sl_cur):
            root.addWidget(sl)

        self.stats_lbl = QLabel()
        self.stats_lbl.setObjectName("stats")
        self._refresh_stats()
        root.addWidget(self.stats_lbl)

        root.addWidget(self._divider())

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
        self.btn_cancel = QPushButton("Cancel")
        self.btn_accept = QPushButton("Accept ->")
        self.btn_accept.setObjectName("accept")
        self.btn_accept.setEnabled(False)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_accept.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_accept)
        root.addLayout(btn_row)

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_roi(self, roi) -> None:
        if roi is None:
            self.roi_lbl.setText("Draw a ROI by dragging on the video")
            self.roi_lbl.setStyleSheet(_STYLE_ROI_NONE)
            self.btn_accept.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return

        x, y, w, h = roi
        self.roi_lbl.setText(f"ROI   x={x}   y={y}   w={w}   h={h}")
        self.roi_lbl.setStyleSheet(_STYLE_ROI_SET)
        self.btn_accept.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.btn_reset.setEnabled(True)

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
            f"   →   {kept} kept, {discarded} discarded   (total in video: {self.n_frames})"
        )

    def _load_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur)
        ok, frame = self.cap.read()
        if ok:
            self.canvas.set_frame(frame)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.btn_accept.isEnabled():
                self.accept()
        elif e.key() == Qt.Key_Escape:
            self.reject()
        elif e.key() == Qt.Key_U:
            self.canvas.undo()
        elif e.key() == Qt.Key_R:
            self.canvas.reset()
        else:
            super().keyPressEvent(e)

    # ── result ────────────────────────────────────────────────────────────

    def get_result(self):
        x, y, w, h = self.canvas.get_roi()
        return x, y, w, h, self.start, self.end


# ─── Public API ───────────────────────────────────────────────────────────

def gui_pick_roi_and_range(
    video_path: str,
    video_context: VideoContext | None = None,
):
    """
    PyQt5 GUI to pick ROI + [start, end_excl).

    The end-frame slider initializes to the video's actual frame count.
    Final start/end are whatever the user accepts in the GUI.

    Returns:
      (x, y, w, h, start, end_excl)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if QApplication.instance() is None:
        for attr in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
            if hasattr(Qt, attr):
                QApplication.setAttribute(getattr(Qt, attr), True)
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    dlg      = _ROIPickerDialog(
        cap,
        n_frames,
        w_vid,
        h_vid,
        n_frames,
        video_context=video_context,
    )
    accepted = dlg.exec_()
    cap.release()

    if not accepted:
        raise RuntimeError("Cancelled ROI/range selection")

    return dlg.get_result()


def preprocess_bgsub_gui(
    video_path: str,
    out_mp4: str | None = None,
    out_raw_mp4: str | None = None,
    gain: float | None = None,
    white_level: float | None = None,
    codec: str | None = None,
    bg_sample_stride: int | None = None,
    bg_percentile: float | None = None,
    crop_params: dict | None = None,
    video_context: VideoContext | None = None,
) -> tuple[str, dict]:
    """
    GUI-driven ROI/range selection + temporal-median background subtraction.

    Background is computed as the pixel-wise median over sampled frames in
    [start, end_excl).  The median is outlier-resistant: a fly must occupy a
    given pixel for more than half of the sampled frames before it contaminates
    the background estimate.

    Subtraction is signed (bg - gray), so both darker-than-background and
    brighter-than-background signal is preserved.  The output pixel formula is:

        vis = white_level - (bg - gray) * gain

    clipped to [0, 255].  On a backlit rig (dark flies, bright background)
    bg > gray where flies are, giving vis < white_level (dark flies on a pale
    background).

    Output path behaviour:
      - If out_mp4 is None:  "<same folder>/<stem>_pp.<ext>"
      - If out_raw_mp4 is given, also write the cropped raw clip there.
      - Otherwise writes to the given path(s) (parent dir created if needed).

    Parameters
    ----------
    crop_params : optional dict with keys x, y, w, h, start, end.
        If provided, skips the GUI and uses these values directly.
        Use this when re-running a video whose crop region is already known
        (e.g. loaded from the ROI library).

    Returns
    -------
    (out_mp4_path, crop_params) where crop_params is a dict:
        {"x": int, "y": int, "w": int, "h": int, "start": int, "end": int}
    """
    cfg = _preprocessing_cfg()
    if gain           is None: gain           = cfg.get("bg_gain",          1.2)
    if white_level    is None: white_level    = cfg.get("bg_white_level",   245)
    if codec          is None: codec          = cfg.get("codec",          "mp4v")
    if bg_sample_stride is None: bg_sample_stride = cfg.get("bg_sample_stride", 1)
    if bg_percentile  is None: bg_percentile  = cfg.get("bg_percentile",  85.0)

    video_path = str(video_path)
    if crop_params is not None:
        x, y, w, h = crop_params["x"], crop_params["y"], crop_params["w"], crop_params["h"]
        start, end_excl = crop_params["start"], crop_params["end"]
        print(f"Using stored crop params: x={x}, y={y}, w={w}, h={h}, frames={start}–{end_excl}")
    else:
        x, y, w, h, start, end_excl = gui_pick_roi_and_range(
            video_path,
            video_context=video_context,
        )

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

    # 1) Collect sampled frames then compute median background in one shot
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    bg_frames = []
    for f in range(start, end_excl):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if (f - start) % bg_sample_stride != 0:
            continue
        roi_bgr = frame_bgr[y:y + h, x:x + w]
        if roi_bgr.shape[0] != h or roi_bgr.shape[1] != w:
            cap.release()
            raise ValueError("ROI out of bounds during background computation.")
        bg_frames.append(_bgr_to_gray_float32(roi_bgr))

    if not bg_frames:
        cap.release()
        raise RuntimeError("No frames available to compute the median background.")

    bg_gray = np.percentile(np.stack(bg_frames, axis=0), bg_percentile, axis=0).astype(np.float32)

    # 2) Write background-subtracted video and optional raw-cropped companion
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    raw_writer = None
    if out_raw_mp4 is not None:
        os.makedirs(os.path.dirname(out_raw_mp4) or ".", exist_ok=True)
        raw_writer = cv2.VideoWriter(out_raw_mp4, fourcc, fps, (w, h), isColor=True)
        if not raw_writer.isOpened():
            cap.release()
            writer.release()
            raise RuntimeError(f"Could not open VideoWriter for: {out_raw_mp4}")

    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        writer.release()
        raise FileNotFoundError(video_path)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end_excl):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        roi_bgr = frame_bgr[y:y + h, x:x + w]
        if roi_bgr.shape[0] != h or roi_bgr.shape[1] != w:
            cap.release()
            writer.release()
            if raw_writer is not None:
                raw_writer.release()
            raise ValueError("ROI out of bounds for this video/frame.")

        gray = _bgr_to_gray_float32(roi_bgr)
        diff = bg_gray - gray  # signed: positive where fly is darker than bg
        vis  = float(white_level) - diff * float(gain)
        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
        if raw_writer is not None:
            raw_writer.write(roi_bgr)
        writer.write(cv2.cvtColor(vis_u8, cv2.COLOR_GRAY2BGR))

    cap.release()
    writer.release()
    if raw_writer is not None:
        raw_writer.release()
        print("Saved raw cropped video:", out_raw_mp4)
    print("Saved bgsub video:", out_mp4)
    print(f"Background ({bg_percentile}th percentile) from {len(bg_frames)} frames (stride={bg_sample_stride}).")
    return out_mp4, {"x": x, "y": y, "w": w, "h": h, "start": start, "end": end_excl}






