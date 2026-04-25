"""Video frame image provider for QML.

QML requests `image://videoframes/<frame_index>?v=<tick>`. We:
  1. Open the video with OpenCV (lazily, on first request).
  2. Seek to the requested frame.
  3. Convert BGR → RGB → QImage and return.

Synchronous for now. If frame seeks become a felt bottleneck, swap in a
worker-thread prefetcher around the current frame.
"""
from __future__ import annotations

from typing import Optional

import cv2
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


class VideoFrameProvider(QQuickImageProvider):
    def __init__(self, backend):
        super().__init__(QQuickImageProvider.Image)
        self._backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._opened_path: str = ""
        self._cached_idx: int = -1
        self._cached_img: Optional[QImage] = None

    def _ensure_cap(self) -> Optional[cv2.VideoCapture]:
        path = self._backend.videoPath
        if not path:
            return None
        if path != self._opened_path:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(path)
            self._opened_path = path
            self._cached_idx = -1
            self._cached_img = None
        return self._cap

    def requestImage(self, id_str: str, size, requested_size) -> QImage:
        # id_str looks like "<frame>?v=<tick>". Strip the query.
        try:
            frame_idx = int(id_str.split("?")[0])
        except (ValueError, AttributeError):
            return QImage()

        if frame_idx == self._cached_idx and self._cached_img is not None:
            return self._cached_img

        cap = self._ensure_cap()
        if cap is None:
            return QImage()

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return QImage()

        # Note: cv2.CAP_PROP_POS_FRAMES is unreliable on some codecs. If seeks
        # land on the wrong frame for your videos, add a verification step
        # (cap.get(CAP_PROP_POS_FRAMES) == frame_idx + 1) and fall back to
        # sequential reads from the nearest keyframe.

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # .copy() because the underlying numpy buffer is reused on the next read
        img = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        self._cached_idx = frame_idx
        self._cached_img = img
        return img
