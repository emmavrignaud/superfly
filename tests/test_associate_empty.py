"""
Regression test: OCSort.update must not crash when a frame has zero
detections while live trackers exist.

The bug (fixed in src/association.py) was that the link-cost block in
associate() called ``lc.max()`` on an array of shape (0, n_trk), which
raises ``ValueError: zero-size array to reduction operation maximum``.

This trips on any video where RF-DETR returns no detections for a frame
after at least one track has been spawned, which is common when flies
briefly leave the vials or detection confidence dips.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ocsort import OCSort


IMG_H, IMG_W = 480, 640


def _box(cx: float, cy: float, score: float = 0.9, size: float = 20.0) -> list[float]:
    half = size / 2.0
    return [cx - half, cy - half, cx + half, cy + half, score]


def _step(tracker: OCSort, dets: np.ndarray) -> None:
    tracker.update(dets, [IMG_H, IMG_W], [IMG_H, IMG_W])


def test_empty_detection_frame_after_spawn():
    """One tracker is alive, then a frame has zero detections. No exception."""
    tracker = OCSort(
        det_thresh=0.1,
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        asso_func="diou",
        inertia=0.2,
        behavioral_weight=0.05,
        aspect_weight=0.05,
    )

    for fx in range(5):
        cx = 100.0 + fx * 2.0
        _step(tracker, np.array([_box(cx, 240.0)], dtype=float))

    _step(tracker, np.empty((0, 5), dtype=float))


def test_empty_detection_frame_no_trackers():
    """Zero detections with zero trackers is also fine (sanity check)."""
    tracker = OCSort(det_thresh=0.1, behavioral_weight=0.05)
    _step(tracker, np.empty((0, 5), dtype=float))


def test_empty_then_resume():
    """Tracking should continue normally after an empty-detection frame."""
    tracker = OCSort(
        det_thresh=0.1,
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        asso_func="diou",
        inertia=0.2,
        behavioral_weight=0.05,
    )

    for fx in range(5):
        cx = 100.0 + fx * 2.0
        _step(tracker, np.array([_box(cx, 240.0)], dtype=float))

    _step(tracker, np.empty((0, 5), dtype=float))

    out = tracker.update(
        np.array([_box(112.0, 240.0)], dtype=float),
        [IMG_H, IMG_W],
        [IMG_H, IMG_W],
    )
    assert out is not None


@pytest.mark.parametrize("n_trackers", [1, 3, 7])
def test_empty_detection_frame_scales(n_trackers: int):
    """Same guard holds regardless of how many trackers are alive."""
    tracker = OCSort(
        det_thresh=0.1,
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        asso_func="diou",
        inertia=0.2,
        behavioral_weight=0.05,
    )

    for fx in range(5):
        dets = np.array(
            [_box(80.0 + i * 60.0 + fx * 1.5, 240.0) for i in range(n_trackers)],
            dtype=float,
        )
        _step(tracker, dets)

    _step(tracker, np.empty((0, 5), dtype=float))
