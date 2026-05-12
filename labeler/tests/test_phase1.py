"""Phase 1 unit tests for the labeler data layer.

Run from repo root:
    python -m pytest labeler/tests/test_phase1.py -v
or:
    python -m unittest labeler.tests.test_phase1
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from labeler.color_engine import track_color_hex, track_color_rgb
from labeler.robo_export import (
    Annotation,
    AnnotationStore,
    SOURCE_HUMAN,
    SOURCE_OCSORT,
    load_ocsort_wide,
    load_raw_detections,
    match_ocsort_to_raw,
)
from labeler.session import (
    annotations_from_payload,
    choose_resume_path,
    load_session,
    save_session,
)


def _write_raw(tmp: Path) -> Path:
    """Two frames, three detections in frame 0, two in frame 1."""
    df = pd.DataFrame([
        # frame 0: three flies
        {"frame": 0, "x1": 100, "y1": 200, "x2": 110, "y2": 210, "conf": 0.9},  # centroid (105, 205)
        {"frame": 0, "x1": 300, "y1": 50,  "x2": 310, "y2": 60,  "conf": 0.8},  # (305, 55)
        {"frame": 0, "x1": 50,  "y1": 400, "x2": 60,  "y2": 410, "conf": 0.7},  # (55, 405)
        # frame 1: two flies
        {"frame": 1, "x1": 102, "y1": 198, "x2": 112, "y2": 208, "conf": 0.85}, # (107, 203)
        {"frame": 1, "x1": 308, "y1": 52,  "x2": 318, "y2": 62,  "conf": 0.75}, # (313, 57)
    ])
    p = tmp / "raw.csv"
    df.to_csv(p, index=False)
    return p


def _write_ocsort_wide(tmp: Path) -> Path:
    """Wide-format OC-SORT output covering both frames.

    id1 → matches (105,205)/(107,203), id2 → matches (305,55)/(313,57).
    Frame 0 also has id3 at (55,405). Frame 1 drops id3 entirely.
    """
    df = pd.DataFrame([
        {"frame": 0, "id1": "(106.0, 205.5)", "id2": "(304.0, 56.0)", "id3": "(55.0, 405.0)"},
        {"frame": 1, "id1": "(108.0, 203.5)", "id2": "(312.0, 57.5)", "id3": ""},
    ])
    p = tmp / "ocsort_wide.csv"
    df.to_csv(p, index=False)
    return p


class TestColorEngine(unittest.TestCase):
    def test_returns_valid_hex(self):
        h = track_color_hex(1)
        self.assertEqual(len(h), 7)
        self.assertTrue(h.startswith("#"))
        int(h[1:], 16)  # parses

    def test_rgb_in_range(self):
        for tid in (0, 1, 5, 10, 255, 9999):
            r, g, b = track_color_rgb(tid)
            for c in (r, g, b):
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 255)

    def test_stable_modulo(self):
        self.assertEqual(track_color_hex(1), track_color_hex(1 + 256))


class TestRawLoader(unittest.TestCase):
    def test_loads_and_groups(self):
        with TemporaryDirectory() as td:
            p = _write_raw(Path(td))
            by_frame = load_raw_detections(str(p))
        self.assertEqual(set(by_frame.keys()), {0, 1})
        self.assertEqual(len(by_frame[0]), 3)
        self.assertEqual(len(by_frame[1]), 2)

    def test_centroid_computed(self):
        with TemporaryDirectory() as td:
            p = _write_raw(Path(td))
            by_frame = load_raw_detections(str(p))
        # frame 0 sorted by x then y → first centroid should be (55, 405)
        self.assertAlmostEqual(by_frame[0][0].x, 55.0)
        self.assertAlmostEqual(by_frame[0][0].y, 405.0)

    def test_det_idx_stable(self):
        with TemporaryDirectory() as td:
            p = _write_raw(Path(td))
            a = load_raw_detections(str(p))
            b = load_raw_detections(str(p))
        for f in a:
            self.assertEqual([d.x for d in a[f]], [d.x for d in b[f]])
            self.assertEqual([d.det_idx for d in a[f]], [d.det_idx for d in b[f]])

    def test_missing_columns_raises(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            pd.DataFrame({"frame": [0], "x1": [1.0]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_raw_detections(str(p))


class TestOcsortLoader(unittest.TestCase):
    def test_melts_and_drops_blanks(self):
        with TemporaryDirectory() as td:
            p = _write_ocsort_wide(Path(td))
            by_frame = load_ocsort_wide(str(p))
        self.assertEqual(len(by_frame[0]), 3)
        # frame 1: id3 was empty → dropped
        self.assertEqual(len(by_frame[1]), 2)
        ids_f1 = {tid for tid, _, _ in by_frame[1]}
        self.assertEqual(ids_f1, {1, 2})

    def test_parse_xy_floats(self):
        with TemporaryDirectory() as td:
            p = _write_ocsort_wide(Path(td))
            by_frame = load_ocsort_wide(str(p))
        # frame 0, id1 → (106.0, 205.5)
        entry = next(e for e in by_frame[0] if e[0] == 1)
        self.assertEqual(entry[1], 106.0)
        self.assertEqual(entry[2], 205.5)


class TestMatching(unittest.TestCase):
    def test_matches_within_tolerance(self):
        with TemporaryDirectory() as td:
            raw = load_raw_detections(str(_write_raw(Path(td))))
            ocs = load_ocsort_wide(str(_write_ocsort_wide(Path(td))))
        anns = match_ocsort_to_raw(raw, ocs, tolerance_px=5.0)
        # All 3 frame-0 suggestions match, both frame-1 suggestions match → 5 total
        self.assertEqual(len(anns), 5)
        for ann in anns.values():
            self.assertEqual(ann.source, SOURCE_OCSORT)

    def test_unmatched_outside_tolerance(self):
        raw = {0: load_raw_detections.__wrapped__ if False else None}  # unused
        # Hand-build a tiny case: one raw det at (100,100), suggestion at (200,200)
        from labeler.robo_export import Detection
        raw_by_frame = {0: [Detection(0, 0, 100, 100, 95, 95, 105, 105, 0.9)]}
        ocs_by_frame = {0: [(7, 200.0, 200.0)]}
        anns = match_ocsort_to_raw(raw_by_frame, ocs_by_frame, tolerance_px=5.0)
        self.assertEqual(len(anns), 0)

    def test_greedy_nearest_first(self):
        from labeler.robo_export import Detection
        # Two raw dets near each other; one suggestion much closer to det 0.
        raw_by_frame = {0: [
            Detection(0, 0, 100, 100, 95, 95, 105, 105, 0.9),
            Detection(0, 1, 103, 100, 98, 95, 108, 105, 0.9),
        ]}
        ocs_by_frame = {0: [(5, 100.5, 100.0), (6, 103.0, 100.0)]}
        anns = match_ocsort_to_raw(raw_by_frame, ocs_by_frame, tolerance_px=5.0)
        # Closest pair (sug 0 → raw 0) wins, then sug 1 → raw 1.
        self.assertEqual(anns[(0, 0)].track_id, 5)
        self.assertEqual(anns[(0, 1)].track_id, 6)


class TestAnnotationStore(unittest.TestCase):
    def setUp(self):
        self.td = TemporaryDirectory()
        td = Path(self.td.name)
        self.raw = load_raw_detections(str(_write_raw(td)))
        ocs = load_ocsort_wide(str(_write_ocsort_wide(td)))
        seed = match_ocsort_to_raw(self.raw, ocs)
        self.store = AnnotationStore(self.raw, seed=seed)

    def tearDown(self):
        self.td.cleanup()

    def test_seeded_from_ocsort(self):
        self.assertEqual(len(self.store.all()), 5)

    def test_assign_creates_human(self):
        # Pick the first detection in frame 0 and assign track 99.
        self.store.assign(0, 0, 99)
        ann = self.store.get(0, 0)
        self.assertEqual(ann.track_id, 99)
        self.assertEqual(ann.source, SOURCE_HUMAN)

    def test_assign_unknown_detection_raises(self):
        with self.assertRaises(KeyError):
            self.store.assign(0, 99, 1)

    def test_undo_restores_previous(self):
        before = self.store.get(0, 0)
        self.store.assign(0, 0, 999)
        self.assertTrue(self.store.undo())
        self.assertEqual(self.store.get(0, 0), before)

    def test_undo_removes_brand_new_assignment(self):
        # Build a store with no seed, so every assign is brand new.
        empty = AnnotationStore(self.raw)
        empty.assign(0, 0, 42)
        self.assertIsNotNone(empty.get(0, 0))
        self.assertTrue(empty.undo())
        self.assertIsNone(empty.get(0, 0))

    def test_merge(self):
        # Find two distinct track ids in the seeded store
        tids = self.store.track_ids()
        a, b = tids[0], tids[1]
        n_a_before = sum(1 for ann in self.store.all().values() if ann.track_id == a)
        n_b_before = sum(1 for ann in self.store.all().values() if ann.track_id == b)
        self.store.merge(a, b)
        n_a_after = sum(1 for ann in self.store.all().values() if ann.track_id == a)
        self.assertEqual(n_a_after, n_a_before + n_b_before)

    def test_split(self):
        tids = self.store.track_ids()
        tid = tids[0]
        # Pick a frame >= 1 and split
        self.store.split(tid, from_frame=1, new_track_id=500)
        for (f, _), ann in self.store.all().items():
            if f >= 1 and ann.track_id == 500:
                self.assertEqual(ann.source, SOURCE_HUMAN)

    def test_export_csv(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "gt.csv"
            n = self.store.export_long_csv(str(out))
            df = pd.read_csv(out)
        self.assertEqual(n, len(df))
        self.assertEqual(list(df.columns), ["frame", "ID", "x", "y"])
        # All exported rows correspond to existing annotations
        self.assertEqual(len(df), len(self.store.all()))


class TestSession(unittest.TestCase):
    def test_choose_resume_path_prefers_saved_session_by_default(self):
        with TemporaryDirectory() as td:
            tdp = Path(td)
            session = tdp / "run.labeler.json"
            autosave = tdp / "run.labeler.autosave.json"
            session.write_text("{}")
            autosave.write_text("{}")

            chosen = choose_resume_path(str(session), str(autosave))

        self.assertEqual(chosen, session.as_posix())

    def test_choose_resume_path_can_prefer_autosave(self):
        with TemporaryDirectory() as td:
            tdp = Path(td)
            session = tdp / "run.labeler.json"
            autosave = tdp / "run.labeler.autosave.json"
            session.write_text("{}")
            autosave.write_text("{}")

            chosen = choose_resume_path(
                str(session),
                str(autosave),
                prefer_autosave=True,
            )

        self.assertEqual(chosen, autosave.as_posix())

    def test_choose_resume_path_falls_back_to_autosave(self):
        with TemporaryDirectory() as td:
            tdp = Path(td)
            session = tdp / "run.labeler.json"
            autosave = tdp / "run.labeler.autosave.json"
            autosave.write_text("{}")

            chosen = choose_resume_path(str(session), str(autosave))

        self.assertEqual(chosen, autosave.as_posix())

    def test_round_trip(self):
        with TemporaryDirectory() as td:
            tdp = Path(td)
            raw_path = _write_raw(tdp)
            ocs_path = _write_ocsort_wide(tdp)
            raw = load_raw_detections(str(raw_path))
            ocs = load_ocsort_wide(str(ocs_path))
            store = AnnotationStore(raw, seed=match_ocsort_to_raw(raw, ocs))
            store.assign(0, 0, 17)  # add a human override

            sess_path = tdp / "session.json"
            save_session(
                str(sess_path),
                video_path="/fake/video.mp4",
                raw_csv=str(raw_path),
                ocsort_csv=str(ocs_path),
                current_frame=1,
                current_mode="frame",
                store=store,
            )

            payload = load_session(str(sess_path))
            anns = annotations_from_payload(payload)

        self.assertEqual(payload["current_frame"], 1)
        self.assertEqual(payload["current_mode"], "frame")
        self.assertEqual(len(anns), len(store.all()))
        # Human override survives
        self.assertEqual(anns[(0, 0)].track_id, 17)
        self.assertEqual(anns[(0, 0)].source, SOURCE_HUMAN)

    def test_version_mismatch(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            p.write_text(json.dumps({"version": 999, "annotations": {}}))
            with self.assertRaises(ValueError):
                load_session(str(p))


if __name__ == "__main__":
    unittest.main()
