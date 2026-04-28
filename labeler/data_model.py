"""Data model for the labeler.

Loads detection bbox cache (the source of truth) and the OC-SORT wide-format
tracking output (suggestions only). Maintains the in-memory annotation state
keyed by (frame, det_idx).

CSV formats consumed
--------------------
Raw detection cache (`src/tracking.py` writes this):
    columns: frame, x1, y1, x2, y2, conf

OC-SORT wide tracking output (`src/tracking.py` writes this):
    columns: frame, id1, id2, ...
    each cell is a string like "(142.50, 88.20)" or empty
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


SOURCE_OCSORT = "ocsort"
SOURCE_HUMAN = "human"
SOURCE_HUMAN_SYNTH = "human_synth"   # human placed a synthetic detection (detector miss)

NN_MATCH_TOLERANCE_PX = 5.0


# ---------------------------------------------------------------------------
# Detection / annotation records
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection in a single frame.

    `det_idx` is the position inside its frame's detection list (sorted by
    x then y for stable ordering across reloads). Synthetic detections use
    *negative* det_idx so they never collide with real ones.
    """
    frame: int
    det_idx: int
    x: float           # centroid
    y: float
    x1: float          # bbox
    y1: float
    x2: float
    y2: float
    conf: float
    is_synthetic: bool = False   # human-placed (detector miss); affects export source


@dataclass
class Annotation:
    track_id: int
    source: str        # SOURCE_OCSORT or SOURCE_HUMAN


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw_detections(csv_path: str) -> dict[int, list[Detection]]:
    """Load the bbox detection cache and group by frame.

    Within each frame, detections are sorted by (x, y) so `det_idx` is stable
    across reloads of the same CSV.
    """
    df = pd.read_csv(csv_path)
    required = {"frame", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw detection CSV missing columns: {sorted(missing)}")
    if "conf" not in df.columns:
        df = df.assign(conf=np.nan)

    df = df.copy()
    df["x"] = (df["x1"] + df["x2"]) / 2.0
    df["y"] = (df["y1"] + df["y2"]) / 2.0

    by_frame: dict[int, list[Detection]] = {}
    for frame, grp in df.groupby("frame"):
        grp = grp.sort_values(["x", "y"]).reset_index(drop=True)
        by_frame[int(frame)] = [
            Detection(
                frame=int(frame),
                det_idx=i,
                x=float(r.x), y=float(r.y),
                x1=float(r.x1), y1=float(r.y1),
                x2=float(r.x2), y2=float(r.y2),
                conf=float(r.conf) if not pd.isna(r.conf) else float("nan"),
            )
            for i, r in grp.iterrows()
        ]
    return by_frame


_XY_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def _parse_xy_cell(cell) -> Optional[tuple[float, float]]:
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return None
    m = _XY_RE.match(s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def load_ocsort_wide(csv_path: str) -> dict[int, list[tuple[int, float, float]]]:
    """Read OC-SORT wide-format CSV and melt to per-frame (track_id, x, y) lists.

    Returns: {frame: [(track_id, x, y), ...]}
    """
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise ValueError("OC-SORT CSV missing 'frame' column")

    id_cols = [c for c in df.columns if c != "frame"]
    parsed_ids: dict[str, int] = {}
    for c in id_cols:
        m = re.match(r"id(\d+)$", c)
        if m:
            parsed_ids[c] = int(m.group(1))

    out: dict[int, list[tuple[int, float, float]]] = {}
    for _, row in df.iterrows():
        frame = int(row["frame"])
        entries: list[tuple[int, float, float]] = []
        for c, tid in parsed_ids.items():
            xy = _parse_xy_cell(row[c])
            if xy is None:
                continue
            entries.append((tid, xy[0], xy[1]))
        out[frame] = entries
    return out


# ---------------------------------------------------------------------------
# Matching OC-SORT suggestions to raw detections
# ---------------------------------------------------------------------------

def match_ocsort_to_raw(
    raw_by_frame: dict[int, list[Detection]],
    ocsort_by_frame: dict[int, list[tuple[int, float, float]]],
    tolerance_px: float = NN_MATCH_TOLERANCE_PX,
) -> dict[tuple[int, int], Annotation]:
    """For each OC-SORT (track_id, x, y), find the nearest raw detection in
    the same frame within `tolerance_px`. Each raw detection can be claimed
    at most once (greedy by ascending distance).

    Returns annotations keyed by (frame, det_idx) with source=SOURCE_OCSORT.
    """
    annotations: dict[tuple[int, int], Annotation] = {}

    for frame, suggestions in ocsort_by_frame.items():
        raw_dets = raw_by_frame.get(frame, [])
        if not raw_dets or not suggestions:
            continue

        raw_xy = np.array([(d.x, d.y) for d in raw_dets])

        candidates: list[tuple[float, int, int]] = []
        for s_i, (_tid, sx, sy) in enumerate(suggestions):
            d2 = (raw_xy[:, 0] - sx) ** 2 + (raw_xy[:, 1] - sy) ** 2
            for r_i, dist2 in enumerate(d2):
                dist = float(np.sqrt(dist2))
                if dist <= tolerance_px:
                    candidates.append((dist, s_i, r_i))
        candidates.sort()

        used_raw: set[int] = set()
        used_sug: set[int] = set()
        for dist, s_i, r_i in candidates:
            if s_i in used_sug or r_i in used_raw:
                continue
            tid = suggestions[s_i][0]
            det = raw_dets[r_i]
            annotations[(det.frame, det.det_idx)] = Annotation(
                track_id=tid, source=SOURCE_OCSORT,
            )
            used_sug.add(s_i)
            used_raw.add(r_i)

    return annotations


# ---------------------------------------------------------------------------
# AnnotationStore — the mutable in-memory state
# ---------------------------------------------------------------------------

@dataclass
class _UndoOp:
    key: tuple[int, int]
    prev: Optional[Annotation]   # None means it didn't exist before


class AnnotationStore:
    """In-memory annotation state with undo.

    Keys are (frame, det_idx). Values are Annotation. The detection
    coordinates themselves live in `raw_by_frame` and are never mutated.
    """

    def __init__(
        self,
        raw_by_frame: dict[int, list[Detection]],
        seed: Optional[dict[tuple[int, int], Annotation]] = None,
    ):
        self.raw_by_frame = raw_by_frame
        self._anns: dict[tuple[int, int], Annotation] = dict(seed or {})
        self._undo: list[_UndoOp] = []

    # ---- queries ----

    def get(self, frame: int, det_idx: int) -> Optional[Annotation]:
        return self._anns.get((frame, det_idx))

    def all(self) -> dict[tuple[int, int], Annotation]:
        return dict(self._anns)

    def track_ids(self) -> list[int]:
        return sorted({a.track_id for a in self._anns.values()})

    # ---- mutations (all push undo) ----

    def assign(self, frame: int, det_idx: int, track_id: int,
               source: str = SOURCE_HUMAN) -> None:
        if (frame, det_idx) not in self._index_set():
            raise KeyError(f"no raw detection at frame={frame}, det_idx={det_idx}")
        key = (frame, det_idx)
        self._undo.append(_UndoOp(key, self._anns.get(key)))
        self._anns[key] = Annotation(track_id=int(track_id), source=source)

    def clear(self, frame: int, det_idx: int) -> None:
        key = (frame, det_idx)
        if key in self._anns:
            self._undo.append(_UndoOp(key, self._anns[key]))
            del self._anns[key]

    def merge(self, track_a: int, track_b: int) -> None:
        """Reassign every detection currently labeled `track_b` to `track_a`."""
        for key, ann in list(self._anns.items()):
            if ann.track_id == track_b:
                self._undo.append(_UndoOp(key, ann))
                self._anns[key] = Annotation(track_id=track_a, source=SOURCE_HUMAN)

    def split(self, track_id: int, from_frame: int, new_track_id: int) -> None:
        """Reassign `track_id` to `new_track_id` for all frames >= from_frame."""
        for key, ann in list(self._anns.items()):
            f, _ = key
            if ann.track_id == track_id and f >= from_frame:
                self._undo.append(_UndoOp(key, ann))
                self._anns[key] = Annotation(track_id=new_track_id, source=SOURCE_HUMAN)

    def undo(self) -> bool:
        if not self._undo:
            return False
        op = self._undo.pop()
        if op.prev is None:
            self._anns.pop(op.key, None)
        else:
            self._anns[op.key] = op.prev
        return True

    # ---- export ----

    def export_long_csv(self, path: str) -> int:
        """Write `frame, ID, x, y` rows for every annotated detection.

        Includes both human-confirmed and OC-SORT-sourced annotations — once
        an OC-SORT suggestion is in the store, it counts as accepted unless
        the human cleared or overrode it.
        Returns the number of rows written.
        """
        rows = []
        for (frame, det_idx), ann in self._anns.items():
            det = self._lookup(frame, det_idx)
            rows.append({
                "frame": frame,
                "ID": ann.track_id,
                "x": det.x,
                "y": det.y,
            })
        df = pd.DataFrame(rows, columns=["frame", "ID", "x", "y"])
        df = df.sort_values(["frame", "ID"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        return len(df)

    # ---- internals ----

    def _lookup(self, frame: int, det_idx: int) -> Detection:
        # Search by det_idx field, not list position — synthetics have
        # negative det_idx and aren't at their nominal index.
        for d in self.raw_by_frame.get(frame, []):
            if d.det_idx == det_idx:
                return d
        raise KeyError(f"no detection at frame={frame}, det_idx={det_idx}")

    def _index_set(self) -> set[tuple[int, int]]:
        return {(f, d.det_idx) for f, dets in self.raw_by_frame.items() for d in dets}
