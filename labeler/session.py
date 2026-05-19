"""Session save/load — JSON file with annotations, paths, view state, and
synthetic detections (human-placed dets the detector missed).

Sessions are portable across machines as long as the referenced video and
CSV files exist at the same paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .data_model import (
    Annotation,
    AnnotationStore,
    Detection,
    SOURCE_HUMAN,
    SOURCE_HUMAN_SYNTH,
    SOURCE_OCSORT,
)


SESSION_VERSION = 3  # bumped from 2 to add confirmed_tracks


def _key_to_str(frame: int, det_idx: int) -> str:
    return f"{frame}:{det_idx}"


def _str_to_key(s: str) -> tuple[int, int]:
    f, d = s.split(":")
    return int(f), int(d)


def choose_resume_path(
    session_path: str,
    autosave_path: str,
    *,
    prefer_autosave: bool = False,
) -> str:
    """Pick the session file to resume from.

    Safe default: prefer the explicit save file when both exist. Autosave is
    only used by default if no saved session exists, or when the caller
    explicitly opts into autosave-first behavior.
    """
    session = Path(session_path)
    autosave = Path(autosave_path)

    if prefer_autosave and autosave.exists():
        return str(autosave.as_posix())
    if session.exists():
        return str(session.as_posix())
    if autosave.exists():
        return str(autosave.as_posix())
    return ""


def save_session(
    path: str,
    *,
    video_path: str,
    raw_csv: str,
    ocsort_csv: Optional[str],
    current_frame: int,
    current_mode: str,
    store: AnnotationStore,
    synthetic_detections: Optional[list[Detection]] = None,
    confirmed_tracks: Optional[list[int]] = None,
) -> None:
    payload = {
        "version": SESSION_VERSION,
        "video_path": str(Path(video_path).as_posix()),
        "raw_csv": str(Path(raw_csv).as_posix()),
        "ocsort_csv": str(Path(ocsort_csv).as_posix()) if ocsort_csv else None,
        "current_frame": int(current_frame),
        "current_mode": current_mode,
        "annotations": {
            _key_to_str(f, d): {"track_id": ann.track_id, "source": ann.source}
            for (f, d), ann in store.all().items()
        },
        "synthetic_detections": [
            {
                "frame": d.frame, "det_idx": d.det_idx,
                "x": d.x, "y": d.y,
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            }
            for d in (synthetic_detections or [])
        ],
        "confirmed_tracks": list(confirmed_tracks or []),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session(path: str) -> dict:
    """Returns the raw payload. Caller is responsible for loading the CSVs
    referenced by `raw_csv` / `ocsort_csv` and constructing an AnnotationStore
    via `annotations_from_payload`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    v = payload.get("version")
    if v not in (1, 2, 3):
        raise ValueError(f"unsupported session version: {v!r} (expected 1, 2, or 3)")
    # Older sessions just lack these keys — fill in defaults.
    payload.setdefault("synthetic_detections", [])
    payload.setdefault("confirmed_tracks", [])
    return payload


def annotations_from_payload(payload: dict) -> dict[tuple[int, int], Annotation]:
    valid_sources = (SOURCE_HUMAN, SOURCE_OCSORT, SOURCE_HUMAN_SYNTH)
    out: dict[tuple[int, int], Annotation] = {}
    for k, v in payload.get("annotations", {}).items():
        src = v.get("source", SOURCE_HUMAN)
        if src not in valid_sources:
            src = SOURCE_HUMAN
        out[_str_to_key(k)] = Annotation(track_id=int(v["track_id"]), source=src)
    return out


def synthetics_from_payload(payload: dict) -> list[Detection]:
    out: list[Detection] = []
    for r in payload.get("synthetic_detections", []):
        out.append(Detection(
            frame=int(r["frame"]), det_idx=int(r["det_idx"]),
            x=float(r["x"]), y=float(r["y"]),
            x1=float(r["x1"]), y1=float(r["y1"]),
            x2=float(r["x2"]), y2=float(r["y2"]),
            conf=float("nan"), is_synthetic=True,
        ))
    return out
