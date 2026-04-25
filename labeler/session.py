"""Session save/load — JSON file with annotations, paths, and view state.

Sessions are portable across machines as long as the referenced video and
CSV files exist at the same paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .data_model import Annotation, AnnotationStore, SOURCE_HUMAN, SOURCE_OCSORT


SESSION_VERSION = 1


def _key_to_str(frame: int, det_idx: int) -> str:
    return f"{frame}:{det_idx}"


def _str_to_key(s: str) -> tuple[int, int]:
    f, d = s.split(":")
    return int(f), int(d)


def save_session(
    path: str,
    *,
    video_path: str,
    raw_csv: str,
    ocsort_csv: Optional[str],
    current_frame: int,
    current_mode: str,
    store: AnnotationStore,
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
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session(path: str) -> dict:
    """Returns the raw payload. Caller is responsible for loading the CSVs
    referenced by `raw_csv` / `ocsort_csv` and constructing an AnnotationStore
    via `annotations_from_payload`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != SESSION_VERSION:
        raise ValueError(
            f"unsupported session version: {payload.get('version')!r} "
            f"(expected {SESSION_VERSION})"
        )
    return payload


def annotations_from_payload(payload: dict) -> dict[tuple[int, int], Annotation]:
    out: dict[tuple[int, int], Annotation] = {}
    for k, v in payload.get("annotations", {}).items():
        src = v.get("source", SOURCE_HUMAN)
        if src not in (SOURCE_HUMAN, SOURCE_OCSORT):
            src = SOURCE_HUMAN
        out[_str_to_key(k)] = Annotation(track_id=int(v["track_id"]), source=src)
    return out
