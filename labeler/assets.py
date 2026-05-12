"""Folder-population helpers: copy source assets into the per-video labeling
folder, melt wide-format tracks into long, write metadata.json, write the
export QC summary.

Goal: a labeling folder is self-contained. Six months from now anyone can
open it and reproduce / inspect / pass it to a colleague without hunting
down the original video or detection cache.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .robo_export import (
    SOURCE_HUMAN,
    SOURCE_OCSORT,
    load_tracks_any,
)

METADATA_FILENAME = "metadata.json"
TRACKS_LONG_FILENAME = "tracks_long.csv"
DETECTIONS_RAW_FILENAME = "detections_raw.csv"


# ---------------------------------------------------------------------------
# File copying
# ---------------------------------------------------------------------------

def _copy_if_missing(src: Path, dst: Path) -> bool:
    """Copy `src` to `dst` only if `dst` doesn't already exist (or is empty).
    Returns True iff a copy was performed.
    """
    if dst.exists() and dst.stat().st_size > 0:
        return False
    if src.resolve() == dst.resolve():
        return False
    shutil.copy2(src, dst)
    return True


def populate_folder(
    out_dir: Path,
    *,
    video_path: Path,
    raw_csv: Path,
    tracks_csv: Optional[Path],
) -> dict:
    """Copy source assets into out_dir and produce a canonical long-format
    `tracks_long.csv` from the input tracks CSV (wide or long, auto-detected).

    Returns a dict describing what was copied/written, used by metadata.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    actions: dict = {"copied": [], "skipped": [], "wrote": []}

    # 1. Video → out_dir/<original-filename>
    video_dst = out_dir / video_path.name
    (actions["copied"] if _copy_if_missing(video_path, video_dst) else actions["skipped"]).append(
        video_dst.name
    )

    # 2. detections_raw.csv (canonical name regardless of source filename)
    raw_dst = out_dir / DETECTIONS_RAW_FILENAME
    (actions["copied"] if _copy_if_missing(raw_csv, raw_dst) else actions["skipped"]).append(
        raw_dst.name
    )

    # 3. tracks_long.csv: canonical long-format version of the input tracks CSV
    if tracks_csv is not None:
        long_dst = out_dir / TRACKS_LONG_FILENAME
        if not (long_dst.exists() and long_dst.stat().st_size > 0):
            _write_tracks_long(tracks_csv, long_dst)
            actions["wrote"].append(long_dst.name)
        else:
            actions["skipped"].append(long_dst.name)

    return actions


def _write_tracks_long(src_csv: Path, long_dst: Path) -> None:
    """Convert a tracks CSV (wide or long) to canonical long-format
    `frame, ID, x, y`. Format is auto-detected from the source columns."""
    by_frame = load_tracks_any(str(src_csv))
    rows = []
    for frame in sorted(by_frame.keys()):
        for tid, x, y in by_frame[frame]:
            rows.append({"frame": frame, "ID": tid, "x": x, "y": y})
    pd.DataFrame(rows, columns=["frame", "ID", "x", "y"]).to_csv(long_dst, index=False)


# ---------------------------------------------------------------------------
# metadata.json
# ---------------------------------------------------------------------------

def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def init_metadata(
    out_dir: Path,
    *,
    repo_root: Path,
    video_path: Path,
    raw_csv: Path,
    tracks_csv: Optional[Path],
    video_props: dict,
    copy_actions: dict,
) -> Path:
    """Write the initial metadata.json. If one already exists, preserve the
    `created_at` field so we don't overwrite the original session start time.
    """
    md_path = out_dir / METADATA_FILENAME
    now = _iso_now()

    existing = {}
    if md_path.exists():
        try:
            existing = json.loads(md_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    payload = {
        "schema_version": 2,
        "created_at": existing.get("created_at", now),
        "last_opened_at": now,
        "last_saved_at": existing.get("last_saved_at"),
        "last_exported_at": existing.get("last_exported_at"),
        "labeler_git_commit": _git_commit(repo_root),
        "sources": {
            "video": str(video_path.as_posix()),
            "raw_detections_csv": str(raw_csv.as_posix()),
            "tracks_csv": str(tracks_csv.as_posix()) if tracks_csv else None,
        },
        "video": video_props,  # {frame_count, fps, width, height}
        "copy_actions": copy_actions,
        "annotation_counts": existing.get("annotation_counts", {
            "total": 0, "human": 0, "ocsort": 0, "tracks": 0,
        }),
    }
    md_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path


def update_metadata_counts(
    out_dir: Path, *, total: int, human: int, ocsort: int, tracks: int,
    saved: bool = False, exported: bool = False,
) -> None:
    """Update the live-state fields of metadata.json without disturbing
    immutable fields. Best-effort; never raise."""
    md_path = out_dir / METADATA_FILENAME
    if not md_path.exists():
        return
    try:
        payload = json.loads(md_path.read_text(encoding="utf-8"))
    except Exception:
        return

    payload["annotation_counts"] = {
        "total": int(total), "human": int(human),
        "ocsort": int(ocsort), "tracks": int(tracks),
    }
    if saved:
        payload["last_saved_at"] = _iso_now()
    if exported:
        payload["last_exported_at"] = _iso_now()

    try:
        md_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Export summary (Ctrl+E)
# ---------------------------------------------------------------------------

def write_export_summary(
    summary_path: Path,
    *,
    annotations: dict,                       # {(frame, det_idx): Annotation}
    raw_by_frame: dict,                      # {frame: [Detection]}
    video_props: dict,
    export_csv_name: str,
) -> None:
    """Write a short human-readable QC summary next to the GT CSV.

    Records: per-track counts, frame coverage, frames missing any annotation,
    mean detection conf on annotated rows.
    """
    n_total_anns = len(annotations)
    n_human = sum(1 for a in annotations.values() if a.source == SOURCE_HUMAN)
    n_ocsort = n_total_anns - n_human

    per_track: dict[int, int] = {}
    for ann in annotations.values():
        per_track[ann.track_id] = per_track.get(ann.track_id, 0) + 1

    n_frames = video_props.get("frame_count", 0) or max(raw_by_frame.keys(), default=-1) + 1

    # Annotated detections: collect their conf for a mean
    confs = []
    annotated_keys = set(annotations.keys())
    for frame, dets in raw_by_frame.items():
        for d in dets:
            if (frame, d.det_idx) in annotated_keys and d.conf == d.conf:  # not NaN
                confs.append(float(d.conf))
    mean_conf = sum(confs) / len(confs) if confs else float("nan")

    # Frames where no detection got annotated (potential missed coverage)
    bare_frames = []
    for frame in range(n_frames):
        dets = raw_by_frame.get(frame, [])
        if not dets:
            continue  # frame has no detections at all — not a coverage gap
        if not any((frame, d.det_idx) in annotated_keys for d in dets):
            bare_frames.append(frame)

    lines: list[str] = []
    lines.append(f"# Ground-truth export summary — {_iso_now()}")
    lines.append(f"# CSV: {export_csv_name}")
    lines.append("")
    lines.append(f"frames in video        : {n_frames}")
    lines.append(f"total annotations      : {n_total_anns}")
    lines.append(f"  human-confirmed      : {n_human}")
    lines.append(f"  ocsort-sourced       : {n_ocsort}")
    lines.append(f"distinct track IDs     : {len(per_track)}")
    lines.append(f"mean det conf (annot.) : {mean_conf:.3f}" if confs else
                 "mean det conf (annot.) : n/a")
    lines.append("")
    lines.append("per-track counts:")
    for tid in sorted(per_track.keys()):
        lines.append(f"  track {tid:>3}  →  {per_track[tid]} frames")
    lines.append("")
    lines.append(f"frames with detections but ZERO annotations: {len(bare_frames)}")
    if bare_frames:
        preview = ", ".join(str(f) for f in bare_frames[:30])
        more = f"  (+{len(bare_frames) - 30} more)" if len(bare_frames) > 30 else ""
        lines.append(f"  e.g. {preview}{more}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
